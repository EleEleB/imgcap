import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutput
from transformers import CLIPVisionConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from torch.nn import CrossEntropyLoss
import sacrebleu
import re
from typing import Dict, Any
from tqdm.auto import tqdm
import numpy as np
import json

class PrefixedLLMConfig:
    def __init__(self, encoder, decoder):
        self.encoder = encoder.config
        self.decoder = decoder.config

import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class PrefixedLLM(PreTrainedModel, GenerationMixin):
    def __init__(self, encoder, decoder):
        super().__init__(decoder.config)
        self.encoder = encoder
        self.config.encoder = encoder.config
        self.decoder = decoder
        self.config.decoder = decoder.config
        self.generation_config = GenerationConfig()

        self.vision_text_proj = nn.Linear(
            self.encoder.config.hidden_size,
            self.decoder.config.hidden_size
        )

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.decoder.set_input_embeddings(value)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        shifted_tokens = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        return shifted_tokens

    def encode_image(self, pixel_values):
        vision_output = self.encoder(pixel_values)
        return self.vision_text_proj(vision_output.last_hidden_state)  # [B, P, H]

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_embeds=None,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        **kwargs
    ):
        # ---- Cached decoding step (>0): no prefix prepend ----
        if past_key_values is not None:
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("In cached decoding, provide input_ids or inputs_embeds.")
                inputs_embeds = self.decoder.get_input_embeddings()(input_ids)

            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )
            return CausalLMOutputWithPast(
                loss=None,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # ---- First step (no cache): build prefix + text ----
        prefix_len = 0

        if inputs_embeds is None:
            if image_embeds is None:
                if pixel_values is None:
                    raise ValueError("Provide pixel_values or image_embeds on the first step.")
                image_embeds = self.encode_image(pixel_values)  # [B, P, H]

            prefix_len = image_embeds.size(1)

            if input_ids is None:
                bos = torch.full(
                    (image_embeds.size(0), 1),
                    self.config.decoder_start_token_id,
                    device=image_embeds.device,
                    dtype=torch.long
                )
                input_ids = bos

            text_embeds = self.decoder.get_input_embeddings()(input_ids)  # [B, T, H]
            inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # [B, P+T, H]
        else:
            # If caller passes inputs_embeds directly, you must also pass a consistent attention_mask.
            # Can't infer prefix_len reliably here.
            prefix_len = 0

        # Make attention_mask match inputs_embeds length
        B, S = inputs_embeds.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((B, S), device=inputs_embeds.device, dtype=torch.long)
        elif attention_mask.size(1) != S:
            pad_len = S - attention_mask.size(1)
            if pad_len > 0:
                prefix_mask = torch.ones((B, pad_len), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                attention_mask = attention_mask[:, :S]
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs
        )

        logits = outputs.logits

        # IMPORTANT: during training, drop prefix positions so logits length matches labels length
        # Condition: we actually prepended a prefix (prefix_len>0) and we have text input_ids.
        if prefix_len > 0 and input_ids is not None:
            logits = logits[:, prefix_len:, :]  # [B, T, V]

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@torch.inference_mode()
def greedy_generate_prefixed(
    model,
    pixel_values,
    max_new_tokens=64,
):
    device = pixel_values.device
    bsz = pixel_values.size(0)

    # Encode image once (projected to decoder hidden size)
    image_embeds = model.encode_image(pixel_values)  # [B, P, H]
    prefix_len = image_embeds.size(1)

    bos_id = model.config.bos_token_id
    if bos_id is None:
        bos_id = getattr(model.config, "decoder_start_token_id", None)
    if bos_id is None:
        raise ValueError("Need bos_token_id or decoder_start_token_id in config.")

    eos_id = model.config.eos_token_id
    pad_id = model.config.pad_token_id
    if pad_id is None:
        pad_id = 0

    # Start tokens: [B, 1]
    generated = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device)

    past_key_values = None
    finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    # Step 0 attention mask covers prefix + current tokens (BOS)
    attention_mask = torch.ones((bsz, prefix_len + generated.size(1)), dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        if past_key_values is None:
            out = model(
                input_ids=generated,         # BOS at start (or full prompt if you have one)
                image_embeds=image_embeds,   # use precomputed vision prefix
                attention_mask=attention_mask,
                use_cache=True,
            )
        else:
            out = model(
                input_ids=generated[:, -1:],  # only last token
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # After EOS: keep padding (or keep EOS if you prefer)
        next_token = torch.where(
            finished.unsqueeze(-1),
            torch.full_like(next_token, pad_id),
            next_token
        )

        generated = torch.cat([generated, next_token], dim=1)

        # Update finished flags
        if eos_id is not None:
            finished = finished | (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break

        # Update cache + attention mask for next step
        past_key_values = out.past_key_values
        attention_mask = torch.cat(
            [attention_mask, torch.ones((bsz, 1), dtype=attention_mask.dtype, device=device)],
            dim=1
        )

    return generated

# shift input ids one token to the right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):

    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def init_decoder_input_ids(eos_token_id, bos_token_id, batch_size):
    eos = torch.tensor([eos_token_id] * batch_size).unsqueeze(-1)
    bos = torch.tensor([bos_token_id] * batch_size).unsqueeze(-1)
    decoder_input_ids = torch.cat([eos, bos], dim = 1)
    decoder_input_ids = decoder_input_ids.expand(batch_size, 2)
    return decoder_input_ids

def print_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def print_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print('total_params', total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable_params', trainable_params)

def _norm(s: str) -> str:
    """Normalizes string by lowering case and collapsing whitespace."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

@torch.no_grad()
def evaluate_and_save_predictions(
    model,
    dataloader,
    tokenizer,
    device,
    output_json_path: str,
    train_config: Dict[str, Any],
    max_new_tokens: int = 16,
    num_beams: int = 1,
):
    """
    Performs inference on the dataloader, computes metrics, and saves results to JSON.
    """
    model.eval()
    # Handle tokenizer wrapper vs direct tokenizer instance
    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    rows = []
    all_hyps = []
    all_refs = []

    print(f"Starting evaluation on device: {device}")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="eval+save")):
        batch = {k: v.to(device) for k, v in batch.items()}

        pixel_values = batch["pixel_values"]
        labels = batch.get("labels", None)

        # Optional: Debugging block to inspect top-5 token probabilities at the first step
        # (Retained from original snippet logic)
        if batch_idx == 0:
            if train_config["model_type"] != "prefix":
                out = model(pixel_values=pixel_values)
            else:
                out = model(input_ids=None, pixel_values=pixel_values) # don't pass hidden states to the prefix model
            logits = out.logits[0, -1]
            top = torch.topk(logits, 5)
            print("\nTop 5 initial tokens (Debug):")
            for i in range(5):
                print(f"Token: {tok.decode([top.indices[i].item()])}, Score: {top.values[i].item():.4f}")

        # Generate captions
        gen_ids = greedy_generate_prefixed(
            model=model,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
        )
        hyp_texts = tok.batch_decode(gen_ids, skip_special_tokens=True)

        # Decode references if labels exist
        if labels is not None:
            labels_for_decode = labels.clone()
            # Replace -100 with pad_token_id to allow decoding
            labels_for_decode[labels_for_decode == -100] = tok.pad_token_id
            ref_texts = tok.batch_decode(labels_for_decode, skip_special_tokens=True)
        else:
            ref_texts = [None] * len(hyp_texts)

        # Process batch results
        for i, (hyp, ref) in enumerate(zip(hyp_texts, ref_texts)):
            hyp_n = _norm(hyp)
            ref_n = _norm(ref) if ref is not None else None

            row = {
                "id": batch_idx * len(hyp_texts) + i,
                "prediction": hyp,
                "prediction_norm": hyp_n,
                "reference": ref,
                "reference_norm": ref_n,
            }

            if ref_n is not None:
                row["exact_match"] = (hyp_n == ref_n)
                # Note: sacrebleu expects a list of references for each hypothesis
                row["sentence_bleu"] = sacrebleu.sentence_bleu(hyp_n, [ref_n]).score

                all_hyps.append(hyp_n)
                all_refs.append(ref_n)

            rows.append(row)

    # Compute aggregate metrics
    metrics = {}
    if all_hyps:
        metrics["exact_match_acc"] = float(np.mean([r["exact_match"] for r in rows if r["reference_norm"] is not None]))
        metrics["bleu"] = sacrebleu.corpus_bleu(all_hyps, [all_refs]).score
        metrics["avg_sentence_bleu"] = float(np.mean([r["sentence_bleu"] for r in rows if r.get("sentence_bleu") is not None]))

    # Construct payload
    payload = {
        "metrics": metrics,
        "predictions": rows,
    }

    # Save to file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return metrics, output_json_path