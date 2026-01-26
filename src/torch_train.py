import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
                AutoTokenizer,
                default_data_collator,
                VisionEncoderDecoderModel,
                VisionEncoderDecoderConfig,
                ViTImageProcessor,
                TrainingArguments,
                Trainer,
                EarlyStoppingCallback,
                AutoProcessor,
                ViTConfig,
                GPT2Config,
                BertConfig,
                )
from tqdm import tqdm
import sacrebleu
from transformers import CLIPModel
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from lib_data_utils import prepare_dataset, PrecomputedTensorDataset
from lib_sys_utils import get_current_time_string
from lib_model import print_params, print_trainable_parameters
import os
import json

train_config = {
    "data_path": "data/train_processed_en_vit-gpt2-coco-en_vit-gpt2-coco-en.pt",
    # "data_path": "data/toy_colors.pt",
    "num_epochs": 3,
    "num_steps": -1,
    "learning_rate": 5e-5,
    "batch_size_train": 8,
    "batch_size_eval": 8,
    "lang": "en",
}


model_checkpoint = f"./models/{get_current_time_string()}" # path of the model checkpoint to load (only used if the previous line is True)

model_name = "ydshieh/vit-gpt2-coco-en"
# model_name = "google-bert/bert-base-uncased"

save_model_to = f"./models/{model_name.split('/')[-1]}_{get_current_time_string()}" # path where to save the fine-tuned model

# load model
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = PrecomputedTensorDataset(train_config['data_path'], limit_n=0, shuffle = True, seed = 42)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# encoder_config = ViTConfig()
# decoder_config = GPT2Config()
# decoder_config = BertConfig()

# config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
# model = VisionEncoderDecoderModel(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print_params(model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.vocab_size = model.config.decoder.vocab_size

model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
model.generation_config.pad_token_id = tokenizer.eos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id

# eval_dataset = PrecomputedTensorDataset(eval_dataset_path, limit_n=0, shuffle = False, seed = 42)

train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size_train'], shuffle=True, collate_fn=default_data_collator)

optimizer = AdamW(params = model.parameters(), lr = train_config['learning_rate'])
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
train_iter = iter(train_loader)

if train_config['num_steps'] == -1:
    train_config['num_steps'] = len(train_dataset) * train_config['num_epochs']
tbar = tqdm(total=train_config['num_steps'])

for step in range(train_config['num_steps']):
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    # Move to device
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer.zero_grad()

    # 1. Prepare Decoder Inputs
    # This automatically shifts labels right and adds start_token_id
    # Labels: [A, B, C, EOS] -> Decoder_Input: [EOS, A, B, C]
    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(batch['labels'])
    
    # 2. Forward Pass (NO LABELS)
    # We deliberately do NOT pass 'labels=' to avoid the internal library shift.
    labels = batch['labels'] # Shape: [batch, seq_len]
    outputs = model(
        pixel_values=batch['pixel_values'],
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=batch.get('decoder_attention_mask', None), # Optional if using default collator
        return_dict=True,
    )
    
    # 3. Manual Loss Calculation
    logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]
    
    # Verify Alignment:
    # logits[0] (from EOS) predicts labels[0] ('a') -> Correct
    
    # Flatten for CrossEntropy
    # Logits: (Batch * Seq_Len, Vocab)
    # Labels: (Batch * Seq_Len)
    loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    # import pdb; pdb.set_trace()
    loss.backward()
    optimizer.step()

    tbar.set_description(f"Loss: {loss.item():.3f}")
    tbar.update(1)

# save model and clip_processor + tokenizer
model.save_pretrained(save_model_to)
feature_extractor.save_pretrained(save_model_to)
tokenizer.save_pretrained(save_model_to)

print(f"model saved at: {save_model_to}") # DEBUGGING

json_path = os.path.join(save_model_to, 'train_config.json')

with open(json_path, 'w', encoding='utf8') as f:
    json.dump(train_config, f, ensure_ascii = False, indent = 4)

# # eval

# import json
# import re
# import numpy as np
# import sacrebleu
# import torch
# from tqdm import tqdm

# def _norm(s: str) -> str:
#     s = s.lower().strip()
#     s = re.sub(r"\s+", " ", s)
#     return s

# @torch.no_grad()
# def evaluate_and_save_predictions(
#     model,
#     dataloader,
#     tokenizer,
#     device,
#     output_json_path: str,
#     max_new_tokens: int = 16,
#     num_beams: int = 1,
# ):
#     model.eval()
#     tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

#     rows = []
#     all_hyps = []
#     all_refs = []

#     for batch_idx, batch in enumerate(tqdm(dataloader, desc="eval+save")):
#         batch = {k: v.to(device) for k, v in batch.items()}

#         pixel_values = batch["pixel_values"]
#         labels = batch.get("labels", None)

#         with torch.no_grad():
#             enc = model.encoder(pixel_values=batch["pixel_values"][:1])
#             dec_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)
#             out = model.decoder(
#                 input_ids=dec_ids,
#                 encoder_hidden_states=enc.last_hidden_state
#             )
#             logits = out.logits[0, -1]
#             top = torch.topk(logits, 5)

#             for i in range(5):
#                 print(tokenizer.decode([top.indices[i].item()]), top.values[i].item())

#         # import pdb; pdb.set_trace()
#         gen_ids = model.generate(
#             pixel_values=pixel_values,
#             max_new_tokens=max_new_tokens,
#             num_beams=num_beams,
#         )
#         hyp_texts = tok.batch_decode(gen_ids, skip_special_tokens=True)

#         if labels is not None:
#             labels_for_decode = labels.clone()
#             labels_for_decode[labels_for_decode == -100] = tok.pad_token_id
#             ref_texts = tok.batch_decode(labels_for_decode, skip_special_tokens=True)
#         else:
#             ref_texts = [None] * len(hyp_texts)

#         for i, (hyp, ref) in enumerate(zip(hyp_texts, ref_texts)):
#             hyp_n = _norm(hyp)
#             ref_n = _norm(ref) if ref is not None else None

#             row = {
#                 "id": batch_idx * len(hyp_texts) + i,
#                 "prediction": hyp,
#                 "prediction_norm": hyp_n,
#                 "reference": ref,
#                 "reference_norm": ref_n,
#             }

#             if ref_n is not None:
#                 row["exact_match"] = (hyp_n == ref_n)
#                 row["sentence_bleu"] = sacrebleu.sentence_bleu(hyp_n, [ref_n]).score

#                 all_hyps.append(hyp_n)
#                 all_refs.append([ref_n])

#             rows.append(row)

#     metrics = {}
#     if all_hyps:
#         metrics["exact_match_acc"] = float(np.mean([r["exact_match"] for r in rows if r["reference_norm"] is not None]))
#         metrics["bleu"] = sacrebleu.corpus_bleu(all_hyps, all_refs).score
#         metrics["avg_sentence_bleu"] = float(np.mean([r["sentence_bleu"] for r in rows if r.get("sentence_bleu") is not None]))

#     payload = {
#         "metrics": metrics,
#         "predictions": rows,
#     }

#     with open(output_json_path, "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)

#     model.train()
#     return metrics, output_json_path

# eval_dataset = PrecomputedTensorDataset(train_config["data_path"], limit_n=0, shuffle=False, seed=42)
# eval_loader = DataLoader(eval_dataset, batch_size=train_config["batch_size_eval"], shuffle=False, collate_fn=default_data_collator)

# pred_json_path = os.path.join(save_model_to, "eval_predictions.json")
# metrics, path = evaluate_and_save_predictions(model, eval_loader, tokenizer, device, pred_json_path)
# print("Eval metrics:", metrics)
# print("Saved predictions to:", path)