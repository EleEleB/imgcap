import json
import re
import os
import numpy as np
import sacrebleu
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    default_data_collator,
)
from lib_data_utils import PrecomputedTensorDataset

# --- Configuration for Toy Setting ---
CONFIG = {
    "data_path": "data/toy_colors.pt",
    "batch_size_eval": 8,
    "model_path": "./models/vit-gpt2-coco-en_2026-01-26--11-18-56",
    "output_file": "eval_predictions.json",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_new_tokens": 16,
    "num_beams": 1
}

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
            enc = model.encoder(pixel_values=batch["pixel_values"][:1])
            dec_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)
            out = model.decoder(
                input_ids=dec_ids,
                encoder_hidden_states=enc.last_hidden_state
            )
            logits = out.logits[0, -1]
            top = torch.topk(logits, 5)
            print("\nTop 5 initial tokens (Debug):")
            for i in range(5):
                print(f"Token: {tok.decode([top.indices[i].item()])}, Score: {top.values[i].item():.4f}")

        # Generate captions
        gen_ids = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
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
                all_refs.append([ref_n])

            rows.append(row)

    # Compute aggregate metrics
    metrics = {}
    if all_hyps:
        metrics["exact_match_acc"] = float(np.mean([r["exact_match"] for r in rows if r["reference_norm"] is not None]))
        metrics["bleu"] = sacrebleu.corpus_bleu(all_hyps, all_refs).score
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

def main():
    # 1. Load Artifacts
    print(f"Loading model from: {CONFIG['model_path']}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_path'])
        model = VisionEncoderDecoderModel.from_pretrained(CONFIG['model_path'])
        # Feature extractor is loaded but typically handled via the precomputed dataset or processor
        # feature_extractor = ViTImageProcessor.from_pretrained(CONFIG['model_path'])
    except OSError as e:
        print(f"Error loading model: {e}")
        print("Ensure 'model_path' in CONFIG points to a valid checkpoint directory.")
        return

    # 2. Configure Model
    model.to(CONFIG['device'])
    
    # ensure generation config is aligned with training config
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # 3. Prepare Dataset
    print(f"Loading dataset: {CONFIG['data_path']}")
    eval_dataset = PrecomputedTensorDataset(
        CONFIG['data_path'], 
        limit_n=0, 
        shuffle=False, 
        seed=42
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=CONFIG['batch_size_eval'], 
        shuffle=False, 
        collate_fn=default_data_collator
    )

    # 4. Execute Evaluation
    output_path = os.path.join(CONFIG['model_path'], CONFIG['output_file'])
    metrics, path = evaluate_and_save_predictions(
        model=model,
        dataloader=eval_loader,
        tokenizer=tokenizer,
        device=CONFIG['device'],
        output_json_path=output_path,
        max_new_tokens=CONFIG['max_new_tokens'],
        num_beams=CONFIG['num_beams']
    )

    print("Evaluation Complete.")
    print("Metrics:", json.dumps(metrics, indent=2))
    print(f"Predictions saved to: {path}")

if __name__ == "__main__":
    main()