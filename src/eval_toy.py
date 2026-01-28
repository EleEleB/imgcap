import json
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderModel,
    default_data_collator,
    CLIPVisionModel,
    AutoModelForCausalLM
)
from lib_data_utils import PrecomputedTensorDataset
from lib_model import PrefixedLLM, evaluate_and_save_predictions


def main():
    # --- Configuration for Toy Setting ---
    CONFIG = {
        # "data_path": "data/toy_colors.pt",
        # "data_path": "data/toy_colors_vit-gpt2-coco-en.pt",
        "data_path": "data/toy_colors_clip-vit-base-patch32_mGPT.pt",
        "batch_size_eval": 8,
        "output_file": "eval_predictions.json",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_new_tokens": 16,
        "num_beams": 1,
    }

    # model_path = "./models/vit-gpt2-coco-en_2026-01-28--13-21-10" # enc-dec
    model_path = "./models/clip-vit-base-patch32_mGPT_2026-01-28--16-23-33" # prefix

    train_config_path = os.path.join(model_path, 'train_config.json')

    with open(train_config_path, 'r', encoding='utf8') as f:
        train_config = json.load(f)

    # 1. Load Artifacts
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if train_config['model_type'] == 'encoder-decoder':
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        CONFIG["data_path"] = "./data/toy_colors_vit-gpt2-coco-en.pt"
    elif train_config['model_type'] == 'prefix':
        # reinstantiate custom model
        clip_name = "openai/clip-vit-base-patch32" # English only, but if only used fsor image it's language-independent
        decoder_name = "ai-forever/mGPT" # multilingual
        clip_encoder = CLIPVisionModel.from_pretrained(clip_name, attn_implementation="eager")
        decoder = AutoModelForCausalLM.from_pretrained(decoder_name, attn_implementation="eager")
        # model = PrefixedLLM.from_pretrained(model_path)
        CONFIG["data_path"] = "./data/toy_colors_clip-vit-base-patch32_mGPT.pt"
        model = PrefixedLLM(encoder=clip_encoder, decoder=decoder)

        # load model weights from checkpoint
        state_dict = torch.load(f"{model_path}/model_state.pt", map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    # ensure generation config is aligned with training config
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    model.to(CONFIG['device'])

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
    output_path = os.path.join(model_path, CONFIG['output_file'])
    metrics, path = evaluate_and_save_predictions(
        model=model,
        dataloader=eval_loader,
        tokenizer=tokenizer,
        device=CONFIG['device'],
        output_json_path=output_path,
        train_config=train_config,
        max_new_tokens=CONFIG['max_new_tokens'],
        num_beams=CONFIG['num_beams']
    )

    print("Evaluation Complete.")
    print("Metrics:", json.dumps(metrics, indent=2))
    print(f"Predictions saved to: {path}")

if __name__ == "__main__":
    main()