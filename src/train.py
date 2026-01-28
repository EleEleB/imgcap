import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
                AutoTokenizer,
                default_data_collator,
                VisionEncoderDecoderModel,
                ViTImageProcessor,
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
    # "train_data_path": "data/gz_tensors/vit-gpt2-coco-en/train_en.pt",
    # "eval_data_path": "data/gz_tensors/vit-gpt2-coco-en/eval_en.pt",
    # "train_data_path": "data/toy_colors_clip-vit-base-patch32.pt",
    # "eval_data_path": "data/toy_colors_clip-vit-base-patch32.pt",
    "train_data_path": "data/toy_colors_vit-gpt2-coco-en.pt",
    "eval_data_path": "data/toy_colors_vit-gpt2-coco-en.pt",
    "num_epochs": 3,
    "num_steps": 100, # < 0 to do the full epochs
    "learning_rate": 5e-5,
    "batch_size_train": 8,
    "batch_size_eval": 8,
    "lang": "en",
    "model_type": "encoder-decoder",
}

model_checkpoint = f"./models/{get_current_time_string()}" # path of the model checkpoint to load (only used if the previous line is True)

model_name = "ydshieh/vit-gpt2-coco-en"
# model_name = "google-bert/bert-base-uncased"

save_model_to = f"./models/{model_name.split('/')[-1]}_{get_current_time_string()}" # path where to save the fine-tuned model

# load model
feature_extractor = ViTImageProcessor.from_pretrained(model_name) # NOTE: this is not used because the dataset already contains pixel values
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = PrecomputedTensorDataset(train_config['train_data_path'], limit_n=0, shuffle = True, seed = 42)
eval_dataset = PrecomputedTensorDataset(train_config['eval_data_path'], limit_n=0, shuffle = True, seed = 42)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

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


train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size_train'], shuffle=True, collate_fn=default_data_collator)
eval_loader = DataLoader(eval_dataset, batch_size=train_config['batch_size_eval'], shuffle=True, collate_fn=default_data_collator)

optimizer = AdamW(params = model.parameters(), lr = train_config['learning_rate'])
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
train_iter = iter(train_loader)

if train_config['num_steps'] < 0:
    train_config['num_steps'] = len(train_dataset) * train_config['num_epochs']

tbar = tqdm(total=train_config['num_steps']) # progress bar

# evaluation and early stopping
eval_every = train_config['num_steps'] // train_config['num_epochs'] # validate once per epoch
best_val_loss = float("inf")
bad_evals = 0
patience = 2
os.makedirs(save_model_to, exist_ok=True) # create the folder where to save the model

# training loop
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
    # compute loss
    loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    # backpropagation
    loss.backward()
    optimizer.step()

    # update progress bar
    tbar.set_description(f"Loss: {loss.item():.3f}")
    tbar.update(1)

    # validation
    if step % eval_every == 0 and step != 0:
        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                labels = batch["labels"]
                decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)

                outputs = model(
                    pixel_values=batch["pixel_values"],
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=batch.get("attention_mask", None),
                )

                logits = outputs.logits
                loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

                val_loss += loss.item()
                n_batches += 1

        val_loss /= n_batches
        model.train() # reset training mode

        print(f"[step {step}] val_loss = {val_loss:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_evals = 0

            # Save the best model so far
            # save model and clip_processor + tokenizer
            model.save_pretrained(save_model_to)
            feature_extractor.save_pretrained(save_model_to)
            tokenizer.save_pretrained(save_model_to)
        else:
            bad_evals += 1
            if bad_evals >= patience:
                print("Early stopping.")
                break


print(f"model saved at: {save_model_to}") # DEBUGGING

json_path = os.path.join(save_model_to, 'train_config.json')

with open(json_path, 'w', encoding='utf8') as f:
    json.dump(train_config, f, ensure_ascii = False, indent = 4)
