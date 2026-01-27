import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
                AutoTokenizer,
                default_data_collator,
                VisionEncoderDecoderModel,
                ViTImageProcessor,
                CLIPVisionModel,
                CLIPProcessor,
                AutoModelForCausalLM,
                )
from tqdm import tqdm
from lib_data_utils import PrecomputedTensorDataset
from lib_sys_utils import get_current_time_string
from lib_model import print_params
import os
import json
from lib_model import PrefixedLLM

train_config = {
    # "train_data_path": "data/gz_tensors/clip-vit-base-patch32_mGPT/train_en.pt",
    # "eval_data_path": "data/gz_tensors/clip-vit-base-patch32_mGPT/eval_en.pt",
    "train_data_path": "data/toy_colors_clip-vit-base-patch32.pt",
    "eval_data_path": "data/toy_colors_clip-vit-base-patch32.pt",
    'encoder_name': "openai/clip-vit-base-patch32", # English only, but if only used for image it's language-independent
    'decoder_name': "ai-forever/mGPT", # multilingual
    "num_epochs": 3,
    "num_steps": -1, # < 0 to run the full number of epochs
    "learning_rate": 5e-5,
    "batch_size_train": 8,
    "batch_size_eval": 8,
    "lang": "it",
    "model_type": "prefix",
}

train_dataset = PrecomputedTensorDataset(train_config['train_data_path'], limit_n=0, shuffle = True, seed = 42)
eval_dataset = PrecomputedTensorDataset(train_config['eval_data_path'], limit_n=0, shuffle = False, seed = 42)

#model_checkpoint = f"./models/{get_current_time_string()}" # path of the model checkpoint to load (only used if the previous line is True)
model_str = f"{train_config['encoder_name'].split('/')[-1]}_{train_config['decoder_name'].split('/')[-1]}"
save_model_to = f"./models/{model_str}_{get_current_time_string()}" # path where to save the fine-tuned model

# load model
feature_extractor = CLIPProcessor.from_pretrained(train_config['encoder_name'])
tokenizer = AutoTokenizer.from_pretrained(train_config['decoder_name'])

#model_checkpoint = f"./models/{get_current_time_string()}" # path of the model checkpoint to load (only used if the previous line is True)

save_model_to = f"./models/{model_str}_{get_current_time_string()}" # path where to save the fine-tuned model

clip_encoder = CLIPVisionModel.from_pretrained(train_config['encoder_name'], attn_implementation="eager") # attn_impl is necessary because of retro-compatibility issue
decoder = AutoModelForCausalLM.from_pretrained(train_config['decoder_name'], attn_implementation="eager")
        
# instantiate model
model = PrefixedLLM(encoder=clip_encoder, decoder=decoder)

for name, p in model.named_parameters():
    if 'vision_text_proj' in name:
        p.requires_grad = True
    else:
        p.requires_grad = False

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
eval_loader = DataLoader(eval_dataset, batch_size=train_config['batch_size_eval'], shuffle=False, collate_fn=default_data_collator)

optimizer = AdamW(params = model.parameters(), lr = train_config['learning_rate'])
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
train_iter = iter(train_loader)

if train_config['num_steps'] < 0:
    train_config['num_steps'] = len(train_dataset) * train_config['num_epochs']

tbar = tqdm(total=train_config['num_steps']) # progrss bar

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
    labels = batch['labels'] # Shape: [batch, seq_len]
    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
    
    # 2. Forward Pass (NO LABELS)
    outputs = model(
        pixel_values=batch['pixel_values'],
        input_ids=decoder_input_ids,
        attention_mask=batch.get('attention_mask', None), # Optional if using default collator
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

    # progress update
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
                    input_ids=decoder_input_ids,
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
            torch.save(model.state_dict(), f"{save_model_to}/model_state.pt") # save weights and parameters
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
