import torch
from torch.utils.data import DataLoader
from transformers import (
                AutoTokenizer,
                default_data_collator,
                VisionEncoderDecoderModel,
                ViTImageProcessor,
                TrainingArguments,
                Trainer,
                EarlyStoppingCallback,
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
    "num_epochs": 3,
    "num_steps": 1000,
    "learning_rate": 1e-5,
    "batch_size_train": 8,
    "batch_size_eval": 8,
    "lang": "en",
}

resume = False # True loads a saved model and continues fine-tuning, False starts from scratch
model_checkpoint = f"./models/{get_current_time_string()}" # path of the model checkpoint to load (only used if the previous line is True)

# train_dataset_path = f"./data/train_processed_{train_config['lang']}.pt"
# eval_dataset_path = f"./data/eval_processed_{train_config['lang']}.pt"
train_dataset_path = "data/train_processed_en_vit-gpt2-coco-en_vit-gpt2-coco-en.pt"
eval_dataset_path = "data/eval_processed_en_vit-gpt2-coco-en_vit-gpt2-coco-en.pt"

if resume == False: # start fine-tuning from scratch
    model_name = "ydshieh/vit-gpt2-coco-en"
else: # resume fine-tuning from checkpoint
    model_name = model_checkpoint

save_model_to = f"./models/{model_name.split('/')[-1]}_{get_current_time_string()}" # path where to save the fine-tuned model

# load model
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = VisionEncoderDecoderModel.from_pretrained(model_name)

for name, p in model.decoder.named_parameters():
    if "crossattention" in name:
        p.requires_grad = True
    else:
        p.requires_grad = False

for name, p in model.encoder.named_parameters():
    p.requires_grad = True

print_trainable_parameters(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print_params(model)
# PREPARE DATASET
# train_dataset = prepare_dataset(train_dataset_path, feature_extractor, tokenizer, batched=True)
# eval_dataset = prepare_dataset(eval_dataset_path, feature_extractor, tokenizer, batched=True)
train_dataset = PrecomputedTensorDataset(train_dataset_path, limit_n=0, shuffle = True, seed = 42)
eval_dataset = PrecomputedTensorDataset(eval_dataset_path, limit_n=0, shuffle = False, seed = 42)

# import pdb; pdb.set_trace()

training_args = TrainingArguments(
    # model and training settings
    max_steps=train_config["num_steps"],
    num_train_epochs=train_config["num_epochs"],                    # number of training epochs --> extend if it's still improving at the end
    per_device_train_batch_size=train_config["batch_size_train"],   # batch size for training
    per_device_eval_batch_size=train_config["batch_size_eval"],     # batch size for evaluation
    learning_rate=train_config["learning_rate"],                    # learning rate
    weight_decay=0.01,                              # weight decay for optimization
    # training enhancements (warmup and mixed-precision training)
    warmup_ratio=0.1,                               # transformers have trouble optimizing without a warm up
    lr_scheduler_type="linear",                     # use linear decay for the learning rate after warmup
    fp16=False,                                      # use mixed-precision training (requires compatible hardware, but speeds up training)
    # max_grad_norm = 1.0,
    # checkpointing and saving
    output_dir=f"./checkpoints",                    # output directory for model checkpoints
    # save_strategy="epoch",                          # save after every epoch (checkpoint)
    # save_total_limit=3,                             # keep only the last X checkpoints
    # evaluation
    # eval_strategy="epoch",                          # evaluate every epoch
    # load_best_model_at_end=True,                    # load best model based on evaluation performance
    # metric_for_best_model="eval_loss",              # monitor this metric for the best model
    # logging
    report_to="none",                               # disable logging to WandB (and other platforms like TensorBoard)
    logging_dir=f"./logs",                          # directory to save logs
    logging_steps=0.01,                             # log every 500 steps
    remove_unused_columns=False,
)


trainer = Trainer(
    model=model.to(device),                         # the model to train + ensure it is moved to GPU
    args=training_args,                             # training arguments
    train_dataset=train_dataset,                    # preprocessed training dataset
    # eval_dataset=eval_dataset,                      # preprocessed evaluation dataset
    tokenizer=tokenizer,                            # tokenizer for the model
    data_collator=default_data_collator,            # data collator for batching
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # early stop
)

trainer.train()

print("training done")  # DEBUGGING

# save model and clip_processor + tokenizer
model.save_pretrained(save_model_to)
feature_extractor.save_pretrained(save_model_to)
tokenizer.save_pretrained(save_model_to)

print(f"model saved at: {save_model_to}") # DEBUGGING

json_path = os.path.join(save_model_to, 'train_config.json')

with open(json_path, 'w', encoding='utf8') as f:
    json.dump(train_config, f, ensure_ascii = False, indent = 4)