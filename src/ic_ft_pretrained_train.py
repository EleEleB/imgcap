import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from tqdm import tqdm
import sacrebleu
from transformers import CLIPModel
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from lib_data_utils import prepare_dataset, PrecomputedTensorDataset

train_config = {
    "num_epochs": 0,
    "learning_rate": 5e-5,
    "batch_size_train": 8,
    "batch_size_eval": 8,
    "lang": "en",
}

resume = False # True loads a saved model and continues fine-tuning, False starts from scratch
model_checkpoint = f"./models/ft_pretrained_partial_{train_config['num_epochs']}ep" # path of the model checkpoint to load (only used if the previous line is True)
save_model_to = f"./models/ft_pretrained_partial_{train_config['num_epochs']}ep" # path where to save the fine-tuned model

# train_dataset_path = f"./data/train_processed_{train_config['lang']}.pt"
# eval_dataset_path = f"./data/eval_processed_{train_config['lang']}.pt"
train_dataset_path = "data/train_processed_en_vit-gpt2-coco-en_vit-gpt2-coco-en.pt"
eval_dataset_path = "data/eval_processed_en_vit-gpt2-coco-en_vit-gpt2-coco-en.pt"

if resume == False: # start fine-tuning from scratch
    model_name = "ydshieh/vit-gpt2-coco-en"
else: # resume fine-tuning from checkpoint
    model_name = model_checkpoint

# load model
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# freeze encoder lower layers (embeddings + layers 0-7. Layers 8-11 remain trainable)
# for name, param in model.encoder.named_parameters():
#     if "embeddings" in name or "layer.0." in name or "layer.1." in name or "layer.2." in name or "layer.3." in name or "layer.4." in name or "layer.5." in name or "layer.6." in name or "layer.7." in name:
#         param.requires_grad = False

# for name, param in model.named_parameters():
#     print(f"{param.requires_grad}, {name}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# PREPARE DATASET
# train_dataset = prepare_dataset(train_dataset_path, feature_extractor, tokenizer, batched=True)
# eval_dataset = prepare_dataset(eval_dataset_path, feature_extractor, tokenizer, batched=True)
train_dataset = PrecomputedTensorDataset(train_dataset_path, limit_n=0, shuffle = True, seed = 42)
eval_dataset = PrecomputedTensorDataset(eval_dataset_path, limit_n=0, shuffle = False, seed = 42)


training_args = TrainingArguments(
    # model and training settings
    num_train_epochs=train_config["num_epochs"],                    # number of training epochs --> extend if it's still improving at the end
    per_device_train_batch_size=train_config["batch_size_train"],   # batch size for training
    per_device_eval_batch_size=train_config["batch_size_eval"],     # batch size for evaluation
    learning_rate=train_config["learning_rate"],                    # learning rate
    weight_decay=0.01,                              # weight decay for optimization
    # training enhancements (warmup and mixed-precision training)
    warmup_ratio=0.1,                               # transformers have trouble optimizing without a warm up
    lr_scheduler_type="linear",                     # use linear decay for the learning rate after warmup
    fp16=True,                                      # use mixed-precision training (requires compatible hardware, but speeds up training)
    max_grad_norm = 1.0,
    # checkpointing and saving
    output_dir=f"./checkpoints",                   # output directory for model checkpoints
    save_strategy="epoch",                          # save after every epoch (checkpoint)
    save_total_limit=3,                             # keep only the last X checkpoints
    # evaluation
    eval_strategy="epoch",                          # evaluate every epoch
    load_best_model_at_end=True,                    # load best model based on evaluation performance
    metric_for_best_model="eval_loss",              # monitor this metric for the best model
    # logging
    report_to="none",                               # disable logging to WandB (and other platforms like TensorBoard)
    logging_dir=f"./logs",                          # directory to save logs
    logging_steps=500,                              # log every 500 steps
    remove_unused_columns=False,
)


trainer = Trainer(
    model=model.to(device),                         # the model to train + ensure it is moved to GPU
    args=training_args,                             # training arguments
    train_dataset=train_dataset,                    # preprocessed training dataset
    eval_dataset=eval_dataset,                      # preprocessed evaluation dataset
    tokenizer=tokenizer,                            # tokenizer for the model
    data_collator=default_data_collator,            # data collator for batching
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # early stop
)

trainer.train()

print("training done")  # DEBUGGING

# save model and clip_processor + tokenizer
model.save_pretrained(save_model_to)
feature_extractor.save_pretrained(save_model_to)
tokenizer.save_pretrained(save_model_to)

print("model saved") # DEBUGGING