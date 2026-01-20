import torch
from transformers import CLIPVisionModel, CLIPProcessor, default_data_collator
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from data_utils import PrecomputedTensorDataset
from model import PrefixedLLM
import os
from sys_utils import get_current_time_string

lang = "it" # en or it
resume = False # True loads a saved model and continues fine-tuning, False starts from scratch
model_checkpoint = "./models/fine_tuned_VEDM" # path of the model checkpoint to load (only used if the previous line is True)
num_epochs = 1
save_model_to = f"./models/fine_tuned_VEDM_{get_current_time_string()}" # path where to save the fine-tuned model

train_dataset_path = f"./data/TrainingDataset_{lang}.txt"
eval_dataset_path = f"./data/TestingDataset_{lang}.txt"

# model part names (also used when continuing to rebuild the model config)
clip_name = "openai/clip-vit-base-patch32" # English only, but if only used for image it's language-independent
decoder_name = "ai-forever/mGPT" # multilingual
#decoder_name = "distilgpt2" # English only ------------ testing only

# encoder
clip_encoder = CLIPVisionModel.from_pretrained(clip_name, attn_implementation="eager") # attn_impl is necessary because of retro-compatibility issue
clip_processor = CLIPProcessor.from_pretrained(clip_name)

# decoder (+ cross attention)
decoder = AutoModelForCausalLM.from_pretrained(decoder_name)
decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        
# instantiate model
model = PrefixedLLM(encoder=clip_encoder, decoder=decoder)
        
# padding in case the tokenizer doesn't already have it
if decoder_tokenizer.pad_token is None: # GPT2 has no pad token
    decoder_tokenizer.pad_token = decoder_tokenizer.eos_token # use eos token for the purpose
decoder_tokenizer.padding_side = 'right'

# # freeze model parameters
# for param in model.parameters():
#     param.requires_grad = False

# # unfreeze cross attention parameters and last layer only
# for name, param in model.named_parameters():
#     if "vision_text_proj" in name:
#         param.requires_grad = True

# for name, param in model.decoder.named_parameters():
#     param.requires_grad = True

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('total_params', total_params)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('trainable_params', trainable_params)

print(model)

train_pt_path = f"./data/train_processed_{lang}.pt"
eval_pt_path = f"./data/test_processed_{lang}.pt"

# Initialize Datasets directly from tensors
# Note: clip_processor and decoder_tokenizer are no longer needed for dataset creation
# but are still needed for model initialization.
train_dataset = PrecomputedTensorDataset(train_pt_path, limit_n=0, shuffle = True, seed = 42)
eval_dataset = PrecomputedTensorDataset(eval_pt_path, limit_n=0, shuffle = False, seed = 42)
steps = 1000
training_args = TrainingArguments(
    max_steps=steps,
    save_steps=steps,
    eval_steps=steps,
    num_train_epochs=num_epochs,                    # number of training epochs --> extend if it's still improving at the end
    per_device_train_batch_size=8,                  # batch size for training
    per_device_eval_batch_size=8,                   # batch size for evaluation
    learning_rate=5e-5,                             # learning rate
    weight_decay=0.00,                              # weight decay for optimization
    # training enhancements (warmup and mixed-precision training)
    warmup_ratio=0.1,                               # transformers have trouble optimizing without a warm up
    lr_scheduler_type="linear",                     # use linear decay for the learning rate after warmup
    fp16=True,                                      # use mixed-precision training (requires compatible hardware, but speeds up training)
    # max_grad_norm = 1.0,
    # checkpointing and saving
    output_dir=f"./checkpoints",                        # output directory for model checkpoints
    save_strategy="steps",                          # save after every epoch (checkpoint)
    eval_strategy="steps",                          # evaluate every epoch
    save_total_limit=3,                             # keep only the last X checkpoints
    # evaluation
    load_best_model_at_end=True,                    # load best model based on evaluation performance
    metric_for_best_model="eval_loss",              # monitor this metric for the best model
    # logging
    report_to="none",                               # disable logging to WandB (and other platforms like TensorBoard)
    logging_dir=f"./logs",                          # directory to save logs
    logging_steps=0.01,                             # log every 500 steps
    remove_unused_columns=False,
    save_safetensors=False,
)

trainer = Trainer(
    model=model.to(device),                         # the model to train + ensure it is moved to GPU
    args=training_args,                             # training arguments
    train_dataset=train_dataset,                    # preprocessed training dataset
    eval_dataset=eval_dataset,                      # preprocessed evaluation dataset
    tokenizer=decoder_tokenizer,                    # tokenizer for the model
    data_collator=default_data_collator,            # data collator for batching
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # early stop
)

trainer.train()

os.makedirs(save_model_to, exist_ok=True)
torch.save(model.state_dict(), f"{save_model_to}/model_state.pt") # save weights and parameters
clip_processor.save_pretrained(save_model_to)
decoder_tokenizer.save_pretrained(save_model_to)
