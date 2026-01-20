import torch
from PIL import Image
from datasets import Dataset
from transformers import CLIPVisionModel, CLIPProcessor, CLIPVisionConfig, default_data_collator
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import VisionEncoderDecoderModel
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
import os



lang = "it" # en or it
resume = True # True loads a saved model and continues fine-tuning, False starts from scratch
model_checkpoint = "./fine_tuned_VEDM_ep17" # path of the model checkpoint to load (only used if the previous line is True)
save_model_to = "./fine_tuned_VEDM_ep30" # path where to save the fine-tuned model
num_epochs = 13


# function that imports the dataset from a txt file
# parameter dataset_path is the file path
# returns the extracted dataset
def load_dataset(dataset_path):
    # load training dataset from file
    with open(dataset_path, mode="r", encoding="utf-8") as f:
        txt = f.read()
    temp_dataset = txt.split("\n")

    dataset = []
    for el in temp_dataset:
        if el != "":
            temp = el.split("\t")
            dataset.append({"image_path": temp[0], "caption": temp[1]})

    return dataset



# function that preprocesses the a training instance and puts it in the format the model wants
# parameter dataset is a list of instances
# returns the data in the format the model wants
def preprocess_example(example):
    # load image
    image = Image.open(example["image_path"]).convert("RGB")

    # use CLIPProcessor to preprocess images (returns PyTorch tensors with correct size and normalization)
    pixel_values = clip_processor(images=image, return_tensors="pt").pixel_values[0]  # [3, H, W]

    # Tokenize caption
    temp = decoder_tokenizer(example["caption"], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    labels = temp.input_ids[0]  # the actual tokenized input, shape [max_length]

    # the attention mask (1 for a real token and 0 for a padding token) ensures the model ignores padding tokens during the training
    attention_mask = temp.attention_mask[0]

    # replace padding token id with -100 to ignore in loss (otherwise loss gets computed on padding)
    labels[labels == decoder_tokenizer.pad_token_id] = -100

    return {"pixel_values": pixel_values, "labels": labels, "attention_mask": attention_mask}


# function that prepares the dataset from filepath to mapping
# parameter dataset_path is the filepath of the txt file containing the dataset
# returns the mapped dataset
def prepare_dataset(dataset_path):
  data = load_dataset(dataset_path)
  dataset = Dataset.from_list(data)

  # preprocess pictures the way the model wants them
  dataset = dataset.map(preprocess_example, batched=False) # map handles the calling so long as preprocess_example has only 1 argument

  return dataset



print("imports done")  # DEBUGGING

train_dataset_path = f"./TrainingDataset_{lang}.txt"
eval_dataset_path = f"./TestingDataset_{lang}.txt"



# model part names (also used when continuing to rebuild the model config)
clip_name = "openai/clip-vit-base-patch32" # English only, but if only used for image it's language-independent
decoder_name = "ai-forever/mGPT" # multilingual
#decoder_name = "distilgpt2" # English only ------------ testing only


# define adapter
class VisionAdapter(nn.Module):
    def __init__(self, clip_dim, decoder_dim, reduction=4):
        super(VisionAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(clip_dim, clip_dim // reduction, bias=False),
            nn.GELU(),
            nn.Linear(clip_dim // reduction, clip_dim, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# wraps clip and adapter together as an encoder
class CLIPWithAdapter(PreTrainedModel):
    config_class = CLIPVisionConfig  # the CLIP config is assigned automatically as part of the init

    def __init__(self, clip_model, adapter):
        super().__init__(clip_model.config)
        self.clip = clip_model
        self.adapter = adapter

    # necessary or it triggers an error
    def get_input_embeddings(self):
        return self.clip.get_input_embeddings()

    # vision encoders don't produce token embeddings so this is none
    def get_output_embeddings(self):
        return None

    def forward(self, pixel_values, input_ids=None, attention_mask=None, **kwargs): # input_ids and attention_mask are required for compatibility (inherited requirement from PreTrainedModel)
        # pass the pixel values through CLIP
        outputs = self.clip(pixel_values=pixel_values, **kwargs)

        # extract vision features (last_hidden_state)
        vision_feats = outputs.last_hidden_state

        # apply the vision adapter to process the features
        #vision_feats = vision_feats.mean(dim=1, keepdim=True)  # add mean pooling
        adapted_feats = self.adapter(vision_feats) # apply the adapter

        # do residual-style blending as in the CLIP-Adapter paper
        new_feats = vision_feats + adapted_feats

        # return full encoder outputs (for cross-attention compatibility)
        return BaseModelOutput(
            last_hidden_state=new_feats,  # processed image embeddings
            hidden_states=outputs.hidden_states,  # encoder's intermediate states
            attentions=outputs.attentions  # attention scores (important for cross-attention)
        )


# encoder
clip_encoder = CLIPVisionModel.from_pretrained(clip_name, attn_implementation="eager") # attn_impl is necessary because of retro-compatibility issue

# decoder (+ cross attention)
decoder_config = AutoConfig.from_pretrained(decoder_name)
decoder_config.add_cross_attention = True
decoder = AutoModelForCausalLM.from_pretrained(decoder_name, config=decoder_config)

# vision adapter (included in encoder)
clip_dim = clip_encoder.config.hidden_size
decoder_dim = decoder.config.hidden_size
vision_adapter = VisionAdapter(clip_dim=clip_dim, decoder_dim=decoder_dim)
encoder = CLIPWithAdapter(clip_encoder, vision_adapter)
#encoder.config.hidden_size = decoder_dim # the adapter already changes the encoder's hidden dimension to the decoder's NOT ANYMORE

# instantiate model
model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

model.config.encoder_hidden_size = clip_dim # redundant, but left for clarity
#model.config.encoder_hidden_size = decoder_dim # redundant, but left for clarity NOT CORRECT ANYMORE


if resume == True:
    # load clip_processor and tokenizer from the checkpoint (ensures token ids are aligned)
    clip_processor = CLIPProcessor.from_pretrained(model_checkpoint)
    decoder_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
    
    # load model weights from checkpoint
    state_dict = torch.load(f"{model_checkpoint}/model_state.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    print("model weights loaded") # DEBUGGING

else:
    # clip processor and tokenizer
    clip_processor = CLIPProcessor.from_pretrained(clip_name)
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name, use_fast=False)

    # initialize adapter and cross attention close to zero (and projection layer, if present, which it shouldn't be)
    for name, param in model.named_parameters():
        if "adapter" in name or "cross_attention" in name or "crossattention" in name or "cross_attn" in name or "enc_to_dec_proj" in name:
            if param.dim() > 1:  # weights (the tensors of biases have only 1 dimension, parameters have more than 1)
                nn.init.normal_(param, mean=0.0, std=1e-4)
            else:  # biases (if any)
                nn.init.zeros_(param) # starts at zero
        

# padding in case the tokenizer doesn't already have it
if decoder_tokenizer.pad_token is None: # GPT2 has no pad token
    decoder_tokenizer.pad_token = decoder_tokenizer.eos_token # use eos token for the purpose

# make sure the model knows which id represents padding
model.config.pad_token_id = decoder_tokenizer.pad_token_id

# set the decoder start token ID (required for generation)
if decoder_tokenizer.bos_token_id is None:  # GPT-like models may not have a BOS token
    model.config.decoder_start_token_id = decoder_tokenizer.eos_token_id
else:
    model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id


print("model created and padding fixed")  # DEBUGGING


# freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# unfreeze cross attention parameters and last layer only
for name, param in model.named_parameters():
    if "cross_attention" in name or "crossattention" in name or "cross_attn" in name or "ln_f" in name or "enc_to_dec_proj" in name or "adapter" in name:
        param.requires_grad = True
        print(f"Trainable: {name}") # DEBUGGING
    else:
        print(f"Frozen: {name}") # DEBUGGING

print("model frozen except cross attention")  # DEBUGGING


# use gpu if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("model moved to device")  # DEBUGGING


# PREPARE DATASET
train_dataset = prepare_dataset(train_dataset_path)
eval_dataset = prepare_dataset(eval_dataset_path)

print("dataset prepared")  # DEBUGGING


training_args = TrainingArguments(
    # model and training settings
    num_train_epochs=num_epochs,                    # number of training epochs --> extend if it's still improving at the end
    per_device_train_batch_size=8,                  # batch size for training
    per_device_eval_batch_size=8,                   # batch size for evaluation
    learning_rate=1e-5,                             # learning rate
    weight_decay=0.00,                              # weight decay for optimization
    # training enhancements (warmup and mixed-precision training)
    warmup_ratio=0.1,                              # transformers have trouble optimizing without a warm up
    lr_scheduler_type="linear",                     # use linear decay for the learning rate after warmup
    fp16=True,                                      # use mixed-precision training (requires compatible hardware, but speeds up training)
    max_grad_norm = 1.0,
    # checkpointing and saving
    output_dir=f"./results",                        # output directory for model checkpoints
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

print("training done")  # DEBUGGING

# save model and clip_processor + tokenizer
#model.save_pretrained(save_model_to) # this saves a "corrupt" config (aka it doesn't save the adapter structure at all)
os.makedirs(save_model_to, exist_ok=True)
torch.save(model.state_dict(), f"{save_model_to}/model_state.pt") # save weights and parameters
clip_processor.save_pretrained(save_model_to)
decoder_tokenizer.save_pretrained(save_model_to)

print("model saved") # DEBUGGING
