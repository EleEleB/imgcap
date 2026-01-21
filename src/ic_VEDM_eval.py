import torch
from torch.utils.data import DataLoader
from PIL import Image
from datasets import Dataset
from transformers import CLIPVisionModel, CLIPProcessor, CLIPVisionConfig, default_data_collator
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import VisionEncoderDecoderModel
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm
import sacrebleu
from transformers import CLIPModel
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

lang = "it" # en or it
eval_dataset_path = f"./data/test_{lang}.txt"
model_checkpoint = "./fine_tuned_VEDM"

output_scores_filepath = ("./fine_tuned_VEDM_Scores.txt")

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
def prepare_dataset(dataset_path, n = 0):
  data = load_dataset(dataset_path)
  if n > 0:
      data = data[:n]
  dataset = Dataset.from_list(data)

  # preprocess pictures the way the model wants them
  dataset = dataset.map(preprocess_example, batched=False) # map handles the calling so long as preprocess_example has only 1 argument

  return dataset


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

print("model instantiated") # DEBUGGING

# load clip_processor and tokenizer from the checkpoint (ensures token ids are aligned)
clip_processor = CLIPProcessor.from_pretrained(model_checkpoint)
decoder_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)

# load model weights from checkpoint
state_dict = torch.load(f"{model_checkpoint}/model_state.pt", map_location="cpu")
model.load_state_dict(state_dict, strict=True)

# padding in case the tokenizer doesn't already have it
if decoder_tokenizer.pad_token is None: # GPT2 has no pad token
    decoder_tokenizer.pad_token = decoder_tokenizer.eos_token # use eos token for the purpose

# prepare dataset
eval_dataset = prepare_dataset(eval_dataset_path, n = 10)

print("dataset prepared") # DEBUGGING

# use gpu if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# GENERATE CANDIDATE CAPTIONS

model.eval()

test_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, collate_fn=default_data_collator)

all_captions = []
for batch in tqdm(test_loader, desc="Generating captions"):
    # get pixel values and move to device
    pixel_values = batch["pixel_values"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad(): # inference, so no gradients needed
        generated_captions = model.generate(pixel_values=pixel_values, attention_mask=attention_mask, max_length=64, num_beams=4, early_stopping=True)

    captions = [decoder_tokenizer.decode(g, skip_special_tokens=True) for g in generated_captions] # skip_special_tokens ignores padding/eos tokens
    all_captions.extend(captions)




# EVALUATION
# ----------

ref_captions_blue = [[caption] for caption in eval_dataset["caption"]] # bleu wants the reference translation(s) as a list of lists
ref_captions = [caption for caption in eval_dataset["caption"]] # for everything else

# BLEU
bleu = sacrebleu.corpus_bleu(all_captions, ref_captions_blue)
bleu_per_instance = [sacrebleu.sentence_bleu(cap, ref) for cap, ref in zip(all_captions, ref_captions_blue)]


# ChrF++
chrf = sacrebleu.corpus_chrf(all_captions, ref_captions_blue, word_order=True)  # word_order=True enables ChrF++
chrf_per_instance = []
for capt, ref in zip(all_captions, ref_captions_blue):
    score = sacrebleu.sentence_chrf(capt, ref, word_order=True)
    chrf_per_instance.append(score.score)


# CLIP-Score & Ref-CLIP-Score
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1').to(device)
#text_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1", cache_folder="./tmp").to(device) # on john

clip_model.eval()

clip_score_per_instance = []
ref_clip_score_per_instance = []
caption_current_batch_start = 0 # used to keep track of index in all_captions because images are in batches
for batch in tqdm(test_loader, desc="Evaluating CLIP similarity"):

    # get image_features
    pixel_values = batch["pixel_values"].to(device)
    with torch.no_grad(): # inference, so no gradients needed
        image_features = clip_model.get_image_features(pixel_values=pixel_values)

    # get caption features
    caption_current_batch_end = caption_current_batch_start + len(pixel_values)
    batch_captions = all_captions[caption_current_batch_start : caption_current_batch_end] # slice captions for this batch
    batch_refs = ref_captions[caption_current_batch_start : caption_current_batch_end] # slice references for this batch
    caption_current_batch_start = caption_current_batch_end

    with torch.no_grad(): # inference, so no gradients needed
        # m-clip tokenizes internally as part of the encode instruction
        text_features = text_model.encode(batch_captions, convert_to_tensor=True)
        reference_features = text_model.encode(batch_refs, convert_to_tensor=True)


    # normalize (dim=-1 means the last dimension of the tensor, which is the embedding dimension)
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features  = F.normalize(text_features,  p=2, dim=-1)
    reference_features = F.normalize(reference_features, p=2, dim=-1)

    # CLIP-SCORE: image - candidate caption cosine similarity (compute dot product of normalized embeddings)
    batch_sims = (image_features * text_features).sum(dim=-1) # CLIP-SCORE
    clip_score_per_instance.extend(batch_sims.tolist())

    # REF-CLIP SCORE: harmonic mean of image-candidate similarity and candidate-reference similarity
    # harmonic mean penalizes cases in which one the terms is high and the other low
    batch_ref_text_sims = (text_features * reference_features).sum(dim=-1)
    # similarities can be negative, and harmonic mean assumes non-negative values, so clamp them (keep max of similarity and 0)
    clamped_batch_sims = torch.clamp(batch_sims, min=0.0)
    clamped_batch_ref_text_sims = torch.clamp(batch_ref_text_sims, min=0.0)
    # harmonic mean
    numer = 2 * clamped_batch_sims * clamped_batch_ref_text_sims
    denom = clamped_batch_sims + clamped_batch_ref_text_sims
    harmonic_mean = torch.where(denom == 0, torch.zeros_like(denom), numer/denom) # avoid dividing by 0 so result is a tensor of 0 where the denominator is 0
    ref_clip_score_per_instance.extend(harmonic_mean.tolist())

#clip_score_mean = np.mean(clip_score_per_instance)
clip_score_median = np.median(clip_score_per_instance)
ref_clip_score_median = np.median(ref_clip_score_per_instance)


with open(output_scores_filepath, mode="w", encoding="utf-8") as f:
    # write overall scores
    f.write(f"OVERALL SCORE\n")
    f.write(f"CLIP-Score\tRef-CLIP-Score\tSacrebleu\tChrF++\n")
    f.write(f"{clip_score_median}\t{ref_clip_score_median}\t{bleu.score}\t{chrf.score}\n")

    f.write(f"SCORE PER INSTANCE\n")
    f.write(f"Instance\tGeneratedCaption\tReferenceCaption\tCLIP-Score\tRefCLIP-Score\tSacrebleu\tCharF++\n")
    # write scores per instance
    for i, (gen_capt, ref_capt, sim, ref_sim, bleuscore, chrfscore) in enumerate(zip(all_captions, ref_captions, clip_score_per_instance, ref_clip_score_per_instance, bleu_per_instance, chrf_per_instance)):
        f.write(f"{i+1}\t{gen_capt}\t{ref_capt}\t{sim}\t{ref_sim}\t{bleuscore.score}\t{chrfscore}\n")

print("file saved") # DEBUGGING
