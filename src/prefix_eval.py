import torch
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel, CLIPProcessor, default_data_collator
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from tqdm import tqdm
import sacrebleu
from transformers import CLIPModel
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from lib_data_utils import PrecomputedTensorDataset
from lib_model import PrefixedLLM, greedy_generate_prefixed
from bert_score import score

lang = "it"
test_pt_path = f"./data/gz_tensors/clip-vit-base-patch32_mGPT/test_{lang}.pt"
model_path = "./models/clip-vit-base-patch32_mGPT_2026-02-05--13-46-29"
output_scores_filepath = (f"./results/prefix_scores_{model_path.split('/')[-1]}.txt")


# reinstantiate custom model
clip_name = "openai/clip-vit-base-patch32" # English only, but if only used fsor image it's language-independent
decoder_name = "ai-forever/mGPT" # multilingual
#decoder_name = "distilgpt2" # English only ------------ testing only
clip_encoder = CLIPVisionModel.from_pretrained(clip_name, attn_implementation="eager")
decoder = AutoModelForCausalLM.from_pretrained(decoder_name, attn_implementation="eager")
model = PrefixedLLM(encoder=clip_encoder, decoder=decoder)

# load model weights from checkpoint
state_dict = torch.load(f"{model_path}/model_state.pt", map_location="cpu")
model.load_state_dict(state_dict, strict=True)

clip_processor = CLIPProcessor.from_pretrained(clip_name)
tokenizer = AutoTokenizer.from_pretrained(decoder_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
model.generation_config.pad_token_id = tokenizer.eos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id

# prepare dataset
test_dataset = PrecomputedTensorDataset(test_pt_path, limit_n = 0, shuffle = False, seed = 42)

# use gpu if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# GENERATE CANDIDATE CAPTIONS

model.eval()

# Batch size can be increased depending on VRAM
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=default_data_collator)

all_captions = []

for batch in tqdm(test_loader, desc="Generating captions"):
    pixel_values = batch["pixel_values"].to(device)
    
    # Note: 'attention_mask' from the dataset is for the Training Labels (text).
    # it is NOT used here because we are generating new text from scratch.
    
    with torch.no_grad():
        generated_ids = greedy_generate_prefixed(
            model=model,
            #tokenizer=tokenizer, # passed only to check EOS id
            pixel_values=pixel_values,
            #max_new_tokens=20,
            #device=device
        )

    # decode (iterate over the batch of generated ids)
    captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    all_captions.extend(captions)

# access the labels tensor directly from the dataset object (test_dataset.labels exists because it was defined it in __init__)
ref_labels_tensor = test_dataset.labels.clone()

# sanitize the labels ( they contain -100 for non-text parts (like image prefix), which must be replaced with the pad_token before decoding)
ref_labels_tensor[ref_labels_tensor == -100] = tokenizer.pad_token_id

# decode the labels back to text
print("Decoding reference captions...")
ref_captions_text = tokenizer.batch_decode(ref_labels_tensor, skip_special_tokens=True)

# print("Reference Captions") # DEBUGGING
# print(ref_captions_text) # DEBUGGING
# print("All Captions") # DEBUGGING
# print(all_captions) # DEBUGGING

#----------------------------------------------------------------------------------------------------------------------------
# EVALUATION

ref_captions = ref_captions_text # for readability in the rest of the script

# corpus bleu wants the reference translation(s) as a list of lists, where each sublist contans 1 reference per hypothesis
#ref_captions_corpus_bleu = [[caption for caption in test_dataset["caption"]]] # for corpus bleu

ref_captions_chrf = [[caption] for caption in ref_captions] # each element must be a list of references for that hypothesis

# # calculate overall BLEU
# bleu = sacrebleu.corpus_bleu(all_captions, ref_captions_corpus_blue)
# print(f"BLEU score: {bleu.score}")

# # calculate sentence-level BLEU
# bleu_per_instance = [sacrebleu.sentence_bleu(cap, [ref]) for cap, ref in zip(all_captions, ref_captions)]

# ChrF++
chrf = sacrebleu.corpus_chrf(all_captions, ref_captions_chrf, word_order=True)
chrf_per_instance = []
for capt, ref in zip(all_captions, ref_captions_chrf):
    c_score = sacrebleu.sentence_chrf(capt, ref, word_order=True)
    chrf_per_instance.append(c_score.score)

# CLIP-Score & Ref-CLIP-Score
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1').to(device)

clip_model.eval()

clip_score_per_instance = []
ref_clip_score_per_instance = []
caption_current_batch_start = 0 

for batch in tqdm(test_loader, desc="Evaluating CLIP similarity"):

    # get image_features
    pixel_values = batch["pixel_values"].to(device)
    with torch.no_grad(): 
        image_features = clip_model.get_image_features(pixel_values=pixel_values)

    # get caption features
    caption_current_batch_end = caption_current_batch_start + len(pixel_values)
    batch_captions = all_captions[caption_current_batch_start : caption_current_batch_end] 
    
    # FIX: Use the now-defined 'ref_captions' (list of strings)
    batch_refs = ref_captions[caption_current_batch_start : caption_current_batch_end] 
    
    caption_current_batch_start = caption_current_batch_end

    with torch.no_grad(): 
        text_features = text_model.encode(batch_captions, convert_to_tensor=True)
        reference_features = text_model.encode(batch_refs, convert_to_tensor=True)

    # normalize 
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features  = F.normalize(text_features,  p=2, dim=-1)
    reference_features = F.normalize(reference_features, p=2, dim=-1)

    # CLIP-SCORE
    batch_sims = (image_features * text_features).sum(dim=-1) 
    clip_score_per_instance.extend(batch_sims.tolist())

    # REF-CLIP SCORE
    batch_ref_text_sims = (text_features * reference_features).sum(dim=-1)
    
    clamped_batch_sims = torch.clamp(batch_sims, min=0.0)
    clamped_batch_ref_text_sims = torch.clamp(batch_ref_text_sims, min=0.0)
    
    numer = 2 * clamped_batch_sims * clamped_batch_ref_text_sims
    denom = clamped_batch_sims + clamped_batch_ref_text_sims
    harmonic_mean = torch.where(denom == 0, torch.zeros_like(denom), numer/denom) 
    ref_clip_score_per_instance.extend(harmonic_mean.tolist())

#clip_score_mean = np.mean(clip_score_per_instance)
clip_score_median = np.median(clip_score_per_instance)
ref_clip_score_median = np.median(ref_clip_score_per_instance)

# BERTSCORE

base_bert_name = "dbmdz/bert-base-italian-uncased" # base model
bert_name = "../recipe_bert/models/ft_bert_it_ep9" # my adapted model"
batch_size_embeddings = 128     # batch size for embeddings
batch_size_bertscore = 64       # batch size for BERTScore

# use gpu if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# load and save model (base model) -- this workaround is necessary due to permission issues
cache_folder = "./tmp"
base_bert_tokenizer = AutoTokenizer.from_pretrained(base_bert_name, cache_dir=cache_folder)
base_bert_model = AutoModel.from_pretrained(base_bert_name, cache_dir=cache_folder)
base_model_folder = "./tmp/base_bert_it"
base_bert_model.save_pretrained(base_model_folder)
base_bert_tokenizer.save_pretrained(base_model_folder)
# base_bert_model.to(device).eval()

# calculate bertscore (base model)
num_layers = base_bert_model.config.num_hidden_layers
base_P, base_R, base_F1 = score(all_captions, ref_captions, model_type=base_model_folder, num_layers=num_layers, lang=None, batch_size=batch_size_bertscore, device=device, verbose=True)
base_bert_f1_list = base_F1.tolist()
base_bert_score_median = np.median(base_bert_f1_list)

# # load model (adapted model)
# bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
bert_model = AutoModel.from_pretrained(bert_name)
# bert_model.to(device).eval()

# calculate bertscore (adapted model)
num_layers = bert_model.config.num_hidden_layers
P, R, F1 = score(all_captions, ref_captions, model_type=bert_name, num_layers=num_layers, lang=None, batch_size=batch_size_bertscore, device=device, verbose=True)
bert_f1_list = F1.tolist()
bert_score_median = np.median(bert_f1_list)

print("BERTScore computed")

# write to file
with open(output_scores_filepath, mode="w", encoding="utf-8") as f:
    # write overall scores
    f.write(f"OVERALL SCORE\n")
    f.write(f"CLIP-Score\tRef-CLIP-Score\tChrF++\tBERTScore (Baseline)\tBERTScore (Adapted)\n")
    f.write(f"{clip_score_median}\t{ref_clip_score_median}\t{chrf.score}\t{base_bert_score_median}\t{bert_score_median}\n")

    f.write(f"SCORE PER INSTANCE\n")
    f.write(f"Instance\tGeneratedCaption\tReferenceCaption\tCLIP-Score\tRefCLIP-Score\tCharF++\tBERTScore (Baseline)\tBERTScore (Adapted)\n")
    
    # write scores per instance
    for i, (gen_capt, ref_capt, sim, ref_sim, chrfscore, base_bscore, bscore) in enumerate(zip(all_captions, ref_captions, clip_score_per_instance, ref_clip_score_per_instance, chrf_per_instance, base_bert_f1_list, bert_f1_list)):
        f.write(f"{i+1}\t{gen_capt}\t{ref_capt}\t{sim}\t{ref_sim}\t{chrfscore}\t{base_bscore}\t{bscore}\n")

print("file saved")
