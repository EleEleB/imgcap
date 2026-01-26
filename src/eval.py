import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from transformers import VisionEncoderDecoderModel, ViTImageProcessor
from tqdm import tqdm
import sacrebleu
from transformers import CLIPModel
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from lib_data_utils import prepare_dataset
import json

lang = "en" # en
eval_dataset_path = f"./data/test_{lang}.txt"

model_name = "./models/vit-gpt2-coco-en_2026-01-26--04-50-56"
# model_name = "ydshieh/vit-gpt2-coco-en"

# load model
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

# prepare dataset
eval_dataset = prepare_dataset(eval_dataset_path, feature_extractor, tokenizer, n = 100)

# use gpu if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.eval()
batch_size = 8
test_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)
all_captions = []

for batch in tqdm(test_loader, desc="Generating captions"):
    # get pixel values and move to device
    pixel_values = batch["pixel_values"].to(device)

    with torch.no_grad(): # inference, so no gradients needed
        generated_captions = model.generate(pixel_values=pixel_values,
                                            # attention_mask=attention_mask,
                                            max_length=64,
                                            # num_beams=4,
                                            # early_stopping=True,
                                            )

    captions = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_captions] # skip_special_tokens ignores padding/eos tokens
    all_captions.extend(captions)


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

# write to file
results = {'scores': {}, 'preds': []}
results['scores'] = {
    'CLIP-Score': clip_score_median,
    'Ref-CLIP-Score': ref_clip_score_median,
    'Sacrebleu': bleu.score,
    'ChrF++': chrf.score,
    }
for i, (gen_capt, ref_capt, sim, ref_sim, bleuscore, chrfscore) in enumerate(zip(all_captions, ref_captions, clip_score_per_instance, ref_clip_score_per_instance, bleu_per_instance, chrf_per_instance)):
    results['preds'].append({
        'Instance': i,
        'GeneratedCaption': str(gen_capt),
        'ReferenceCaption': str(ref_capt),
        'CLIP-Score': str(sim),
        'RefCLIP-Score': str(ref_sim),
        'Sacrebleu': str(bleuscore),
        'CharF++': str(chrfscore),
    })

output_scores_filepath = (f"results/{model_name.split('/')[-1]}_scores.json")

with open(output_scores_filepath, 'w', encoding='utf8') as f:
    json.dump(results, f, ensure_ascii = False, indent = 4)

print(f'Scores saved to: {output_scores_filepath}')