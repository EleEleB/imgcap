import torch
from PIL import Image
from transformers import CLIPProcessor, AutoTokenizer
from tqdm import tqdm # Recommended for progress tracking
import os

# Import loading function from your existing utility
from data_utils import load_dataset

# Configuration
LANG = "it" 
SPLIT = 'test'
DATASET_PATH = f"./data/{SPLIT}_{LANG}.txt"
OUTPUT_PATH = f"./data/{SPLIT}_processed_{LANG}.pt"
CLIP_NAME = "openai/clip-vit-base-patch32"
DECODER_NAME = "ai-forever/mGPT"

def preprocess_and_save():
    # 1. Initialize Processors
    print("Initializing processors...")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME)
    decoder_tokenizer = AutoTokenizer.from_pretrained(DECODER_NAME)
    
    # Handle padding token if missing
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
    decoder_tokenizer.padding_side = 'right'

    # 2. Load Raw Data
    print(f"Loading raw data from {DATASET_PATH}...")
    raw_data = load_dataset(DATASET_PATH)
    
    # Storage lists
    pixel_values_list = []
    labels_list = []
    attention_mask_list = []

    # 3. Process Instances
    print("Processing images and tokens...")
    for example in tqdm(raw_data):
        try:
            # A. Image Processing
            image = Image.open(example["image_path"]).convert("RGB")
            # Returns dictionary with 'pixel_values'
            pv = clip_processor(images=image, return_tensors="pt").pixel_values[0] # Shape: [3, H, W]
            
            # B. Text Processing
            encodings = decoder_tokenizer(
                example["caption"], 
                padding="max_length", 
                truncation=True, 
                max_length=64, 
                return_tensors="pt"
            )
            
            input_ids = encodings.input_ids[0]
            attn_mask = encodings.attention_mask[0]
            
            # Replace padding token id with -100 for loss computation
            input_ids[input_ids == decoder_tokenizer.pad_token_id] = -100

            # C. Collect
            pixel_values_list.append(pv)
            labels_list.append(input_ids)
            attention_mask_list.append(attn_mask)

        except Exception as e:
            print(f"Error processing {example['image_path']}: {e}")
            continue

    # 4. Stack and Save
    if len(pixel_values_list) == 0:
        raise ValueError("No data processed.")

    print("Stacking tensors...")
    tensor_data = {
        "pixel_values": torch.stack(pixel_values_list),   # (N, 3, 224, 224)
        "labels": torch.stack(labels_list),               # (N, Seq_Len)
        "attention_mask": torch.stack(attention_mask_list) # (N, Seq_Len)
    }

    print(f"Saving to {OUTPUT_PATH}...")
    torch.save(tensor_data, OUTPUT_PATH)
    print("Done.")

if __name__ == "__main__":
    preprocess_and_save()