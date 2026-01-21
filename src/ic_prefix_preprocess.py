import torch
from PIL import Image
from transformers import CLIPProcessor, AutoTokenizer
from tqdm import tqdm
from lib_data_utils import load_dataset

lang = "it" 
split = "test" # "test", "train", "eval"
dataset_path = f"./data/{split}_{lang}.txt"
output_path = f"./data/{split}_processed_{lang}.pt"
clip_name = "openai/clip-vit-base-patch32"
decoder_name = "ai-forever/mGPT"

# preprocess dataset only once and save them as tensors to speed up loading during training

# initialize processors
print("Initializing processors...")
clip_processor = CLIPProcessor.from_pretrained(clip_name)
decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)

# handle padding token if missing
if decoder_tokenizer.pad_token is None:
    decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
decoder_tokenizer.padding_side = 'right'

# load raw data
print(f"Loading raw data from {dataset_path}...")
raw_data = load_dataset(dataset_path)

# storage lists
pixel_values_list = []
labels_list = []
attention_mask_list = []

# process instances
print("Processing images and tokens...")
for example in tqdm(raw_data):
    try:
        # image processing
        image = Image.open(example["image_path"]).convert("RGB")
        # returns dictionary with 'pixel_values'
        p_values = clip_processor(images=image, return_tensors="pt").pixel_values[0] # shape: [3, H, W]
        
        # text processing
        encodings = decoder_tokenizer(example["caption"], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        
        input_ids = encodings.input_ids[0]
        attn_mask = encodings.attention_mask[0]
        
        # replace padding token id with -100 for loss computation
        for i in range(len(input_ids)):
            if input_ids[i] == decoder_tokenizer.pad_token_id:
                input_ids[i] = -100

        # collect
        pixel_values_list.append(p_values)
        labels_list.append(input_ids)
        attention_mask_list.append(attn_mask)

    except Exception as e:
        print(f"Error processing {example['image_path']}: {e}")
        continue

# stack and save
if len(pixel_values_list) == 0:
    raise ValueError("No data processed.")

print("Stacking tensors...")
tensor_data = {
    "pixel_values": torch.stack(pixel_values_list),   # (N, 3, 224, 224)
    "labels": torch.stack(labels_list),               # (N, Seq_Len)
    "attention_mask": torch.stack(attention_mask_list) # (N, Seq_Len)
}

print(f"Saving to {output_path}...")
torch.save(tensor_data, output_path)
print("Done.")

