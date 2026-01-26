import torch
from PIL import Image
from transformers import CLIPProcessor, AutoTokenizer, ViTImageProcessor
from tqdm import tqdm
from lib_data_utils import load_dataset

encoder_name = "ydshieh/vit-gpt2-coco-en"
decoder_name = encoder_name

tokenizer = AutoTokenizer.from_pretrained(decoder_name)
img_processor = ViTImageProcessor.from_pretrained(encoder_name)

# handle padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
lang = "en"

for split in ['train', 'eval', 'test']:
    # split = "test" # "test", "train", "eval"
    dataset_path = f"./data/{split}_{lang}.txt"

    print("Initializing processors...")
    
    model_combo = f"{encoder_name.split('/')[-1]}_{decoder_name.split('/')[-1]}"
    output_path = f"./data/gz_tensors/{split}_{lang}.pt"

    # load raw data
    print(f"Loading raw data from {dataset_path}...")
    raw_data = load_dataset(dataset_path)

    # storage lists
    pixel_values_list = []
    labels_list = []
    attn_mask_list =    []
    # process instances
    print("Processing images and tokens...")
    for example in tqdm(raw_data): 
        try:
            # image processing
            image = Image.open(example["image_path"]).convert("RGB")
            p_values = img_processor(images=image, return_tensors="pt").pixel_values[0]
            
            # NOTE: here we need to add the eos token. the tokenizer won't do it on its own
            caption = example["caption"]
            caption = caption + tokenizer.eos_token
            
            encodings = tokenizer(
                caption, 
                padding="max_length", 
                max_length=64, 
                truncation=True, 
                return_tensors="pt"
            )
            labels = encodings.input_ids[0]
            attn_mask = encodings.attention_mask[0]
            labels[attn_mask == 0] = -100

            pixel_values_list.append(p_values)
            labels_list.append(labels)
            attn_mask_list.append(attn_mask)

        except Exception as e:
            print(f"Error processing {example['image_path']}: {e}")
            continue

    # stack and save
    if len(pixel_values_list) == 0:
        raise ValueError("No data processed.")

    print("Stacking tensors...")
    tensor_data = {
        "pixel_values": torch.stack(pixel_values_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attn_mask_list),
    }

    print(f"Saving to {output_path}...")
    torch.save(tensor_data, output_path)
    print("Done.")

