import numpy as np
import torch
from transformers import AutoTokenizer, AutoImageProcessor

COLOR_MAP = {
    "red":   (255, 0, 0),
    "green": (0, 255, 0),
    "blue":  (0, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
}

model_name = "ydshieh/vit-gpt2-coco-en"

def make_color_image_np(color_name: str, size: int = 224) -> np.ndarray:
    rgb = np.array(COLOR_MAP[color_name], dtype=np.uint8)
    return np.full((size, size, 3), rgb, dtype=np.uint8)

def build_toy_color_dataset(
    N_per_color: int,
    output_path: str,
    image_size: int = 224,
    max_length: int = 16,
):
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # pad token for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pixel_values_list, labels_list, dec_attn_mask_list = [], [], []

    for color in COLOR_MAP:
        caption = f"a {color} square" + tokenizer.eos_token
        for _ in range(N_per_color):
            img = make_color_image_np(color, size=image_size)
            pv = image_processor(images=img, return_tensors="pt").pixel_values[0]

            tok = tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            input_ids = tok.input_ids[0]
            dec_mask = tok.attention_mask[0]

            labels = input_ids.clone()
            labels[dec_mask == 0] = -100

            pixel_values_list.append(pv)
            labels_list.append(labels)
            dec_attn_mask_list.append(dec_mask)

    tensor_data = {
        "pixel_values": torch.stack(pixel_values_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(dec_attn_mask_list),
    }

    torch.save(tensor_data, output_path)
    return tensor_data

if __name__ == "__main__":
    build_toy_color_dataset(5, f"data/toy_colors_{model_name.split('/')[-1]}.pt", 224, 16)