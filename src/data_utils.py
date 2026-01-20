from PIL import Image
from datasets import Dataset
import torch

class PrecomputedTensorDataset(Dataset):
    """
    Loads pre-computed tensors from a .pt file with optional deterministic shuffling.
    """
    def __init__(self, pt_file_path, limit_n=0, shuffle=False, seed=42):
        print(f"Loading tensors from {pt_file_path}...")
        data = torch.load(pt_file_path, map_location="cpu") # Load to CPU RAM first
        
        # Extract tensors
        pixel_values = data["pixel_values"]
        labels = data["labels"]
        attention_mask = data["attention_mask"]
        
        total_samples = len(labels)

        # Apply deterministic shuffling
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(seed)
            permutation = torch.randperm(total_samples, generator=generator)
            
            pixel_values = pixel_values[permutation]
            labels = labels[permutation]
            attention_mask = attention_mask[permutation]
        
        # Apply limit_n (truncation)
        if limit_n > 0:
            self.pixel_values = pixel_values[:limit_n]
            self.labels = labels[:limit_n]
            self.attention_mask = attention_mask[:limit_n]
        else:
            self.pixel_values = pixel_values
            self.labels = labels
            self.attention_mask = attention_mask
            
        self.length = len(self.labels)
        print(f"Loaded {self.length} samples (Shuffle={shuffle}, Seed={seed}).")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Returns dict strictly matching the Trainer expects
        return {
            "pixel_values": self.pixel_values[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_mask[idx]
        }

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
def preprocess_example(example, clip_processor, decoder_tokenizer):
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
def prepare_dataset(dataset_path, clip_processor, decoder_tokenizer, n = 0):
  data = load_dataset(dataset_path)
  if n > 0:
      data = data[:n]
  dataset = Dataset.from_list(data)

  # preprocess pictures the way the model wants them
  dataset = dataset.map(lambda x: preprocess_example(x, clip_processor, decoder_tokenizer), batched=False) # map handles the calling so long as preprocess_example has only 1 argument

  return dataset