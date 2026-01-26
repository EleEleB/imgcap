from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
import torch

# class that loads pre-computed tensors from a .pt file with optional deterministic shuffling.
class PrecomputedTensorDataset(TorchDataset):
    def __init__(self, pt_file_path, limit_n=0, shuffle=False, seed=42):
        data = torch.load(pt_file_path, map_location="cpu")
        attention_mask = data['attention_mask']
        pixel_values = data["pixel_values"]
        labels = data["labels"]

        n = len(labels)
        if shuffle:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(n, generator=g)
            attention_mask = attention_mask[perm]
            pixel_values = pixel_values[perm]
            labels = labels[perm]

        if limit_n > 0:
            attention_mask = attention_mask[:limit_n]
            pixel_values = pixel_values[:limit_n]
            labels = labels[:limit_n]

        self.attention_mask = attention_mask
        self.pixel_values = pixel_values
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
        "pixel_values": self.pixel_values[idx],
        "labels": self.labels[idx],
        "decoder_attention_mask": self.attention_mask[idx],
        }

# function that imports the dataset from a txt file
# parameter dataset_path is the file path
# returns the extracted dataset
def load_dataset(dataset_path):
    # load dataset from file
    with open(dataset_path, mode="r", encoding="utf-8") as f:
        txt = f.read()
    temp_dataset = txt.split("\n")

    dataset = []
    for el in temp_dataset:
        if el != "":
            temp = el.split("\t")
            dataset.append({"image_path": temp[0], "caption": temp[1]})
    return dataset


# function that preprocesses the a training instance and puts it in the format the model wants (NOT BATCHED)
# parameter example is one instance
# parameter image_processor is the image processor
# parameter decoder tokenizer is the tokenizer
# returns the data in the format the model wants
def preprocess_example(example, image_processor, text_tokenizer):
    # load image
    image = Image.open(example["image_path"]).convert("RGB")

    # use CLIPProcessor to preprocess images (returns PyTorch tensors with correct size and normalization)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values[0] # shape: [3, H, W]

    # Tokenize caption
    temp = text_tokenizer(example["caption"], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    labels = temp.input_ids[0]  # the actual tokenized input

    # the attention mask (1 for a real token and 0 for a padding token) ensures the model ignores padding tokens during the training
    attention_mask = temp.attention_mask[0]

    # replace padding token id with -100 to ignore in loss (otherwise loss gets computed on padding)
    labels[labels == text_tokenizer.pad_token_id] = -100

    return {"pixel_values": pixel_values, "labels": labels, "attention_mask": attention_mask}


# function that preprocesses the a batch of training instances and puts it in the format the model wants
# parameter batch is a list of instances
# parameter image_processor is the image processor
# parameter decoder tokenizer is the tokenizer
# returns the data in the format the model wants
def preprocess_batch(batch, image_processor, text_tokenizer):
    # load images
    images = []
    for path in batch["image_path"]:
        image = Image.open(path).convert("RGB")
        images.append(image)

    # use CLIPProcessor to preprocess images (returns PyTorch tensors with correct size and normalization)
    pixel_values = image_processor(images=images, return_tensors="pt").pixel_values  # shape: [batch_size, 3, H, W]

    # tokenize captions
    temp = text_tokenizer(batch["caption"], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    labels = temp.input_ids # the actual tokenized input
    
    # the attention mask (1 for a real token and 0 for a padding token) ensures the model ignores padding tokens during the training
    attention_mask = temp.attention_mask

    # replace padding token id with -100 to ignore in loss (otherwise loss gets computed on padding)
    labels[labels == text_tokenizer.pad_token_id] = -100

    return {"pixel_values": pixel_values, "labels": labels, "attention_mask": attention_mask,}



# function that prepares the dataset from filepath to mapping
# parameter dataset_path is the filepath of the txt file containing the dataset
# parameter image_processor is the image processor for the image part of the dataset
# parameter text_tokenizer is the tokenizer for the text part of the dataset
# parameter n is only passed during tests to have them performed on only a portion of the dataset for speed
# parameter batched specifies whether the data should be handled in batches or not
# returns the mapped dataset
def prepare_dataset(dataset_path, image_processor, text_tokenizer, n = 0, batched = False):
    data = load_dataset(dataset_path)

    # during tests, n indicates how many instances of the dataset should be used
    if n > 0:
        data = data[:n]

    dataset = Dataset.from_list(data)

    if batched == False:
        dataset = dataset.map(lambda x: preprocess_example(x, image_processor, text_tokenizer), batched=False) # preprocess pictures the way the model wants them
    else:
        dataset = dataset.map(lambda x: preprocess_batch(x, image_processor, text_tokenizer), batched=True, batch_size=8) # preprocess pictures the way the model wants them
        
    return dataset