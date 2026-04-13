import json
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

model_name = "t5-base"
BATCH_SIZE = 2
tokeniser = AutoTokenizer.from_pretrained(model_name)

class MSRDataset(Dataset):
    def __init__(self, input_file_path):
        with open (input_file_path,"r") as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # take 1 index and gve one training eg
        example = self.data[idx] #dict

        x = example["x"]
        xpos = example["x+"]
        y = example["summary"]

        def tokenize(x, max_length):
            result = tokeniser(x, padding = "max_length", truncation = True, max_length = max_length, return_tensors="pt")
            return result
        
        encoded_x = tokenize(x, 64)
        encoded_xpos = tokenize(xpos, 64)
        encoded_y = tokenize(y, 32)

        x_input_ids = encoded_x["input_ids"].squeeze(0)
        x_attention = encoded_x["attention_mask"].squeeze(0)
        xpos_input_ids = encoded_xpos["input_ids"].squeeze(0)
        xpos_attention = encoded_xpos["attention_mask"].squeeze(0)
        y_input_ids = encoded_y["input_ids"].squeeze(0)
        y_attention = encoded_y["attention_mask"].squeeze(0)

        return{
            "x_input_ids": x_input_ids,
            "x_attention": x_attention,
            "xpos_input_ids": xpos_input_ids,
            "xpos_attention": xpos_attention,
            "y_input_ids": y_input_ids,
            "y_attention": y_attention,
        }

# dataset = MSRDataset("../dataset_prep/trial_train_data.json")
# dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True)
# batch = next(iter(dataloader))

# print(batch["x_input_ids"].shape)
# print(batch["xp_input_ids"].shape)
# print(batch["y_input_ids"].shape)

    
    
