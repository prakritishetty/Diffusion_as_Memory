import json
import torch
from torch.utils.data import Dataset

class MSRAugmentedDataset(Dataset):
    def __init__(self, input_file_path, tokenizer, drop_prob=0.15):
        with open (input_file_path,"r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.drop_prob = drop_prob
        self.max_xt_items = 10
        self.max_length = 64

        self.pad_id = tokenizer.pad_token_id
        self.cls_like_ids = set(filter(lambda x: x is not None, 
            [
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
            ]
        ))
        
    def __len__(self):
        return len(self.data)
    
    def _drop_tokens(self, input_ids, attention_mask):
        # Randomly drop tokens with a certain probability
        # get valid tokens first from attention mask, and no special tokens to be dropped.
        valid = attention_mask.bool()
        special_tokens_mask = torch.zeros_like(valid)
        for sid in self.cls_like_ids:
            special_tokens_mask = special_tokens_mask | (input_ids == sid)

        # possibble tokens to drop
        candidates = valid & ~special_tokens_mask & (input_ids != self.pad_id)

        L = input_ids.size(0)
        drop_mask = torch.zeros_like(candidates)
        rand = torch.rand(L, device=input_ids.device)
        # this is boolean indexing, only selects candidates set to true, to assign values - only at true positions.
        # if for valid candidates, the random value is less than drop_prob, set it to true - it is dropped
        drop_mask[candidates] = rand[candidates] < self.drop_prob

        # this is of length L, with true at positions to keep.
        keep_mask = valid & (~drop_mask)  
        if keep_mask.sum().item() <= 1:
            return input_ids
        # drop all the false, this returns all the true indexes
        kept_ids = input_ids[keep_mask]

        # repad to length L, with pad_id
        new_ids = torch.full_like(input_ids, self.pad_id)
        new_ids[: kept_ids.size(0)] = kept_ids[:L]
        return new_ids
    
    def _tokenize(self, text, max_length):
        result = self.tokenizer(text, padding = "max_length", truncation = True, max_length = max_length, return_tensors="pt")
        return result

    def __getitem__(self, idx):
        # take 1 index and gve one training eg
        example = self.data[idx] #dict

        x = example["x"]
        xt_list = example.get("xt", [x])
        # y = example["y"]
        
        encoded_x = self._tokenize(x, 64)
        # encoded_y = self._tokenize(y, 32)

        x_input_ids = encoded_x["input_ids"].squeeze(0)
        x_attention = encoded_x["attention_mask"].squeeze(0)
        # xpos_input_ids = self._drop_tokens(x_input_ids, x_attention)
        # xpos_attention = (xpos_input_ids != self.pad_id).long()

        xt_count = min(len(xt_list), self.max_xt_items)
        xt_input_ids = torch.full(
            (self.max_xt_items, self.max_length), self.pad_id, dtype=torch.long
        )
        for i in range(xt_count):
            enc_xt = self._tokenize(xt_list[i],64)
            xt_input_ids[i] = enc_xt["input_ids"].squeeze(0)

        return{
            "x_text": x,
            "x_input_ids": x_input_ids,
            "x_attention": x_attention,
            "xpos_input_ids": x_input_ids,
            "xpos_attention": x_attention,
            "xt_input_ids": xt_input_ids, # [max_xt_items, seq_len]
            "xt_count": torch.tensor(xt_count, dtype=torch.long),
        }