import json
import torch
import torch.utils.data as Dataset
import random

class LatentDiffusionDataset:
    def __init__(self, input_file_path, tokenizer):
        with open (input_file_path,"r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = 64
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return input_ids and attention mask for x, xt, xprev
        
        :param self: Description
        :param idx: Description
        """
        example = self.data[idx] #dict

        x0 = example["x"]
        xt = example["xt"]

        # sample a random timestep t
        t = random.randint(1, len(xt) - 1)
        xprev = xt[t-1]

        # tokenize x, xt[t], xprev
        x0_tokens = self._tokenize(x0, 64)
        x0_input_ids = x0_tokens["input_ids"].squeeze(0)
        x0_attention = x0_tokens["attention_mask"].squeeze(0)

        xt_tokens = self._tokenize(xt[t], 64)
        xt_input_ids = xt_tokens["input_ids"].squeeze(0)
        xt_attention = xt_tokens["attention_mask"].squeeze(0)

        xprev_tokens = self._tokenize(xprev, 64)
        xprev_input_ids = xprev_tokens["input_ids"].squeeze(0)
        xprev_attention = xprev_tokens["attention_mask"].squeeze(0)


        return{
            "x0_input_ids": x0_input_ids,
            "x0_attention": x0_attention,
            "xt_input_ids": xt_input_ids,
            "xt_attention": xt_attention,
            "xprev_input_ids": xprev_input_ids,
            "xprev_attention": xprev_attention,
            "x0_text": x0,
            "xt_text": xt[t],
            "xprev_text": xprev,
            "t": t,
            "max_t": len(xt) - 1,
        }

    def _tokenize(self, text, max_length):
        result = self.tokenizer(text, padding = "max_length", truncation = True, max_length = max_length, return_tensors="pt")
        return result