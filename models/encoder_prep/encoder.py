import torch
import torch.nn as nn
from transformers import T5EncoderModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="t5-small"):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.hidden_dim_size = self.encoder.config.d_model

    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = True
        )

        return outputs.last_hidden_state

