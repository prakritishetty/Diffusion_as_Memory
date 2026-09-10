import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class DecoderY(nn.Module):
    def __init__(self, hidden_dim, u_dim, num_slots, model_name = "t5-small"):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.u_proj = nn.Linear(u_dim, hidden_dim)

    def forward(self, u ,labels):
        B = u.size(0)
        u_proj = self.u_proj(u)
        u_slots = u_proj.unsqueeze(1).expand(B, self.num_slots, self.hidden_dim)

        attention_mask = torch.ones(B, self.num_slots, device = u.device)

        outputs = self.model(
            encoder_outputs = (u_slots,),
            attention_mask = attention_mask,
            labels = labels,
            return_dict = True
        )

        return outputs.loss, outputs.logits
