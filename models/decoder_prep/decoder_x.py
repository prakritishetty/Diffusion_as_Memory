import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class DecoderX(nn.Module):
    def __init__(self, model_name = "t5-small"):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self, encoder_hidden_states, attention_mask, labels):
        outputs = self.model(
            encoder_outputs = (encoder_hidden_states, ),
            attention_mask = attention_mask,
            labels = labels,
            return_dict = True
        )

        return outputs.loss, outputs.logits
    
    @torch.no_grad()
    def generate(
        self,
        encoder_hidden_states,
        attention_mask,
        max_new_tokens=64,
        num_beams=4,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    ):
        """
        Decodings for inference.
        """
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        generated_ids = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        return generated_ids