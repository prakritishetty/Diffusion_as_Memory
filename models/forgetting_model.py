import torch
import torch.nn as nn
import torch.nn.functional as F


class ForgettingModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        slot_pooling: nn.Module,
        u_head: nn.Module,
        v_head: nn.Module,
        decoder_x: nn.Module,
        g_psi: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.slot_pooling = slot_pooling
        self.u_head = u_head
        self.v_head = v_head
        self.decoder_x = decoder_x
        self.g_psi = g_psi

    def info_nce_loss(self, u, upos, temperature=0.1):
        """
        InfoNCE loss. TODO: look for other loss functions
        """
        if u.shape != upos.shape:
            raise ValueError(f"u and upos must have the same shape, got {u.shape} vs {upos.shape}")

        u = F.normalize(u, dim=-1)
        upos = F.normalize(upos, dim=-1)

        # # Backward-compatible path for [B, D]
        # if u.dim() == 2:
        #     logits = torch.matmul(u, upos.T) / temperature
        #     labels = torch.arange(u.size(0), device=u.device)
        #     return F.cross_entropy(logits, labels)

        # Slot-wise contrastive path for [B, L, D]
        if u.dim() == 3:
            # For each slot index l, contrast examples across batch dimension B.
            # This keeps positives aligned as (b, l) <-> (b, l).
            u_slots = u.transpose(0, 1)       # [L, B, D]
            upos_slots = upos.transpose(0, 1) # [L, B, D]
            logits = torch.einsum("lbd,lcd->lbc", u_slots, upos_slots) / temperature  # [L, B, B]
            labels = torch.arange(u.size(0), device=u.device).unsqueeze(0).expand(u.size(1), -1)
            return F.cross_entropy(logits.reshape(-1, u.size(0)), labels.reshape(-1))

        raise ValueError(f"Expected u to be 2D or 3D, got tensor with {u.dim()} dims")
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, batch):
        device = self.device
        lambda_u, lambda_x = 1.0, 1.0

        input_ids = batch["x_input_ids"].to(device)
        attention_mask = batch["x_attention"].to(device)
        xpos_input_ids = batch["xpos_input_ids"].to(device)
        xpos_attention_mask = batch["xpos_attention"].to(device)
        labels_x = batch["x_input_ids"].to(device)
        # labels_y = batch["y_input_ids"].to(device)
        # y_attention_mask = batch["y_attention"].to(device)

        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        Hpos = self.encoder(input_ids=xpos_input_ids, attention_mask=xpos_attention_mask)
        # print("shape of outputs after encoder", H.shape) # b, 64, 512

        outputs = self.slot_pooling(H, attention_mask)
        # print("shape of outputs after slot pooling", outputs.shape) # b, 8, 512
        pos_outputs = self.slot_pooling(Hpos, xpos_attention_mask)

        u = self.u_head(outputs)
        upos = self.u_head(pos_outputs)
        v0 = self.v_head(outputs)

        B, L, _ = v0.shape
        slot_mask = torch.ones((B,L), device = device)

        use_u_for_v0 = True  # can make this a config later, this is an optional flag for u+v0 instead of v0

        if use_u_for_v0:
            t_zero = torch.zeros(B, dtype=torch.long, device=device)
            v0 = self.g_psi(v_hat_0=v0, t=t_zero)

        loss_x, logits_x = self.decoder_x(v0, slot_mask, labels_x)
        # loss_y, logits_y = self.decoder_y(u, labels_y)
        loss_nce = self.info_nce_loss(u, upos)

        total_loss = (
            lambda_u * loss_nce +
            lambda_x * loss_x
            # lambda_y * loss_y
        )

        return total_loss, logits_x, loss_nce, loss_x

    @torch.no_grad()
    def encode_latents(self, batch):
        """
        Extract latent representations (u, v0) without running decoders.
        """
        device = self.device

        input_ids = batch["x_input_ids"].to(device)
        attention_mask = batch["x_attention"].to(device)

        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.slot_pooling(H, attention_mask)

        u = self.u_head(outputs)    # [B, 128]
        v0 = self.v_head(outputs)   # [B, 8, 512]

        return u, v0


    @torch.no_grad()
    def encode_xt_latents(self, batch_input_ids, batch_attention_mask):
        """
        Adding this as a seperate function to not break other implementations.
        Ideally this should be used for all encoding, as it is more flexible and can be used for both x and xt.
        """
        device = self.device

        input_ids = batch_input_ids.to(device)
        attention_mask = batch_attention_mask.to(device)

        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.slot_pooling(H, attention_mask)

        u = self.u_head(outputs)    # [B, 128]
        v0 = self.v_head(outputs)   # [B, 8, 512]

        return u, v0
    
    @torch.no_grad()
    def decode_latents(self, v0, attention_mask, max_new_tokens=64, num_beams=4):
        """
        Decode latent for inference.
        """
        return self.decoder_x.generate(v0, attention_mask)