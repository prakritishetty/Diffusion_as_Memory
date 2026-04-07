import torch
import os
import json
import time
from models.encoder_prep.encoder import TextEncoder
from models.slot_pooling_prep.slot_pooling import SlotPooling
from models.uv_heads_prep.u_head import UHead
from models.uv_heads_prep.v_head import VHead
from models.decoder_prep.decoder_x import DecoderX
from models.forgetting_model import ForgettingModel
from models.g_psi_module.semantic_projection import SemanticProjectionModule
from models.g_psi_module.g_psi_config import G_psi_config


class ETATracker:
    """
    Tracks per-epoch wall-clock time and computes an estimated time remaining.
    """

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self._epoch_times: list[float] = []
        self._epoch_start: float | None = None
        self._completed: int = 0

    def start_epoch(self):
        """Call at the beginning of each epoch."""
        self._epoch_start = time.time()

    def end_epoch(self) -> tuple[float, float, str]:
        """
        Call at the end of each epoch.
        """
        if self._epoch_start is None:
            raise RuntimeError("end_epoch() called before start_epoch()")

        elapsed = time.time() - self._epoch_start
        self._epoch_times.append(elapsed)
        self._completed += 1
        self._epoch_start = None

        remaining = self.total_epochs - self._completed
        avg = sum(self._epoch_times) / len(self._epoch_times)
        eta_s = avg * remaining

        eta_h = int(eta_s // 3600)
        eta_m = int((eta_s % 3600) // 60)
        eta_s_part = int(eta_s % 60)
        eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_s_part:02d}"

        return elapsed, eta_s, eta_str

    def wandb_metrics(self, elapsed: float, eta_s: float) -> dict:
        """
        Returns a dict of W&B-ready timing metrics to pass to wandb.log() func.
        """
        return {
            "epoch_time_s": elapsed,
            "eta_seconds": eta_s,
        }


def build_p0_model(device, l_slots, u_dim):
    """
    Reconstruct the P0 ForgettingModel architecture (needed to load state dict).
    
    :param: device: "cuda" or "cpu"
    :param: l_slots: number of slots
    :param: u_dim: dimension of u vector
    """
    encoder = TextEncoder()
    slot_pool = SlotPooling(hidden_dim=encoder.hidden_dim_size, num_slots=l_slots)
    u_head = UHead(hidden_dim=encoder.hidden_dim_size)
    v_head = VHead(hidden_dim=encoder.hidden_dim_size)
    decoder_x = DecoderX()
    g_psi = SemanticProjectionModule(config=G_psi_config,no_use_u=True,no_use_vt=True)

    model = ForgettingModel(
        encoder=encoder,
        slot_pooling=slot_pool,
        u_head=u_head,
        v_head=v_head,
        decoder_x=decoder_x,
        g_psi=g_psi,
    )
    model.to(device)
    return model


def select_xt_labels(batch, t, device, xt_bucket_size):
    """Pick the right degraded xt label for each sample based on timestep.

    Bucket mapping: index = min(t // XT_BUCKET_SIZE, xt_count - 1)
    Variable-length handling: xt lists can be 1–10 items long.
    If a sample has only 3 items and t maps to index 5, it clamps
    to the last available item (index 2).
    """
    xt_input_ids = batch["xt_input_ids"].to(device)   # [B, max_xt_items, seq_len]
    xt_count = batch["xt_count"].to(device)            # [B]
    B = t.shape[0]

    raw_index = t // xt_bucket_size                           # [B]
    xt_index = torch.min(raw_index, xt_count - 1)             # clamp per sample
    labels = xt_input_ids[torch.arange(B, device=device), xt_index]  # [B, seq_len]
    return labels, xt_index


def log_sample_outputs(sample_outputs, tokenizer, epoch, output_dir, label_source="xt"):
    """Decode and save predictions for all validation batches."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for batch, logits_noisy, t_vals, xt_idx in sample_outputs:
        pred_noisy = tokenizer.batch_decode(
            torch.argmax(logits_noisy, dim=-1), skip_special_tokens=True
        )
        original = tokenizer.batch_decode(
            batch["x_input_ids"], skip_special_tokens=True
        )
        if label_source == "x":
            # Ablation mode: decoder target is x for all t.
            xt_target = tokenizer.batch_decode(
                batch["x_input_ids"], skip_special_tokens=True
            )
            xt_index_vals = torch.full((len(xt_target),), -1, dtype=torch.long)
        else:
            # Decode the xt target that was used as the noisy label.
            xt_all = batch["xt_input_ids"]  # [batch_size, max_xt_items, seq_len]
            batch_size = xt_all.shape[0]
            xt_target_ids = xt_all[torch.arange(batch_size), xt_idx.cpu()]
            xt_target = tokenizer.batch_decode(xt_target_ids, skip_special_tokens=True)
            xt_index_vals = xt_idx.cpu()

        for i in range(len(original)):
            results.append(
                {
                    "original": original[i],
                    "xt_target": xt_target[i],
                    "xt_index": xt_index_vals[i].item(),
                    "recon_noisy": pred_noisy[i],
                    "t": t_vals[i].item(),
                }
            )

    out_path = os.path.join(output_dir, f"epoch_{epoch + 1}_samples.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)


def convert_tokens_to_text_and_log(sample_outputs, tokenizer, epoch, output_dir):
    """Utility to decode token IDs to text."""
    results = []

    for batch, logits in sample_outputs:
        pred_texts = tokenizer.batch_decode(
            torch.argmax(logits, dim=-1), skip_special_tokens=True
        )
        x0_text = batch["x0_text"]
        xt_text = batch["xt_text"]
        xprev_text = batch["xprev_text"]
        t = batch["t"]
        for i in range(len(pred_texts)):
            results.append(
                {
                    "x0": x0_text[i],
                    "xt": xt_text[i],
                    "xprev": xprev_text[i],
                    "pred_"
                    "recon": pred_texts[i],
                    "t": t[i].item(),
                }
            )
    
    out_path = os.path.join(output_dir, f"epoch_{epoch + 1}_samples.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
def save_checkpoint(g_psi, decoder, optimizer, epoch, train_loss, val_loss, path):
    """Save Phase 3 checkpoint (G_psi + fine-tuned decoder)."""
    torch.save(
        {
            "epoch": epoch,
            "g_psi_state_dict": g_psi.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )
    

def save_denoiser_checkpoint(denoiser, optimizer, epoch, train_loss, val_loss, path):
    """Save Denoiser"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": denoiser.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )    


def save_decoder_gpsi_checkpoint(g_psi, decoder, optimizer, epoch, train_loss, val_loss, path):
    """Save Decoder and G_psi"""
    torch.save(
        {
            "epoch": epoch,
            "decoder_state_dict": decoder.state_dict(),
            "g_psi_state_dict": g_psi.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )    


def save_model_checkpoint(diffusion_model, optimizer, epoch, train_loss, val_loss, path):
    """Save Diffusion Model"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": diffusion_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )
