import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer

# Ensure project root is on sys.path when running this script from a subdirectory.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataloader.dataloader_augmentated import MSRAugmentedDataset
from models.decoder_prep.decoder_x import DecoderX
from models.encoder_prep.encoder import TextEncoder
from models.forgetting_model import ForgettingModel
from models.slot_pooling_prep.slot_pooling import SlotPooling
from models.uv_heads_prep.u_head import UHead
from models.uv_heads_prep.v_head import VHead
from models.g_psi_module.semantic_projection import SemanticProjectionModule
from models.g_psi_module.g_psi_config import G_psi_config


def build_model(device: torch.device) -> ForgettingModel:
    encoder = TextEncoder()
    slot_pool = SlotPooling(hidden_dim=encoder.hidden_dim_size, num_slots=8)
    u_head = UHead(hidden_dim=encoder.hidden_dim_size, output_dim=128)
    v_head = VHead(hidden_dim=encoder.hidden_dim_size)
    decoder_x = DecoderX()
    g_psi = SemanticProjectionModule(config=G_psi_config,no_use_u=True,no_use_vt=True)

    model = ForgettingModel(
        encoder=encoder,
        slot_pooling=slot_pool,
        u_head=u_head,
        v_head=v_head,
        decoder_x=decoder_x,
        g_psi=g_psi
    )
    return model.to(device)


def load_checkpoint(model: ForgettingModel, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "train_loss": checkpoint.get("train_loss"),
            "val_loss": checkpoint.get("val_loss"),
        }
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        metadata = {"epoch": None, "train_loss": None, "val_loss": None}
    else:
        raise ValueError("Unsupported checkpoint format. Expected dict with model weights.")

    model.load_state_dict(state_dict)
    model.eval()
    return metadata


@torch.no_grad()
def run_inference(
    model: ForgettingModel,
    dataloader: DataLoader,
    tokenizer: T5Tokenizer,
    max_samples: int,
    log_every: int,
    use_wandb: bool,
    latents_output_path: str,
) -> Tuple[float, List[Dict[str, str]], int, Dict[str, Any]]:
    total_loss = 0.0
    num_batches = 0
    total_examples = 0
    collected_samples: List[Dict[str, str]] = []
    all_u: List[torch.Tensor] = []
    all_v0: List[torch.Tensor] = []
    all_texts: List[str] = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference"), start=1):
        loss, logits_x, _, _ = model(batch)
        u, v0 = model.encode_latents(batch)
        all_u.append(u.detach().cpu())
        all_v0.append(v0.detach().cpu())
        batch_texts = tokenizer.batch_decode(batch["x_input_ids"], skip_special_tokens=True)
        all_texts.extend(batch_texts)

        loss_value = loss.item()
        total_loss += loss_value
        num_batches += 1

        batch_size = batch["x_input_ids"].size(0)
        total_examples += batch_size

        if use_wandb and (batch_idx % log_every == 0):
            wandb.log({"inference/batch_loss": loss_value, "inference/batch_idx": batch_idx})

        if len(collected_samples) >= max_samples:
            continue

        pred_ids_x = torch.argmax(logits_x, dim=-1).detach().cpu()

        decoded_x_pred = tokenizer.batch_decode(pred_ids_x, skip_special_tokens=True)

        decoded_x_true = tokenizer.batch_decode(batch["x_input_ids"], skip_special_tokens=True)

        remaining = max_samples - len(collected_samples)
        for x_true, x_pred in zip(
            decoded_x_true[:remaining],
            decoded_x_pred[:remaining],
        ):
            collected_samples.append(
                {
                    "x_true": x_true,
                    "x_pred": x_pred,
                }
            )

    if num_batches == 0:
        raise RuntimeError("Inference dataloader is empty.")

    if all_u:
        latents_parent = os.path.dirname(latents_output_path)
        if latents_parent:
            os.makedirs(latents_parent, exist_ok=True)
        latents_payload = {
            "u": torch.cat(all_u, dim=0),
            "v0": torch.cat(all_v0, dim=0),
            "texts": all_texts,
        }
        torch.save(latents_payload, latents_output_path)
    else:
        latents_payload = {"u": None, "v0": None, "texts": []}

    avg_loss = total_loss / num_batches
    return avg_loss, collected_samples, total_examples, latents_payload


def log_samples_to_wandb(samples: List[Dict[str, str]], use_wandb: bool) -> None:
    if not use_wandb or not samples:
        return

    table = wandb.Table(columns=["x_true", "x_pred"])
    for row in samples:
        table.add_data(row["x_true"], row["x_pred"])

    wandb.log({"inference/samples": table})


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference script for ForgettingModel with W&B logging")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data-path", type=str, required=True, help="Path to inference dataset JSON")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-samples", type=int, default=50, help="Max decoded samples to save/log")
    parser.add_argument("--log-every", type=int, default=10, help="Log batch loss every N batches")
    parser.add_argument(
        "--output-json",
        type=str,
        default="./output/p0/inference/forgetting_model_predictions.json",
        help="Where to save decoded predictions",
    )
    parser.add_argument(
        "--latents-output",
        type=str,
        default="./data/latents/inference/test_latents_p0_output.pt",
        help="Where to save extracted u/v0 latents during inference",
    )
    parser.add_argument("--wandb-project", type=str, default="diffusion-as-memory", help="W&B project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-off", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    if args.log_every <= 0:
        raise ValueError("--log-every must be a positive integer")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = MSRAugmentedDataset(args.data_path, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(device)
    checkpoint_meta = load_checkpoint(model, args.model_path, device)

    use_wandb = not args.wandb_off
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_path": args.model_path,
                "data_path": args.data_path,
                "batch_size": args.batch_size,
                "max_samples": args.max_samples,
                "device": str(device),
                "checkpoint_epoch": checkpoint_meta.get("epoch"),
                "checkpoint_train_loss": checkpoint_meta.get("train_loss"),
                "checkpoint_val_loss": checkpoint_meta.get("val_loss"),
            },
        )

    avg_loss, samples, total_examples, _ = run_inference(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        log_every=args.log_every,
        use_wandb=use_wandb,
        latents_output_path=args.latents_output,
    )

    output_parent = os.path.dirname(args.output_json)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Inference complete. Avg loss: {avg_loss:.4f}")
    print(f"Examples processed: {total_examples}")
    print(f"Saved decoded predictions to: {args.output_json}")
    print(f"Saved extracted latents to: {args.latents_output}")

    if use_wandb:
        wandb.log(
            {
                "inference/avg_loss": avg_loss,
                "inference/num_examples": total_examples,
                "inference/num_samples_logged": len(samples),
                "inference/latents_output": args.latents_output,
            }
        )
        log_samples_to_wandb(samples, use_wandb=True)
        wandb.run.summary["avg_loss"] = avg_loss
        wandb.run.summary["num_examples"] = total_examples
        wandb.finish()


if __name__ == "__main__":
    main()
