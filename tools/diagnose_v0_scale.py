"""Gate 0 -- is the latent at the scale the diffusion process assumes?

The forward process is `v_t = sqrt(alpha_bar)*v0 + sqrt(1-alpha_bar)*eps` with `eps ~ N(0, I)`.
That only produces *graded* corruption when `v0` has roughly unit variance. If `std(v0)` is much
larger than 1, the noise term is negligible until `alpha_bar` is tiny, so nothing degrades and
then everything degrades at once. That is a strong candidate explanation for the "cliff" observed
in earlier runs, and it is independent of the supervision-collapse diagnosis -- it was never
checked.

Ten minutes, and it decides whether the v-head needs LayerNorm or a fixed `sigma_data` rescale.

Works on a new checkpoint or on one of the archived P0 checkpoints under `reference/runs/`.

    python -m tools.diagnose_v0_scale --p0 reference/runs/saved_latent_models_outputs/.../best_model.pt
    python -m tools.diagnose_v0_scale --preset small --untrained
"""

from __future__ import annotations

import argparse
import json
import os

import torch


def effective_snr_table(v0_std: float, T: int = 1000, schedule: str = "cosine") -> list[dict]:
    """Signal-to-noise as it actually is, given the measured latent scale."""
    from diffusion.schedule import NoiseSchedule

    sched = NoiseSchedule(T=T, schedule_type=schedule)
    rows = []
    for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        t = max(1, int(frac * T))
        ab = float(sched.alpha_bar(torch.tensor([t])))
        signal = (ab ** 0.5) * v0_std
        noise = (1 - ab) ** 0.5
        rows.append({
            "t": t, "alpha_bar": round(ab, 6),
            "signal_rms": round(signal, 4), "noise_rms": round(noise, 4),
            "snr_db": round(20 * torch.log10(torch.tensor(max(signal / max(noise, 1e-9),
                                                              1e-9))).item(), 2),
        })
    return rows


@torch.no_grad()
def measure(model, loader, device, max_batches: int = 40) -> dict:
    model.eval()
    v_chunks, u_chunks = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        u, v0 = model.encode_latents(batch["x_input_ids"].to(device),
                                     batch["x_attention"].to(device))
        v_chunks.append(v0.float().cpu())
        u_chunks.append(u.float().cpu())
    v = torch.cat(v_chunks)
    u = torch.cat(u_chunks)
    per_dim = v.reshape(-1, v.shape[-1]).std(0)
    return {
        "n_examples": int(v.shape[0]),
        "v0_std": float(v.std()),
        "v0_mean": float(v.mean()),
        "v0_abs_max": float(v.abs().max()),
        "v0_per_dim_std_min": float(per_dim.min()),
        "v0_per_dim_std_max": float(per_dim.max()),
        "v0_per_slot_std": [round(float(s), 4) for s in v.std(dim=(0, 2))],
        "u_norm_mean": float(u.norm(dim=-1).mean()),
        "u_std": float(u.std()),
    }


def verdict(v0_std: float) -> tuple[str, str]:
    if 0.5 <= v0_std <= 2.0:
        return "OK", "latent is near unit scale; the schedule will degrade it gradually"
    if v0_std > 2.0:
        return "TOO LARGE", (
            f"noise is swamped until alpha_bar ~ {1/(v0_std**2):.4f}; expect no visible "
            "degradation and then a cliff. Enable model.v_head_normalize or set "
            f"v_head.set_data_scale({v0_std:.3f})."
        )
    return "TOO SMALL", (
        "signal is swamped immediately; even small t will look destroyed. Scale the latent up "
        f"by ~{1/max(v0_std,1e-6):.2f} or disable normalisation."
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gate 0: measure the detail-latent scale")
    ap.add_argument("--p0", default=None, help="P0 checkpoint; omit with --untrained")
    ap.add_argument("--preset", default="small", help="used when --untrained")
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/diagnostics/v0_scale.json")
    args = ap.parse_args(argv)

    from configs import get as get_preset
    from models.memory_model import MemoryModel
    from training.common import (
        build_cache, build_loaders, build_tokenizer, config_from_checkpoint, load_checkpoint,
    )

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.untrained or not args.p0:
        cfg = get_preset(args.preset)
        model = MemoryModel(cfg).to(device)
        source = f"untrained:{args.preset}"
    else:
        cfg = config_from_checkpoint(args.p0)
        model = MemoryModel(cfg).to(device)
        model.load_state_dict(load_checkpoint(args.p0, device)["model_state_dict"], strict=False)
        source = args.p0
    if args.data_dir:
        cfg = cfg.override(**{"data.data_dir": args.data_dir})

    tok = build_tokenizer(cfg)
    loaders = build_loaders(cfg, tok, build_cache(cfg), (args.split if args.split in
                                                         ("train", "val", "test") else "train",))
    loader = next(iter(loaders.values()))

    stats = measure(model, loader, device, args.batches)
    tag, advice = verdict(stats["v0_std"])
    stats.update({
        "source": source, "verdict": tag, "advice": advice,
        "v_head_normalize": cfg.model.v_head_normalize,
        "effective_snr": effective_snr_table(stats["v0_std"], cfg.diffusion.T,
                                             cfg.diffusion.schedule),
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nsource            {source}")
    print(f"v0 std            {stats['v0_std']:.4f}   (target ~1.0)")
    print(f"v0 mean           {stats['v0_mean']:+.4f}")
    print(f"v0 |max|          {stats['v0_abs_max']:.3f}")
    print(f"per-dim std       {stats['v0_per_dim_std_min']:.3f} .. "
          f"{stats['v0_per_dim_std_max']:.3f}")
    print(f"u norm            {stats['u_norm_mean']:.4f}")
    print(f"\nVERDICT: {tag} -- {advice}")
    print(f"\n{'t':>6}{'alpha_bar':>12}{'signal':>10}{'noise':>10}{'SNR dB':>9}")
    for row in stats["effective_snr"]:
        print(f"{row['t']:>6}{row['alpha_bar']:>12.6f}{row['signal_rms']:>10.4f}"
              f"{row['noise_rms']:>10.4f}{row['snr_db']:>9.2f}")
    print(f"\nwritten to {args.out}")
    return stats


if __name__ == "__main__":
    main()
