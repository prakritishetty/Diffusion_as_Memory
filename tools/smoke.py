"""Full-pipeline smoke test through the real CLI entry points.

`tests/` exercises the modules; this exercises the *scripts* -- argument parsing, checkpoint
hand-off between phases, config compatibility checks, caching, and the evaluation runner. It is
the last thing to run before committing GPU time: if this passes, "press go" is a scheduling
problem rather than a debugging one.

Runs on CPU in about a minute with a ~1M-parameter model and no network access.

    python -m tools.smoke
    python -m tools.smoke --keep      # leave the artefacts for inspection
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback


def _banner(msg: str) -> None:
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="End-to-end pipeline smoke test")
    ap.add_argument("--keep", action="store_true", help="do not delete the temp workspace")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args(argv)

    from tests.stubs import write_sample_split

    work = args.workdir or tempfile.mkdtemp(prefix="dam_smoke_")
    data_dir = write_sample_split(os.path.join(work, "data"), n_repeat=8)
    runs = os.path.join(work, "runs")
    cache = os.path.join(work, "cache")
    print(f"workspace: {work}")

    steps: list[tuple[str, float, bool]] = []
    ok = True
    t_all = time.time()

    try:
        from configs import get as get_preset
        from training import train_p0, train_p1, train_p2

        base_override = {
            "data.data_dir": data_dir, "out_dir": runs, "cache.root": cache,
            "cache.enabled": True,
        }

        # -- P0 ------------------------------------------------------------------------------
        _banner("P0  autoencoder (encoder, slots, u/v heads, decoders)")
        t = time.time()
        cfg = get_preset("tiny").override(**base_override, **{"train.epochs": 2})
        cfg_path = os.path.join(work, "tiny.json")
        cfg.save(cfg_path)
        p0_path = train_p0.main([
            "--preset", "tiny", "--name", "smoke", "--data-dir", data_dir, "--epochs", "2",
        ])
        # train_p0 writes under cfg.out_dir, which the preset fixes to "runs"; relocate check
        assert os.path.exists(p0_path), f"missing P0 checkpoint at {p0_path}"
        steps.append(("P0", time.time() - t, True))

        # -- P1 ------------------------------------------------------------------------------
        _banner("P1  denoiser on cached latents (x0-pred, gist-anchored, CFG, EMA)")
        t = time.time()
        p1_path = train_p1.main(["--p0-checkpoint", p0_path, "--epochs", "2"])
        assert os.path.exists(p1_path), f"missing P1 checkpoint at {p1_path}"
        steps.append(("P1", time.time() - t, True))

        # -- P2 ------------------------------------------------------------------------------
        _banner("P2  G_psi + decoder (L_proj + L_CE, DDIM chain matched to inference)")
        t = time.time()
        p2_path = train_p2.main([
            "--p0-checkpoint", p0_path, "--p1-checkpoint", p1_path, "--epochs", "2",
        ])
        assert os.path.exists(p2_path), f"missing P2 checkpoint at {p2_path}"
        steps.append(("P2", time.time() - t, True))

        # -- evaluation -------------------------------------------------------------------------
        _banner("EVAL  free-running ladders + countable metrics")
        t = time.time()
        from evaluation import evaluate
        res = evaluate.main([
            "--p0", p0_path, "--p1", p1_path, "--p2", p2_path,
            "--data-dir", data_dir, "--split", "test", "--limit", "8",
            "--n-levels", "4", "--batch-size", "4",
            "--out", os.path.join(runs, "smoke_eval.json"),
        ])
        assert "frac_strictly_monotonic" in res
        steps.append(("EVAL", time.time() - t, True))

        # -- ablation gate ------------------------------------------------------------------------
        _banner("ABLATION  shuffled-cue gate + oracle/noise brackets")
        t = time.time()
        from ablations import runner
        runner.main([
            "--p0", p0_path, "--p1", p1_path, "--p2", p2_path,
            "--data-dir", data_dir, "--split", "test", "--limit", "8",
            "--n-levels", "4", "--batch-size", "4",
            "--only", "baseline", "shuffled_cue", "oracle_latent", "noise_only",
            "--out", os.path.join(runs, "smoke_ablation.json"),
        ])
        steps.append(("ABLATION", time.time() - t, True))

        # -- data-only tools -------------------------------------------------------------------------
        _banner("DATA  ceiling + truncation baseline (no GPU, no model)")
        t = time.time()
        from baselines import truncation
        from evaluation import ceiling
        ceiling.main(["--data-dir", data_dir, "--splits", "test",
                      "--out", os.path.join(runs, "smoke_ceiling.json")])
        truncation.main(["--data-dir", data_dir, "--split", "test",
                         "--out", os.path.join(runs, "smoke_trunc.json")])
        steps.append(("DATA", time.time() - t, True))

    except Exception:
        ok = False
        traceback.print_exc()
        steps.append(("FAILED", time.time() - t_all, False))

    _banner("SMOKE SUMMARY")
    for name, secs, good in steps:
        print(f"  {'PASS' if good else 'FAIL'}  {name:<10} {secs:6.1f}s")
    print(f"\n  total {time.time() - t_all:.1f}s")

    if ok and not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    elif args.keep:
        print(f"\n  artefacts kept in {work}")

    if ok:
        print("\nPipeline is wired end to end. Remaining unknowns are scientific, not mechanical.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
