"""Append-only record of every probe and every scored evaluation.

Training curves go to wandb. These do not: a probe produces one row of numbers with no time
axis, and what you want from it is a diff against the last run, not a plot. Keeping them in a
single JSONL means `git diff runs/ledger.jsonl` shows exactly what moved between two runs, and
`report()` renders the same rows as the table that goes in the paper.

    from utils import ledger
    ledger.append("D", "ladder_integrity", {"source_at_rung0_rate": 1.0}, verdict="PASS")
    print(ledger.report())
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

DEFAULT_PATH = "runs/ledger.jsonl"

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def verdict_of(*checks: bool, warn: bool = False) -> str:
    """PASS when every check holds, else WARN or FAIL."""
    if all(checks):
        return PASS
    return WARN if warn else FAIL


def append(
    phase: str,
    probe: str,
    metrics: dict[str, Any],
    *,
    config_hash: str | None = None,
    verdict: str | None = None,
    notes: str | None = None,
    path: str = DEFAULT_PATH,
) -> dict[str, Any]:
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase": phase,
        "probe": probe,
        "config_hash": config_hash,
        "verdict": verdict,
        "notes": notes,
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def read(
    path: str = DEFAULT_PATH,
    phase: str | None = None,
    probe: str | None = None,
) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # a half-written line never invalidates the rest
                continue
            if phase and row.get("phase") != phase:
                continue
            if probe and row.get("probe") != probe:
                continue
            rows.append(row)
    return rows


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1e4 else f"{v:.3e}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v[:8]) + ("…]" if len(v) > 8 else "]")
    if isinstance(v, dict):
        return json.dumps(v, default=str)[:80]
    return str(v)


def report(
    path: str = DEFAULT_PATH,
    phase: str | None = None,
    latest_only: bool = True,
) -> str:
    rows = read(path, phase=phase)
    if not rows:
        return f"(ledger empty: {path})"
    if latest_only:
        seen: dict[tuple[str, str], dict] = {}
        for r in rows:
            seen[(r.get("phase"), r.get("probe"))] = r
        rows = list(seen.values())

    out: list[str] = []
    for r in rows:
        head = f"### [{r.get('phase')}] {r.get('probe')}"
        bits = [b for b in (r.get("verdict"), r.get("ts"), r.get("config_hash")) if b]
        out.append(f"{head}  —  {'  ·  '.join(bits)}")
        if r.get("notes"):
            out.append(f"_{r['notes']}_")
        out.append("")
        out.append("| metric | value |")
        out.append("|---|---|")
        for k, v in (r.get("metrics") or {}).items():
            out.append(f"| {k} | {_fmt(v)} |")
        out.append("")
    return "\n".join(out)


def print_table(metrics: dict[str, Any], title: str = "", width: int = 34) -> None:
    """Console rendering used by the probe scripts."""
    if title:
        print(f"\n{title}")
        print("-" * (width + 24))
    for k, v in metrics.items():
        print(f"  {k:<{width}} {_fmt(v)}")


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Render the probe/eval ledger")
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--phase", default=None)
    ap.add_argument("--all", action="store_true", help="every row, not just the latest per probe")
    args = ap.parse_args(argv)
    print(report(args.path, args.phase, latest_only=not args.all))


if __name__ == "__main__":
    main()
