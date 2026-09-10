"""
Generate x+ (meaning-preserving synonym rewrite) for every record in a JSON dataset
using locally-mounted Gemma on Unity (/datasets/...).

Input:  a JSON file that is a LIST of objects (your cleaned train_part2_clean.json)
Output: JSONL (one JSON object per line) with a new field: "x_plus"

Features:
- Resume-safe: skips ids already present in out_jsonl (valid JSONL required)
- Writes incrementally (so crashes don't lose everything)
- Deterministic option (greedy decode) or sampled decode
- Robust parsing: extracts rewritten sentence, avoids echoing prompt
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Any, Iterable, Set, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_TEMPLATE = """Rewrite the sentence by replacing a few words with synonyms.
Keep meaning identical.
Do NOT add new facts.
Output ONLY the rewritten sentence (no quotes, no explanation).

Sentence:
{sentence}
"""


def read_json_list(path: str) -> list[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON list at {path}, got {type(obj)}")
    return obj


def load_done_ids(out_jsonl: str, id_key: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(out_jsonl):
        return done

    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Output file {out_jsonl} is not valid JSONL.\n"
                    f"First bad line: {line_no}\n"
                    f"Line contents (first 200 chars): {line[:200]!r}\n"
                    f"Fix: delete/rename the file and re-run, or convert it to JSONL."
                ) from e
            if id_key in obj:
                done.add(str(obj[id_key]))
    return done


def clean_model_output(text: str) -> str:
    """
    Try to extract just the rewritten sentence.
    Handles common patterns like:
      "Rewritten sentence: ...."
      "Rewrite: ...."
      or prompt echo.
    """
    t = text.strip()

    # If model includes a label, strip it.
    t = re.sub(r"^\s*(Rewritten sentence|Rewrite|Output)\s*:\s*", "", t, flags=re.I).strip()

    # Sometimes it echoes the instruction. Keep last non-empty line.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Heuristic: pick the last line (often the actual answer).
    candidate = lines[-1]

    # Strip wrapping quotes if any
    candidate = candidate.strip().strip('"').strip("'").strip()

    return candidate


@torch.inference_mode()
def generate_xplus(
    model,
    tokenizer,
    sentence: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    prompt = PROMPT_TEMPLATE.format(sentence=sentence.strip())
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # remove None keys
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    out = model.generate(**inputs, **gen_kwargs)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    # Remove prompt prefix if echoed
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt) :]

    xplus = clean_model_output(decoded)

    # If it failed or returned identical, just return something safe (still ok; you can re-run later)
    if not xplus:
        return sentence.strip()

    return xplus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="Input JSON (list of objects)")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL file (append/resume)")
    ap.add_argument("--model_dir", required=True, help="Local snapshot directory (contains config.json)")
    ap.add_argument("--id_key", default="id", help="Record id key")
    ap.add_argument("--x_key", default="x", help="Sentence field key")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N new items")
    ap.add_argument("--skip_existing_xplus", action="store_true", help="Skip records that already have x_plus")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--do_sample", action="store_true", help="Use sampling (else greedy).")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    records = read_json_list(args.in_path)

    done_ids = load_done_ids(args.out_jsonl, args.id_key)
    print(f"Loaded {len(records)} records")
    print(f"Already done (from {args.out_jsonl}): {len(done_ids)}")

    # dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print("Loading tokenizer/model from:", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch_dtype,
    )
    model.eval()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    processed = 0
    new_written = 0

    with open(args.out_jsonl, "a", encoding="utf-8") as out_f:
        for idx, rec in enumerate(records):
            rec_id = str(rec.get(args.id_key, idx))

            if rec_id in done_ids:
                continue
            if args.skip_existing_xplus and "x_plus" in rec and isinstance(rec["x_plus"], str) and rec["x_plus"].strip():
                continue

            x = rec.get(args.x_key, "")
            if not isinstance(x, str) or not x.strip():
                # still write a record, but x_plus mirrors x
                x = "" if not isinstance(x, str) else x

            x_plus = generate_xplus(
                model,
                tokenizer,
                x,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            # Build output object: keep everything, add x_plus
            out_obj = dict(rec)
            out_obj["x_plus"] = x_plus

            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_f.flush()

            new_written += 1
            processed += 1
            if new_written % 10 == 0:
                print(f"Wrote {new_written} new records... (last id={rec_id})", flush=True)

            if args.limit and processed >= args.limit:
                break

    print(f"Done. Newly written: {new_written}")
    print(f"Output: {args.out_jsonl}")


if __name__ == "__main__":
    main()
