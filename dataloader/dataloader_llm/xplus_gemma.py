import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_PROMPT_TEMPLATE = """Rewrite this sentence by replacing a few words with synonyms.
Keep meaning identical.
Only output rewritten sentence.

Sentence:
{sentence}
"""


def _clean_generation(text: str) -> str:
    """Extract a clean single-line rewritten sentence from model output."""
    t = text.strip()

    # If model echoes prompt, try to grab last non-empty line
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Common patterns: "Rewritten sentence: ..."
    m = re.search(r"(?i)rewritten sentence\s*:\s*(.*)$", t)
    if m:
        cand = m.group(1).strip()
        return cand.strip("\"' ").strip()

    # Otherwise use last line
    cand = lines[-1]
    cand = cand.strip("\"' ").strip()
    return cand


@dataclass
class GemmaXPlusGenerator:
    model_dir: str
    device: str = "cuda"
    dtype: str = "bfloat16"  # "float16" also ok
    max_new_tokens: int = 48
    temperature: float = 0.7
    top_p: float = 0.9

    def __post_init__(self):
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"model_dir not found: {self.model_dir}")

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(self.dtype, torch.bfloat16)

        # local_files_only=True ensures it never tries HF internet
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)

        # Use CausalLM (text-only); device_map="auto" works on A100
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            local_files_only=True,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=torch_dtype,
        )
        self.model.eval()

        # For some models tokenizers need pad token
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def generate_xplus(self, x: str, prompt_template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
        prompt = prompt_template.format(sentence=x)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move to model device
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # decoded likely contains the prompt + completion; clean it
        # Try to remove the prompt prefix if it exists
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):].strip()

        xplus = _clean_generation(decoded)

        # Safety fallback: if model returns empty or same sentence, lightly modify
        if not xplus:
            return x
        return xplus
