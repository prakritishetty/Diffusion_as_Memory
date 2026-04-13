import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    # .json assumed to be a list
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Expected a list in JSON file: {path}")
    return obj


class MSRGistDataset(Dataset):
    """
    Produces:
      - raw_x: str
      - raw_x_plus: str (generated on the fly if generator provided)
      - x_input_ids, x_attention_mask
      - xplus_input_ids, xplus_attention_mask
      - labels (token ids for y)
      - id (as string)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 128,
        y_key: str = "summary",
        include_xt: bool = False,
        xplus_generator=None,
        xplus_cache: Optional[Dict[str, str]] = None,
        prompt_template: Optional[str] = None,
    ):
        self.data_path = data_path
        self.rows = _load_json_or_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.y_key = y_key
        self.include_xt = include_xt

        self.xplus_generator = xplus_generator
        self.xplus_cache = xplus_cache if xplus_cache is not None else {}
        self.prompt_template = prompt_template

        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.rows)

    def _get_id(self, ex: Dict[str, Any]) -> str:
        # Your data had ids like "36_37" or numbers; normalize
        _id = ex.get("id", "")
        return str(_id)

    def _get_x(self, ex: Dict[str, Any]) -> str:
        return str(ex.get("x", ""))

    def _get_y(self, ex: Dict[str, Any]) -> str:
        # Some of your earlier scripts used "y"; others used "summary"
        val = ex.get(self.y_key, "")
        return str(val)

    def _get_xplus(self, x: str) -> str:
        # Always generate x+ on the fly (no cache)
        if self.xplus_generator is None:
            return x

        if self.prompt_template is None:
            return self.xplus_generator.generate_xplus(x)

        return self.xplus_generator.generate_xplus(x, prompt_template=self.prompt_template)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.rows[idx]
        ex_id = self._get_id(ex)

        x = self._get_x(ex)
        y = self._get_y(ex)

        xplus = self._get_xplus(x)

        # Tokenize x
        x_tok = self.tokenizer(
            x,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Tokenize x+
        xp_tok = self.tokenizer(
            xplus,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Tokenize y as labels (seq2seq-style labels)
        y_tok = self.tokenizer(
            y,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "id": ex_id,
            "raw_x": x,
            "raw_x_plus": xplus,
            "x_input_ids": x_tok["input_ids"].squeeze(0),
            "x_attention_mask": x_tok["attention_mask"].squeeze(0),
            "xplus_input_ids": xp_tok["input_ids"].squeeze(0),
            "xplus_attention_mask": xp_tok["attention_mask"].squeeze(0),
            "labels": y_tok["input_ids"].squeeze(0),
        }

        if self.include_xt:
            item["xt"] = ex.get("xt", None)

        return item
