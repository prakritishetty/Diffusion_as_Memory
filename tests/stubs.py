"""Offline stand-ins so model/pipeline tests need no network and no tokenizer downloads."""

from __future__ import annotations

import json
import os
import tempfile

import torch


class StubTokenizer:
    """Whitespace tokeniser with a hashed vocab; mimics the slice of the HF API we use."""

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.name_or_path = "stub"

    def _id(self, tok: str) -> int:
        return 3 + (abs(hash(tok)) % (self.vocab_size - 4))

    def __call__(self, texts, padding="max_length", truncation=True, max_length=32,
                 return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        ids, masks = [], []
        for t in texts:
            seq = [self._id(w) for w in t.split()][: max_length - 1] + [self.eos_token_id]
            mask = [1] * len(seq)
            pad = max_length - len(seq)
            ids.append(seq + [self.pad_token_id] * pad)
            masks.append(mask + [0] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

    def batch_decode(self, ids, skip_special_tokens=True):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return [" ".join(str(i) for i in row if i > 2) for row in ids]

    def decode(self, ids, skip_special_tokens=True):
        return self.batch_decode([ids], skip_special_tokens)[0]


SAMPLE_RECORDS = [
    {
        "id": "1",
        "x": "Washington has never demanded good governance and democracy from Cairo in return.",
        "xt": [
            "Washington has never demanded governance and democracy from Cairo.",
            "Washington has never demanded governance from Cairo.",
            "Washington has never demanded anything from Cairo.",
            "Washington has never demanded anything.",
        ],
        "y": "Demands for governance are not reciprocated.",
    },
    {
        "id": "2",
        "x": "The NMFS estimates that in 2000 more than 16,000 tons of seabass were caught.",
        "xt": [
            "The NMFS estimates that in 2000 a large amount of seabass was caught.",
            "The NMFS estimates that seabass was caught.",
            "Fish were caught.",
        ],
        "y": "Quantities of a fish species were harvested.",
    },
    {
        "id": "3",
        "x": "Join us for the Sale of the Century at JCC Auction on Saturday November 13 1999.",
        "xt": [
            "Join us for the Sale of the Century at JCC Auction in November.",
            "Join us for the Sale of the Century.",
            "Join us for a sale.",
            "Join us.",
        ],
        "y": "A sale event.",
    },
    {
        "id": "4",
        "x": "Exposition Park near downtown Los Angeles is the site of several museums.",
        "xt": [
            "Exposition Park near downtown has several museums.",
            "The park has museums.",
            "There is a park.",
        ],
        "y": "A location contains museums.",
    },
]


def write_sample_split(dirpath: str, n_repeat: int = 4) -> str:
    """Write a small train/validate/test trio and return the directory."""
    os.makedirs(dirpath, exist_ok=True)
    records = []
    for r in SAMPLE_RECORDS * n_repeat:
        records.append({**r, "id": f"{r['id']}_{len(records)}"})
    for name in ("train.json", "validate.json", "test.json"):
        with open(os.path.join(dirpath, name), "w", encoding="utf-8") as f:
            json.dump(records, f)
    return dirpath


def temp_data_dir(n_repeat: int = 4) -> str:
    return write_sample_split(tempfile.mkdtemp(prefix="dam_test_"), n_repeat)
