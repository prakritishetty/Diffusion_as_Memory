"""MSR abstraction-ladder dataset.

Fixes carried over from the audit:

* **The ladder now starts at the source.** `levels = [x] + xt`, so rung 0 is the verbatim text.
  Previously `t // 100` meant even `t = 1` targeted `xt[0]`, an abstraction -- the model could
  never be asked to reproduce full detail, and the bottom of the ladder simply did not exist.
* **`max_seq_len` 64 -> 128.** At 64 tokens roughly a third of MSR examples were silently
  truncated (mean 36.8 words, p95 56, max 135), i.e. the detail we claim to preserve was being
  cut before the encoder saw it.
* **Contrastive positives are real.** The old loader set `xpos_input_ids = x_input_ids`, so the
  InfoNCE positive was the anchor itself. The default positive is now another rung of the *same*
  ladder, which is both free and exactly the property we need: the gist must stay fixed while
  detail fades.
* Tokenisation is computed once and cached rather than re-run every epoch.

Timestep and rung are *not* drawn here -- they come from `LevelMap` in the training loop, so a
single definition of the supervision map is shared by training, evaluation and inference.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from utils.cache import CacheStore, stable_hash


@dataclass
class Example:
    uid: str
    x: str
    y: str
    levels: list[str]          # [x, xt_0, ..., xt_{n-2}]
    x_plus: list[str]


def _read_records(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_examples(
    records: list[dict[str, Any]],
    include_source_as_level0: bool = True,
    max_levels: int = 13,
) -> list[Example]:
    out: list[Example] = []
    for i, r in enumerate(records):
        x = (r.get("x") or "").strip()
        if not x:
            continue
        xt = [s.strip() for s in (r.get("xt") or []) if s and s.strip()]
        levels = ([x] + xt) if include_source_as_level0 else (xt or [x])
        levels = levels[:max_levels]
        out.append(Example(
            uid=str(r.get("id", i)),
            x=x,
            y=(r.get("y") or "").strip() or (levels[-1] if levels else x),
            levels=levels,
            x_plus=[s.strip() for s in (r.get("x_plus") or []) if s and s.strip()],
        ))
    return out


class MemoryLadderDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        cfg,
        cache: CacheStore | None = None,
        split: str = "train",
    ):
        self.cfg = cfg
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = cfg.data.max_seq_len
        self.max_gist_len = cfg.data.max_gist_len
        self.max_levels = cfg.data.max_levels
        self.pad_id = tokenizer.pad_token_id

        self.examples = build_examples(
            _read_records(path), cfg.data.include_source_as_level0, cfg.data.max_levels
        )

        deps = [os.path.basename(path), len(self.examples), self.max_seq_len,
                self.max_gist_len, self.max_levels,
                getattr(tokenizer, "name_or_path", "tok"),
                stable_hash([e.x for e in self.examples[:64]])]
        self.enc = (
            cache.get_or_compute(f"tokenised_{split}", deps, self._tokenise_all)
            if cache else self._tokenise_all()
        )
        self.gist_embeddings: torch.Tensor | None = None

    # -- one-off tokenisation ---------------------------------------------------------------
    def _tok(self, texts: list[str], max_len: int):
        return self.tokenizer(
            texts, padding="max_length", truncation=True, max_length=max_len,
            return_tensors="pt",
        )

    def _tokenise_all(self) -> dict[str, torch.Tensor]:
        n, L, S = len(self.examples), self.max_levels, self.max_seq_len

        x = self._tok([e.x for e in self.examples], S)
        y = self._tok([e.y for e in self.examples], self.max_gist_len)

        lvl_ids = torch.full((n, L, S), self.pad_id, dtype=torch.long)
        lvl_mask = torch.zeros((n, L, S), dtype=torch.long)
        n_levels = torch.zeros(n, dtype=torch.long)
        for i, ex in enumerate(self.examples):
            k = min(len(ex.levels), L)
            n_levels[i] = k
            enc = self._tok(ex.levels[:k], S)
            lvl_ids[i, :k] = enc["input_ids"]
            lvl_mask[i, :k] = enc["attention_mask"]

        max_pos = max((len(e.x_plus) for e in self.examples), default=0)
        if max_pos:
            pos_ids = torch.full((n, max_pos, S), self.pad_id, dtype=torch.long)
            pos_mask = torch.zeros((n, max_pos, S), dtype=torch.long)
            n_pos = torch.zeros(n, dtype=torch.long)
            for i, ex in enumerate(self.examples):
                if ex.x_plus:
                    enc = self._tok(ex.x_plus, S)
                    n_pos[i] = len(ex.x_plus)
                    pos_ids[i, : len(ex.x_plus)] = enc["input_ids"]
                    pos_mask[i, : len(ex.x_plus)] = enc["attention_mask"]
        else:
            pos_ids = torch.zeros((n, 0, S), dtype=torch.long)
            pos_mask = torch.zeros((n, 0, S), dtype=torch.long)
            n_pos = torch.zeros(n, dtype=torch.long)

        return {
            "x_input_ids": x["input_ids"], "x_attention": x["attention_mask"],
            "y_input_ids": y["input_ids"], "y_attention": y["attention_mask"],
            "level_input_ids": lvl_ids, "level_attention": lvl_mask, "n_levels": n_levels,
            "xplus_input_ids": pos_ids, "xplus_attention": pos_mask, "n_xplus": n_pos,
        }

    def attach_gist_embeddings(self, emb: torch.Tensor) -> None:
        assert emb.shape[0] == len(self.examples), "gist embedding count mismatch"
        self.gist_embeddings = emb

    # -- helpers ------------------------------------------------------------------------------
    def _labels(self, ids: torch.Tensor) -> torch.Tensor:
        lab = ids.clone()
        lab[lab == self.pad_id] = -100
        return lab

    def _drop_tokens(self, ids: torch.Tensor, mask: torch.Tensor, p: float) -> torch.Tensor:
        valid = mask.bool() & (ids != self.pad_id)
        keep = valid & (torch.rand(ids.shape) >= p)
        if int(keep.sum()) <= 1:
            return ids
        kept = ids[keep]
        out = torch.full_like(ids, self.pad_id)
        out[: kept.numel()] = kept[: ids.numel()]
        return out

    def _positive(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src = self.cfg.data.xplus_source
        e = self.enc

        if src == "identity":  # reproduces the original defect, for the ablation
            return e["x_input_ids"][idx], e["x_attention"][idx]

        if src == "paraphrase" and int(e["n_xplus"][idx]) > 0:
            j = random.randrange(int(e["n_xplus"][idx]))
            return e["xplus_input_ids"][idx, j], e["xplus_attention"][idx, j]

        if src == "dropout":
            ids = self._drop_tokens(e["x_input_ids"][idx], e["x_attention"][idx],
                                    self.cfg.data.xplus_dropout_prob)
            return ids, (ids != self.pad_id).long()

        # Default: another rung of the same ladder. The gist must be invariant along it.
        n = int(e["n_levels"][idx])
        hi = min(n, self.cfg.data.ladder_positive_max_level + 1)
        j = random.randrange(hi) if hi > 1 else 0
        return e["level_input_ids"][idx, j], e["level_attention"][idx, j]

    # -- torch Dataset --------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        e = self.enc
        xpos_ids, xpos_mask = self._positive(idx)
        item = {
            "idx": torch.tensor(idx, dtype=torch.long),
            "uid": self.examples[idx].uid,
            "x_text": self.examples[idx].x,
            "y_text": self.examples[idx].y,
            "x_input_ids": e["x_input_ids"][idx],
            "x_attention": e["x_attention"][idx],
            "x_labels": self._labels(e["x_input_ids"][idx]),
            "y_labels": self._labels(e["y_input_ids"][idx]),
            "xpos_input_ids": xpos_ids,
            "xpos_attention": xpos_mask,
            "level_input_ids": e["level_input_ids"][idx],
            "level_attention": e["level_attention"][idx],
            "n_levels": e["n_levels"][idx],
        }
        if self.gist_embeddings is not None:
            item["gist_embedding"] = self.gist_embeddings[idx]
        return item

    def level_texts(self, idx: int) -> list[str]:
        return self.examples[idx].levels

    def all_level_counts(self) -> list[int]:
        return self.enc["n_levels"].tolist()


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        out[key] = torch.stack(vals) if torch.is_tensor(vals[0]) else vals
    return out


def labels_for_levels(
    level_input_ids: torch.Tensor,   # [B, max_levels, S]
    k: torch.Tensor,                 # [B]
    pad_id: int,
) -> torch.Tensor:
    """Gather the rung-k token ids for each example and convert them into decoder labels."""
    B = level_input_ids.shape[0]
    ids = level_input_ids[torch.arange(B, device=level_input_ids.device), k.clamp(min=0)]
    labels = ids.clone()
    labels[labels == pad_id] = -100
    return labels
