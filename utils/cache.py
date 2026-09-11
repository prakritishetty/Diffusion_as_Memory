"""Content-addressed disk cache.

Everything expensive and deterministic goes through here: tokenisation, frozen gist embeddings,
`(u, v0)` latents after P0, rung latents for `L_proj`, DDIM trajectories, generated text,
spaCy parses and metric intermediates.

The failure mode that matters is a *stale* hit -- silently training on latents produced by a
different checkpoint. Keys therefore include the code version plus whatever the caller declares
as a dependency, and every hit is logged with its key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import torch

log = logging.getLogger(__name__)

_TENSOR_EXT = ".pt"
_JSON_EXT = ".json"
_PICKLE_EXT = ".pkl"
_NPY_EXT = ".npz"


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(p)
        elif isinstance(p, str):
            h.update(p.encode("utf-8"))
        elif isinstance(p, (int, float, bool)) or p is None:
            h.update(repr(p).encode())
        elif isinstance(p, torch.Tensor):
            h.update(str(tuple(p.shape)).encode())
            h.update(p.detach().cpu().numpy().tobytes())
        elif isinstance(p, np.ndarray):
            h.update(str(p.shape).encode())
            h.update(p.tobytes())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def file_hash(path: str, chunk: int = 1 << 20) -> str:
    """Hash of a file's bytes -- used to bind a cache entry to a specific checkpoint."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    def __str__(self) -> str:
        return f"hits={self.hits} misses={self.misses} writes={self.writes}"


class CacheStore:
    """Namespaced, content-addressed store rooted at `root`.

    >>> store = CacheStore("cache", code_version="v3")
    >>> latents = store.get_or_compute("p0_latents", ["train", ckpt_hash], compute_fn)
    """

    def __init__(self, root: str = "cache", code_version: str = "v3", enabled: bool = True):
        self.root = os.path.abspath(root)
        self.code_version = code_version
        self.enabled = enabled
        self.stats = CacheStats()
        self._lock = threading.Lock()
        if enabled:
            os.makedirs(self.root, exist_ok=True)

    # -- keys ---------------------------------------------------------------------------
    def key(self, namespace: str, deps: Iterable[Any]) -> str:
        return f"{namespace}__{stable_hash(self.code_version, namespace, list(deps))}"

    def _path(self, key: str, ext: str) -> str:
        ns = key.split("__", 1)[0]
        d = os.path.join(self.root, ns)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, key + ext)

    # -- primitive io -------------------------------------------------------------------
    def _load(self, key: str) -> Any:
        for ext, loader in (
            (_TENSOR_EXT, lambda p: torch.load(p, map_location="cpu", weights_only=False)),
            (_NPY_EXT, lambda p: dict(np.load(p, allow_pickle=False))),
            (_JSON_EXT, lambda p: json.load(open(p, encoding="utf-8"))),
            (_PICKLE_EXT, lambda p: pickle.load(open(p, "rb"))),
        ):
            path = self._path(key, ext)
            if os.path.exists(path):
                return loader(path)
        raise KeyError(key)

    def _store(self, key: str, value: Any) -> None:
        if isinstance(value, (torch.Tensor, dict)) and _all_tensors(value):
            path = self._path(key, _TENSOR_EXT)
            torch.save(value, path + ".tmp")
        elif isinstance(value, (list, dict, str, int, float, bool)) or value is None:
            path = self._path(key, _JSON_EXT)
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(value, f)
        else:
            path = self._path(key, _PICKLE_EXT)
            with open(path + ".tmp", "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(path + ".tmp", path)  # atomic: a killed job never leaves a half-written entry

    # -- public api ---------------------------------------------------------------------
    def has(self, namespace: str, deps: Iterable[Any]) -> bool:
        if not self.enabled:
            return False
        key = self.key(namespace, deps)
        return any(os.path.exists(self._path(key, e))
                   for e in (_TENSOR_EXT, _NPY_EXT, _JSON_EXT, _PICKLE_EXT))

    def get_or_compute(
        self,
        namespace: str,
        deps: Iterable[Any],
        compute: Callable[[], Any],
        *,
        force: bool = False,
    ) -> Any:
        if not self.enabled:
            return compute()
        key = self.key(namespace, deps)
        if not force:
            try:
                value = self._load(key)
                with self._lock:
                    self.stats.hits += 1
                log.info("cache HIT  %s", key)
                return value
            except (KeyError, EOFError, RuntimeError):
                pass
        with self._lock:
            self.stats.misses += 1
        log.info("cache MISS %s -- computing", key)
        value = compute()
        self._store(key, value)
        with self._lock:
            self.stats.writes += 1
        return value

    def purge(self, namespace: str | None = None) -> int:
        target = self.root if namespace is None else os.path.join(self.root, namespace)
        if not os.path.exists(target):
            return 0
        n = sum(len(files) for _, _, files in os.walk(target))
        shutil.rmtree(target)
        log.warning("purged %d cache files from %s", n, target)
        return n

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"root": self.root, "enabled": self.enabled,
                               "stats": str(self.stats), "namespaces": {}}
        if not os.path.isdir(self.root):
            return out
        for ns in sorted(os.listdir(self.root)):
            d = os.path.join(self.root, ns)
            if not os.path.isdir(d):
                continue
            files = [os.path.join(d, f) for f in os.listdir(d)]
            out["namespaces"][ns] = {
                "entries": len(files),
                "mb": round(sum(os.path.getsize(f) for f in files) / 1e6, 1),
            }
        return out


def _all_tensors(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return len(value) > 0 and all(
            isinstance(v, (torch.Tensor, int, float, str, list)) for v in value.values()
        )
    return False


_GLOBAL: CacheStore | None = None


def get_cache(cfg=None) -> CacheStore:
    """Process-wide store, configured once from `ExperimentConfig.cache`."""
    global _GLOBAL
    if _GLOBAL is None:
        if cfg is None:
            _GLOBAL = CacheStore()
        else:
            _GLOBAL = CacheStore(cfg.cache.root, cfg.cache.code_version, cfg.cache.enabled)
    return _GLOBAL


def reset_cache() -> None:
    global _GLOBAL
    _GLOBAL = None
