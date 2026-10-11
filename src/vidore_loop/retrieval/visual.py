"""Channel B, visual: a late-interaction model over each page image, scored by MaxSim.

A page image becomes one vector per patch and a question one vector per token; a
page's score is, for each question token, its best-matching patch, summed. Two
open models are wired in, each with its own index under
`data/cache/index/<key>/visual/<model>/`:

- `evie`: tencent/EVIE-4.5B (Apache-2.0), finance 72.3 on English queries. Vectors
  are cut to their first 128 dimensions and renormalised (EVIE is trained for this
  Matryoshka truncation). Built on Qwen3.5, whose linear-attention kernels exist only
  for CUDA, so on Apple silicon it runs ~9.6 s a page.
- `tomoro`: TomoroAI/tomoro-colqwen3-embed-4b (Apache-2.0), finance 69.7. Qwen3-VL
  with standard attention; 320-dimensional vectors, kept whole.

Builds are resumable: every `PART` pages are written as one part, and a rerun skips
parts already on disk, so a stopped run loses at most one part. Search uses the model
whose index covers the most pages (`active`). Pages are encoded from the viewer's
cached JPEGs (1,600 px on the long edge; originals are 1,700 × 2,200), which each
model's processor resizes again anyway.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from vidore_loop.config import settings
from vidore_loop.data import pages

PART = 50
MIN_FREE_PCT = 15  # stop cleanly (resumable) below this much free memory


class LowMemoryError(RuntimeError):
    """The machine is short of memory; the build stopped at a part boundary and can be rerun."""


def free_memory_pct() -> int | None:
    """macOS's own free-memory figure (`memory_pressure`); None where it is unavailable."""
    import re
    import subprocess

    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"free percentage: (\d+)%", out)
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class VisualModel:
    key: str
    model_id: str
    dim: int
    finance_en: float  # English-query NDCG@10 on finance_en (MTEB), for the record
    kwargs: dict[str, Any] = field(default_factory=dict)


MODELS: dict[str, VisualModel] = {
    "evie": VisualModel("evie", "tencent/EVIE-4.5B", 128, 72.3),
    "tomoro": VisualModel(
        "tomoro",
        "TomoroAI/tomoro-colqwen3-embed-4b",
        320,
        69.7,
        {"trust_remote_code": True, "attn_implementation": "sdpa"},
    ),
}


def index_dir(key: str, model_key: str) -> Path:
    return settings().hf_home / "index" / key / "visual" / model_key


def device() -> str:
    import torch

    return (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )


@cache
def model(model_key: str) -> Any:
    import torch
    from sentence_transformers import MultiVectorEncoder

    m = MODELS[model_key]
    kw = dict(m.kwargs)
    trust = kw.pop("trust_remote_code", False)
    return MultiVectorEncoder(
        m.model_id,
        device=device(),
        trust_remote_code=trust,
        model_kwargs={"torch_dtype": torch.bfloat16, **kw},
    )


def _cut(vectors: Any, dim: int) -> np.ndarray:
    """First `dim` dimensions, renormalised, as float16."""
    v = np.asarray(
        vectors.float().cpu().numpy() if hasattr(vectors, "cpu") else vectors, dtype=np.float32
    )[:, :dim]
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v.astype(np.float16)


def _lock(key: str) -> Path:
    return settings().hf_home / "index" / key / "visual" / "building.lock"


def building(key: str) -> bool:
    """A build is running for this dataset (its lock names a live process)."""
    import os

    lock = _lock(key)
    if not lock.exists():
        return False
    try:
        os.kill(int(lock.read_text().strip()), 0)
    except (ValueError, ProcessLookupError, PermissionError):
        return False
    return True


def build(
    key: str, model_key: str = "evie", limit: int | None = None, batch_size: int = 2
) -> dict[str, Any]:
    """Encode the pages (or the first `limit`) part by part, skipping parts already built."""
    import os

    lock = _lock(key)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    try:
        return _build(key, model_key, limit, batch_size)
    finally:
        lock.unlink(missing_ok=True)


def _build(key: str, model_key: str, limit: int | None, batch_size: int) -> dict[str, Any]:
    from PIL import Image

    m = MODELS[model_key]
    all_ids = sorted(pages.page_index(key))
    ids = all_ids[:limit] if limit else all_ids
    out = index_dir(key, model_key)
    out.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(ids), PART):
        part = out / f"part_{start:05d}.npz"
        chunk_ids = ids[start : start + PART]
        if part.exists() and len(np.load(part)["ids"]) == len(chunk_ids):
            continue
        free = free_memory_pct()
        if free is not None and free < MIN_FREE_PCT:
            raise LowMemoryError(
                f"{free}% memory free (< {MIN_FREE_PCT}%): stopped before part {start // PART}; "
                "rerun the same command to resume"
            )
        enc = model(model_key)
        vecs: list[np.ndarray] = []
        t0 = time.time()
        for i in range(0, len(chunk_ids), batch_size):
            images = []
            for cid in chunk_ids[i : i + batch_size]:
                path = pages.image_path(key, cid)
                assert path is not None, cid
                images.append(Image.open(path).convert("RGB"))
            vecs += [
                _cut(v, m.dim)
                for v in enc.encode_document(images, batch_size=batch_size, show_progress_bar=False)
            ]
        seconds = time.time() - t0
        np.savez(
            part,
            ids=np.asarray(chunk_ids, dtype=np.int64),
            lengths=np.asarray([len(v) for v in vecs], dtype=np.int32),
            vectors=np.concatenate(vecs),
        )
        _write_stats(key, model_key, len(chunk_ids), seconds)
    _load.cache_clear()
    return stats_or_empty(key, model_key)


def _write_stats(key: str, model_key: str, encoded: int, seconds: float) -> dict[str, Any]:
    out = index_dir(key, model_key)
    prev = stats(key, model_key) or {}
    ids, lengths = [], []
    for p in sorted(out.glob("part_*.npz")):
        z = np.load(p)
        ids += z["ids"].tolist()
        lengths += z["lengths"].tolist()
    total_s = prev.get("encode_seconds", 0.0) + seconds
    total_n = prev.get("encoded_pages", 0) + encoded
    m = MODELS[model_key]
    s = {
        "model": m.model_id,
        "model_key": model_key,
        "dim": m.dim,
        "pages": len(ids),
        "complete": len(ids) == len(pages.page_index(key)),
        "encoded_pages": total_n,
        "encode_seconds": round(total_s, 1),
        "seconds_per_page": round(total_s / total_n, 2) if total_n else None,
        "mean_vectors_per_page": round(float(np.mean(lengths)), 1) if lengths else None,
        "device": device(),
    }
    (out / "stats.json").write_text(json.dumps(s, indent=1))
    return s


def stats_or_empty(key: str, model_key: str) -> dict[str, Any]:
    return stats(key, model_key) or {}


def stats(key: str, model_key: str) -> dict[str, Any] | None:
    p = index_dir(key, model_key) / "stats.json"
    return json.loads(p.read_text()) if p.exists() else None


def active(key: str) -> str | None:
    """The model whose index covers the most pages (EVIE on a tie); None when there is none."""
    order = list(MODELS)  # on a tie the first model wins: evie, the chosen default
    best = [(s["pages"], -order.index(k), k) for k in MODELS if (s := stats(key, k)) and s["pages"]]
    return max(best)[2] if best else None


def available(key: str) -> bool:
    return active(key) is not None


@cache
def _load(key: str, model_key: str) -> tuple[Any, Any, list[int]]:
    """Every indexed page's vectors, padded into one (pages, max_len, dim) tensor on the device."""
    import torch

    ids: list[int] = []
    flat, lengths = [], []
    for p in sorted(index_dir(key, model_key).glob("part_*.npz")):
        z = np.load(p)
        ids += z["ids"].tolist()
        lengths += z["lengths"].tolist()
        flat.append(z["vectors"])
    vectors = np.concatenate(flat)
    dim = vectors.shape[1]
    n, longest = len(lengths), int(max(lengths))
    padded = np.zeros((n, longest, dim), dtype=np.float16)
    mask = np.zeros((n, longest), dtype=bool)
    start = 0
    for i, ln in enumerate(lengths):
        padded[i, :ln] = vectors[start : start + ln]
        mask[i, :ln] = True
        start += ln
    dev = device()
    return torch.from_numpy(padded).to(dev), torch.from_numpy(mask).to(dev), ids


def search(key: str, query: str, k: int = 100, chunk: int = 256) -> list[tuple[int, float]]:
    """Top k pages as (corpus_id, MaxSim score), best first, with the active model."""
    import torch

    model_key = active(key)
    assert model_key, f"no visual index for {key}"
    docs, mask, ids = _load(key, model_key)
    qv = model(model_key).encode_query([query], show_progress_bar=False)[0]
    q = torch.from_numpy(_cut(qv, MODELS[model_key].dim)).to(docs.device)
    scores = []
    with torch.inference_mode():
        for i in range(0, docs.shape[0], chunk):
            sim = torch.einsum("qd,ntd->nqt", q, docs[i : i + chunk]).float()
            sim = sim.masked_fill(~mask[i : i + chunk, None, :], float("-inf"))
            scores.append(sim.max(dim=2).values.sum(dim=1))
    all_scores = torch.cat(scores).cpu().numpy()
    top = np.argsort(-all_scores)[:k]
    return [(ids[int(i)], float(all_scores[i])) for i in top]
