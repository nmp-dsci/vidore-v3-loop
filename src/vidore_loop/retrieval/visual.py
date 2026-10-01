"""Channel B, visual: a late-interaction model over each page image, scored by MaxSim.

EVIE-4.5B (tencent, Apache-2.0) turns a page image into one vector per patch and a
question into one vector per token. The score of a page is, for each question token,
its best-matching patch, summed. Vectors are cut to their first `DIM` dimensions and
renormalised (EVIE is trained for this Matryoshka truncation) and stored as float16
in `data/cache/index/<key>/visual/`: one array of every patch vector, one of offsets.

Pages are encoded from the viewer's cached JPEGs (1,600 px on the long edge; the
originals are 1,700 × 2,200), which the model's processor resizes again anyway.
"""

from __future__ import annotations

import json
import time
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from vidore_loop.config import settings
from vidore_loop.data import pages

MODEL_ID = "tencent/EVIE-4.5B"
DIM = 128


def index_dir(key: str) -> Path:
    return settings().hf_home / "index" / key / "visual"


def device() -> str:
    import torch

    return (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )


@cache
def model() -> Any:
    import torch
    from sentence_transformers import MultiVectorEncoder

    return MultiVectorEncoder(
        MODEL_ID, device=device(), model_kwargs={"torch_dtype": torch.bfloat16}
    )


def _cut(vectors: Any) -> np.ndarray:
    """First DIM dimensions, renormalised, as float16."""
    v = np.asarray(
        vectors.float().cpu().numpy() if hasattr(vectors, "cpu") else vectors, dtype=np.float32
    )[:, :DIM]
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v.astype(np.float16)


def build(key: str, limit: int | None = None, batch_size: int = 2) -> dict[str, Any]:
    """Encode every page (or the first `limit`); returns pages, seconds and seconds per page."""
    from PIL import Image

    ids = sorted(pages.page_index(key))[:limit] if limit else sorted(pages.page_index(key))
    enc = model()
    chunks: list[np.ndarray] = []
    lengths: list[int] = []
    t0 = time.time()
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        images = []
        for cid in batch:
            path = pages.image_path(key, cid)
            assert path is not None, cid
            images.append(Image.open(path).convert("RGB"))
        for v in enc.encode_document(images, batch_size=batch_size, show_progress_bar=False):
            cut = _cut(v)
            chunks.append(cut)
            lengths.append(len(cut))
    seconds = time.time() - t0
    out = index_dir(key)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", np.concatenate(chunks))
    np.save(out / "lengths.npy", np.asarray(lengths, dtype=np.int32))
    stats = {
        "model": MODEL_ID,
        "dim": DIM,
        "pages": len(ids),
        "complete": len(ids) == len(pages.page_index(key)),
        "seconds": round(seconds, 1),
        "seconds_per_page": round(seconds / max(1, len(ids)), 2),
        "mean_vectors_per_page": round(float(np.mean(lengths)), 1),
        "device": device(),
    }
    (out / "ids.json").write_text(json.dumps(ids))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    _load.cache_clear()
    return stats


def available(key: str) -> bool:
    return (index_dir(key) / "ids.json").exists()


def stats(key: str) -> dict[str, Any] | None:
    p = index_dir(key) / "stats.json"
    return json.loads(p.read_text()) if p.exists() else None


@cache
def _load(key: str) -> tuple[Any, Any, list[int]]:
    """Every page's vectors, padded into one (pages, max_len, DIM) tensor on the device, and its mask."""
    import torch

    d = index_dir(key)
    flat = np.load(d / "vectors.npy")
    lengths = np.load(d / "lengths.npy")
    ids: list[int] = json.loads((d / "ids.json").read_text())
    n, longest = len(lengths), int(lengths.max())
    padded = np.zeros((n, longest, DIM), dtype=np.float16)
    mask = np.zeros((n, longest), dtype=bool)
    start = 0
    for i, ln in enumerate(lengths):
        padded[i, :ln] = flat[start : start + ln]
        mask[i, :ln] = True
        start += ln
    dev = device()
    return torch.from_numpy(padded).to(dev), torch.from_numpy(mask).to(dev), ids


def search(key: str, query: str, k: int = 100, chunk: int = 256) -> list[tuple[int, float]]:
    """Top k pages as (corpus_id, MaxSim score), best first."""
    import torch

    docs, mask, ids = _load(key)
    q = torch.from_numpy(_cut(model().encode_query([query], show_progress_bar=False)[0])).to(
        docs.device
    )
    scores = []
    with torch.inference_mode():
        for i in range(0, docs.shape[0], chunk):
            sim = torch.einsum("qd,ntd->nqt", q, docs[i : i + chunk]).float()
            sim = sim.masked_fill(~mask[i : i + chunk, None, :], float("-inf"))
            scores.append(sim.max(dim=2).values.sum(dim=1))
    all_scores = torch.cat(scores).cpu().numpy()
    top = np.argsort(-all_scores)[:k]
    return [(ids[int(i)], float(all_scores[i])) for i in top]
