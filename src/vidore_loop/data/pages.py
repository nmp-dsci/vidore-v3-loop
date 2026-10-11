"""Page images and OCR text, fetched from the Hub by row group and kept under `data/cache/pages/`.

A corpus is 0.4–1.5 GB of PNGs, so the viewer never loads one whole. The page
index (corpus id → parquet file, row group, document, page number) is built once
from the id columns alone, a few hundred KB. A page is written on first view
together with every other page in its row group (~230 pages, ~100 MB, ~10 s),
resized to a JPEG the browser can show. `vidoreloop pages` prefetches a dataset.

Bounding boxes in the qrels are in the original page's pixels, so each page's
JSON keeps the original width and height beside the markdown.
"""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from typing import Any

from vidore_loop.config import settings
from vidore_loop.data.registry import get

MAX_EDGE = 1600
JPEG_QUALITY = 85
_locks: dict[tuple[str, str, int], threading.Lock] = {}
_locks_guard = threading.Lock()


def pages_dir(key: str) -> Path:
    d = settings().hf_home / "pages" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fs() -> Any:
    from huggingface_hub import HfFileSystem

    return HfFileSystem()


def _corpus_files(key: str) -> list[str]:
    ds = get(key)
    files: list[str] = sorted(_fs().glob(f"datasets/{ds.repo_id}@{ds.revision}/corpus/*.parquet"))
    return files


def page_index(key: str) -> dict[int, dict[str, Any]]:
    """corpus_id → {file, group, doc_id, page}; built from the id columns once, then read from disk."""
    path = pages_dir(key) / "index.json"
    if path.exists():
        raw = json.loads(path.read_text())
        return {int(k): v for k, v in raw.items()}
    import pyarrow.parquet as pq

    fs = _fs()
    out: dict[int, dict[str, Any]] = {}
    for f in _corpus_files(key):
        pf = pq.ParquetFile(fs.open(f))
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["corpus_id", "doc_id", "page_number_in_doc"])
            for cid, doc, page in zip(
                t.column("corpus_id").to_pylist(),
                t.column("doc_id").to_pylist(),
                t.column("page_number_in_doc").to_pylist(),
                strict=True,
            ):
                out[int(cid)] = {"file": f, "group": g, "doc_id": doc, "page": int(page)}
    path.write_text(json.dumps({str(k): v for k, v in sorted(out.items())}))
    return out


def page_meta(key: str, corpus_id: int) -> dict[str, Any] | None:
    """Document, page number, original size and OCR markdown; fetches the row group if needed."""
    path = pages_dir(key) / f"{corpus_id}.json"
    if not path.exists() and not ensure_page(key, corpus_id):
        return None
    meta: dict[str, Any] = json.loads(path.read_text())
    return meta


def image_path(key: str, corpus_id: int) -> Path | None:
    path = pages_dir(key) / f"{corpus_id}.jpg"
    if path.exists() or ensure_page(key, corpus_id):
        return path
    return None


def ensure_page(key: str, corpus_id: int) -> bool:
    """Write the page's row group to disk if it is not there; False for an unknown corpus id."""
    entry = page_index(key).get(corpus_id)
    if entry is None:
        return False
    lock_key = (key, entry["file"], entry["group"])
    with _locks_guard:
        lock = _locks.setdefault(lock_key, threading.Lock())
    with lock:
        if (pages_dir(key) / f"{corpus_id}.jpg").exists():
            return True
        write_group(key, entry["file"], entry["group"])
    return True


def write_group(key: str, file: str, group: int) -> int:
    """Decode one row group's pages to `<id>.jpg` + `<id>.json`; returns pages written."""
    import pyarrow.parquet as pq
    from PIL import Image

    out = pages_dir(key)
    rows = pq.ParquetFile(_fs().open(file)).read_row_group(group).to_pylist()
    for r in rows:
        cid = int(r["corpus_id"])
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        width, height = img.size
        img.thumbnail((MAX_EDGE, MAX_EDGE))
        img.save(out / f"{cid}.jpg", quality=JPEG_QUALITY)
        meta = {
            "corpus_id": cid,
            "doc_id": r["doc_id"],
            "page": int(r["page_number_in_doc"]),
            "width": width,
            "height": height,
            "markdown": r["markdown"] or "",
        }
        (out / f"{cid}.json").write_text(json.dumps(meta))
    return len(rows)


def prefetch(key: str) -> int:
    """Every page of a dataset, group by group; returns pages written (already-present groups skipped)."""
    groups = {(e["file"], e["group"]): cid for cid, e in page_index(key).items()}
    written = 0
    for (file, group), cid in sorted(groups.items()):
        if not (pages_dir(key) / f"{cid}.jpg").exists():
            written += write_group(key, file, group)
    return written
