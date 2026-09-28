"""Load a ViDoRe V3 config at its pinned revision into the local cache (`data/cache/`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vidore_loop.config import settings
from vidore_loop.data.registry import LIGHT_CONFIGS, Config, get

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset


def load(key: str, config: Config) -> HFDataset:
    """One config of one dataset, as a `datasets.Dataset` (the upstream `test` split)."""
    from datasets import load_dataset

    ds = get(key)
    loaded: HFDataset = load_dataset(
        ds.repo_id,
        config,
        split="test",
        revision=ds.revision,
        cache_dir=str(settings().hf_home),
    )
    return loaded


def download(key: str, *, corpus: bool = False) -> dict[str, int]:
    """Fetch a dataset's configs into the cache; returns rows per config.

    Without `corpus` only the light configs come down (a few MB); the corpus
    carries every page image and is 0.4–1.5 GB per dataset.
    """
    configs: tuple[Config, ...] = (*LIGHT_CONFIGS, "corpus") if corpus else LIGHT_CONFIGS
    return {c: len(load(key, c)) for c in configs}


def qrels_by_query(rows: Any) -> dict[int, dict[int, int]]:
    """`qrels` rows → {query_id: {corpus_id: grade}} for the metrics."""
    out: dict[int, dict[int, int]] = {}
    for r in rows:
        out.setdefault(int(r["query_id"]), {})[int(r["corpus_id"])] = int(r["score"])
    return out
