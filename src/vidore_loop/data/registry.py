"""The eight public ViDoRe V3 datasets, each pinned to a Hub revision.

`revision` is the dataset's `main` on 2026-09-28, the bytes every run here reads.
`paper_revision` is the commit each dataset card names as the one used for the
paper's end-to-end evaluation (arXiv 2601.08620); keep it for a like-for-like
comparison. The two private datasets (nuclear, telecom) are scored only by the
MTEB leaderboard and are not reachable from here.

Every dataset has four configs, each with a single `test` split: `corpus`
(page image + OCR markdown), `queries` (six languages, reference `answer`),
`qrels` (relevance 1 = critical, 2 = full, with bounding boxes) and
`documents_metadata`. Nothing upstream is held out, so our train/test cut is
our own (`data/splits/`, not built yet).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Config = Literal["corpus", "queries", "qrels", "documents_metadata"]
CONFIGS: tuple[Config, ...] = ("corpus", "queries", "qrels", "documents_metadata")
# The configs a loop cycle needs without images; `corpus` is ~0.4–1.5 GB per dataset.
LIGHT_CONFIGS: tuple[Config, ...] = ("queries", "qrels", "documents_metadata")
LANGUAGES: tuple[str, ...] = ("english", "french", "spanish", "german", "italian", "portuguese")


@dataclass(frozen=True)
class Dataset:
    key: str
    domain: str
    doc_language: str
    documents: int
    pages: int
    queries: int  # originals, before the six-language translation
    revision: str
    paper_revision: str

    @property
    def repo_id(self) -> str:
        return f"vidore/vidore_v3_{self.key}"

    @property
    def queries_all_languages(self) -> int:
        return self.queries * len(LANGUAGES)


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        "finance_en", "Finance", "en", 6, 2942, 309,
        "7f432c176d82e27546501ad8064a713ac3071809",
        "0fe6508053e8aa31b1f1eaec4553f79e3974f59a",
    ),
    Dataset(
        "finance_fr", "Finance", "fr", 5, 2384, 320,
        "1d808daa08032ffecdf62da151a7f7a8fe2bd0c9",
        "d1178382ea47039073ad08e5e2c97ef51d6b629e",
    ),
    Dataset(
        "computer_science", "Computer Science", "en", 2, 1360, 215,
        "d5cc75883d92e294f0c0fc2662551c9708a06ebc",
        "7b91f10e18b72a763dd17a0c05d66bf985b98f1d",
    ),
    Dataset(
        "hr", "HR", "en", 14, 1110, 318,
        "0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3",
        "95f2f83a5a09590a89e34960479f9438e48bca77",
    ),
    Dataset(
        "energy", "Energy", "fr", 42, 2229, 308,
        "caec06d3c73434d635f710f93bcd898331c59f20",
        "0f6e77a3f73911e7c7834835391229d2a623e8c0",
    ),
    Dataset(
        "industrial", "Industrial", "en", 27, 5244, 283,
        "e26c864724f5dd71a3d7d739272d95637764cee9",
        "233d20721f4deb392a09a86cca01761adbc91157",
    ),
    Dataset(
        "pharmaceuticals", "Pharmaceuticals", "en", 52, 2313, 364,
        "3abd4aa8a9445fb5538a78a19ba50bd57bd22b5c",
        "262e203d7c59c55947f7042812e0bcc1eee190b1",
    ),
    Dataset(
        "physics", "Physics", "fr", 42, 1674, 302,
        "a0de276f515acc044b72cae8de53a44bb5a8f1f5",
        "c5a4712eeeaf5194c918466ebc20c137b6c82c35",
    ),
)  # fmt: skip

BY_KEY: dict[str, Dataset] = {d.key: d for d in DATASETS}


def get(key: str) -> Dataset:
    try:
        return BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown dataset {key!r}; one of: {', '.join(BY_KEY)}") from None
