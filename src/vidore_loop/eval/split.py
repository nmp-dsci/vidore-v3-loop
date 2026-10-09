"""Our train / test split of a dataset's English questions (M3).

ViDoRe V3 ships one `test` split; we divide its English questions 50/50 so a
pipeline can be tuned on one half and reported once on the other. The split is
stratified by each question's full set of query types (finance has 18 such
combinations): questions are shuffled within each combination with a fixed seed,
then dealt alternately to train and test across the combinations in turn, so
every combination and the whole split are balanced to within one question.

Written once to `data/splits/<key>.json` and committed; a run records the file's
hash, so a run can never silently be scored on a different split.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from vidore_loop.config import ROOT
from vidore_loop.data import catalog

SEED = 300
SPLITS = ("train", "test")
SPLITS_DIR = ROOT / "data" / "splits"


def path(key: str) -> Path:
    return SPLITS_DIR / f"{key}.json"


def stratum(q: dict[str, Any]) -> str:
    return "+".join(sorted(q["query_types"])) or "untyped"


def make(questions: list[dict[str, Any]], seed: int = SEED) -> dict[str, list[int]]:
    """Deal the questions into train and test, stratified by query-type combination."""
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for q in questions:
        groups[stratum(q)].append(int(q["query_id"]))
    out: dict[str, list[int]] = {"train": [], "test": []}
    turn = 0  # carried across groups, so odd-sized groups even out overall
    for name in sorted(groups):
        ids = sorted(groups[name])
        rng.shuffle(ids)
        for qid in ids:
            out[SPLITS[turn % 2]].append(qid)
            turn += 1
    return {s: sorted(v) for s, v in out.items()}


def write(key: str, seed: int = SEED) -> dict[str, Any]:
    """Create the split file for a dataset; refuses to overwrite an existing one."""
    p = path(key)
    if p.exists():
        raise FileExistsError(f"{p} exists: a split is fixed once runs are scored on it")
    qs = catalog.queries(key)
    split = make(qs, seed)
    doc = {
        "dataset": key,
        "seed": seed,
        "stratified_by": "the sorted set of query types",
        "n": {s: len(v) for s, v in split.items()},
        **split,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def load(key: str) -> dict[str, Any]:
    p = path(key)
    if not p.exists():
        raise FileNotFoundError(f"no split for {key}: run `make split DATASET={key}`")
    doc: dict[str, Any] = json.loads(p.read_text())
    return doc


def ids(key: str, which: str) -> list[int]:
    """The query ids of one split; `all` is every English question."""
    if which == "all":
        return sorted(int(q["query_id"]) for q in catalog.queries(key))
    if which not in SPLITS:
        raise ValueError(f"split is train, test or all, not {which!r}")
    return [int(i) for i in load(key)[which]]


def fingerprint(key: str) -> str:
    """A short hash of the split file, recorded on every run."""
    return hashlib.sha256(path(key).read_bytes()).hexdigest()[:12]


def assignment(key: str) -> dict[int, str]:
    doc = load(key)
    return {int(i): s for s in SPLITS for i in doc[s]}
