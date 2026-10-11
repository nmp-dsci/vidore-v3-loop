"""Our train / test / holdout split of a dataset's English questions.

ViDoRe V3 ships one `test` split; we draw a random sample of 100 questions to tune
on (train), another 100 to report on once (test), and leave the rest as a holdout
that no run may score. The sample is stratified by each question's full set of query
types (finance has 18 such combinations): every question gets the key
(position in its shuffled combination + one random offset per combination) / its size,
and the questions are taken in key order, so each combination is sampled in proportion
to its size. The first 200 in that order are the sample, dealt alternately to train
and test within each combination, so the two match in mix to within one question.

`extend` then deals the holdout into train and test, so the split grows to 50/50
without moving a question already scored: within each combination a holdout question
goes to whichever side has fewer, so train and test stay matched in mix.

Written to `data/splits/<key>.json` and committed; a run records the file's hash, so a
run can never silently be scored on a different split. A retired split moves to
`data/splits/archive/` with the runs scored on it (their answers stay reusable).
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
VERSION = 2
SPLITS = ("train", "test")  # the splits a run may score
HOLDOUT = "holdout"  # the rest: never scored
SIZE = {"train": 100, "test": 100}
SPLITS_DIR = ROOT / "data" / "splits"


def path(key: str) -> Path:
    return SPLITS_DIR / f"{key}.json"


def stratum(q: dict[str, Any]) -> str:
    return "+".join(sorted(q["query_types"])) or "untyped"


def make(
    questions: list[dict[str, Any]], seed: int = SEED, size: dict[str, int] | None = None
) -> dict[str, list[int]]:
    """A stratified random sample for train and test; every other question is holdout."""
    size = size or SIZE
    if sum(size.values()) > len(questions):
        raise ValueError(f"{sum(size.values())} questions asked for, {len(questions)} exist")
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for q in questions:
        groups[stratum(q)].append(int(q["query_id"]))
    keyed: list[tuple[float, float, int]] = []
    for name in sorted(groups):
        ids = sorted(groups[name])
        rng.shuffle(ids)
        offset = rng.random()
        keyed += [((i + offset) / len(ids), rng.random(), qid) for i, qid in enumerate(ids)]
    n = sum(size[s] for s in SPLITS)
    sample = {qid for *_, qid in sorted(keyed)[:n]}
    out: dict[str, list[int]] = {s: [] for s in (*SPLITS, HOLDOUT)}
    turn = 0  # carried across groups, so the two samples even out overall
    for name in sorted(groups):
        for qid in sorted(groups[name], key=lambda q: rng.random()):
            if qid not in sample:
                out[HOLDOUT].append(qid)
                continue
            open_ = [s for s in SPLITS if len(out[s]) < size[s]]
            which = SPLITS[turn % 2] if len(open_) == 2 else open_[0]
            out[which].append(qid)
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
        "version": VERSION,
        "seed": seed,
        "stratified_by": "the sorted set of query types",
        "sampling": "a stratified random sample of 100 train and 100 test; the rest is holdout",
        "n": {s: len(v) for s, v in split.items()},
        **split,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def grow(
    questions: list[dict[str, Any]], prev: dict[str, Any], seed: int = SEED
) -> dict[str, list[int]]:
    """Deal every holdout question into train or test; train and test keep what they had."""
    rng = random.Random(seed)
    by_id = {int(q["query_id"]): stratum(q) for q in questions}
    out = {s: [int(i) for i in prev[s]] for s in SPLITS}
    held: dict[str, list[int]] = defaultdict(list)
    for i in prev.get(HOLDOUT, []):
        held[by_id[int(i)]].append(int(i))
    turn = 0
    for name in sorted(held):
        ids = sorted(held[name])
        rng.shuffle(ids)
        for qid in ids:
            here = {s: sum(1 for i in out[s] if by_id[i] == name) for s in SPLITS}
            which = min(SPLITS, key=lambda s: (here[s], len(out[s]), (SPLITS.index(s) + turn) % 2))
            out[which].append(qid)
            turn += 1
    return {**{s: sorted(v) for s, v in out.items()}, HOLDOUT: []}


def extend(key: str, seed: int = SEED) -> dict[str, Any]:
    """Grow the split to every question; the current file moves to the archive first."""
    prev = load(key)
    if not prev.get(HOLDOUT):
        raise ValueError(f"{key}'s split has no holdout left to deal")
    old = prev.get("version", 1)
    archived = SPLITS_DIR / "archive" / f"{key}.v{old}.json"
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_bytes(path(key).read_bytes())
    split = grow(catalog.queries(key), prev, seed)
    doc = {
        "dataset": key,
        "version": old + 1,
        "seed": seed,
        "stratified_by": "the sorted set of query types",
        "sampling": (
            f"v{old}'s train and test kept, its holdout dealt to the side with fewer in each"
            " group: 50/50, no holdout"
        ),
        "extends": f"archive/{archived.name}",
        "n": {s: len(v) for s, v in split.items()},
        **split,
    }
    path(key).write_text(json.dumps(doc, indent=1) + "\n")
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
    if which == HOLDOUT:
        raise ValueError("the holdout is never scored")
    if which not in SPLITS:
        raise ValueError(f"split is train, test or all, not {which!r}")
    return [int(i) for i in load(key)[which]]


def fingerprint(key: str) -> str:
    """A short hash of the split file, recorded on every run."""
    return hashlib.sha256(path(key).read_bytes()).hexdigest()[:12]


def assignment(key: str) -> dict[int, str]:
    """Query id → train, test or holdout."""
    doc = load(key)
    return {int(i): s for s in (*SPLITS, HOLDOUT) for i in doc.get(s, [])}


def current(key: str, fingerprint_: str | None) -> bool:
    """Whether a run's recorded split hash is today's split file."""
    try:
        return fingerprint_ == fingerprint(key)
    except FileNotFoundError:
        return False
