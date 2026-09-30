"""The boards, from committed files: MTEB per-model scores and the pipeline / paper tables.

`docs/research/mteb_vidore3_scores_2026-09-28.json` holds every model's ViDoRe(v3)
result from embeddings-benchmark/results; each task value is `[mean over six query
languages, English-query score, languages, results commit, mteb version]`. The
viewer shows the English-query score, which matches the pipeline board's numbers
(nemotron-colembed-vl-8b-v2 is 69.1 on both).
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

from vidore_loop.config import ROOT

RESEARCH = ROOT / "docs" / "research"
MTEB_FILE = RESEARCH / "mteb_vidore3_scores_2026-09-28.json"
BOARDS_FILE = RESEARCH / "boards.json"

# our dataset key → MTEB task stem (`Vidore3<stem>Retrieval`)
MTEB_TASK = {
    "finance_en": "FinanceEn",
    "pharmaceuticals": "Pharmaceuticals",
    "industrial": "Industrial",
    "hr": "Hr",
    "computer_science": "ComputerScience",
}
ENGLISH = tuple(MTEB_TASK)


@cache
def mteb_models() -> list[dict[str, Any]]:
    """One row per model (its best revision) with all five English sets, best mean first."""
    raw: dict[str, dict[str, list[Any]]] = json.loads(MTEB_FILE.read_text())
    best: dict[str, dict[str, Any]] = {}
    for key, tasks in raw.items():
        model, revision, bench = key.split("|")
        if bench != "v3" or not all(MTEB_TASK[k] in tasks for k in ENGLISH):
            continue
        scores = [round(tasks[MTEB_TASK[k]][1] * 100, 1) for k in ENGLISH]
        row = {
            "model": model,
            "revision": revision[:7],
            "scores": scores,
            "mean": round(sum(scores) / len(scores), 1),
        }
        if model not in best or row["mean"] > best[model]["mean"]:
            best[model] = row
    return sorted(best.values(), key=lambda r: -float(r["mean"]))


@cache
def boards() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(BOARDS_FILE.read_text())
    return data


def leaderboard() -> dict[str, Any]:
    b = boards()
    return {
        "datasets": list(ENGLISH),
        "models": mteb_models(),
        "pipelines": b["pipelines"],
        "answers": b["answers"],
        "grounding": b["grounding"],
        "about": b["_about"],
    }
