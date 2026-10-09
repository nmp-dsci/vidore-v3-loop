"""M3 offline: the split, weighted fusion, a scored run's folder, its guards, and its API."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from vidore_loop.data import catalog, pages
from vidore_loop.eval import runs, split
from vidore_loop.retrieval import fuse, pipeline
from vidore_loop.serving import app as app_module

TYPES = [["extractive"], ["extractive", "numerical"], ["boolean"], ["multi-hop"]]


def _questions(n: int = 20) -> list[dict[str, Any]]:
    return [
        {
            "query_id": i,
            "query": f"question {i}",
            "query_types": TYPES[i % len(TYPES)],
            "content_type": ["Table", "Text"] if i % 2 else ["Chart"],
            "query_format": "question" if i % 3 else "keyword",
            "query_generator": "human" if i % 2 else "sdg",
            # page 1000+i is the full answer; odd questions also have a part page
            "pages": [{"corpus_id": 1000 + i, "score": 2}]
            + ([{"corpus_id": 2000 + i, "score": 1}] if i % 2 else []),
        }
        for i in range(n)
    ]


# ---- split ----------------------------------------------------------------------


def test_split_is_deterministic_balanced_and_disjoint() -> None:
    qs = _questions(21)
    a, b = split.make(qs), split.make(qs)
    assert a == b
    assert not set(a["train"]) & set(a["test"])
    assert sorted(a["train"] + a["test"]) == list(range(21))
    assert (len(a["train"]), len(a["test"])) == (11, 10)
    by_id = {q["query_id"]: split.stratum(q) for q in qs}
    tr, te = Counter(by_id[i] for i in a["train"]), Counter(by_id[i] for i in a["test"])
    assert all(abs(tr[s] - te[s]) <= 1 for s in set(by_id.values()))
    assert split.make(qs, seed=1) != a  # the seed matters


def test_split_file_is_written_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(split, "SPLITS_DIR", tmp_path)
    monkeypatch.setattr(catalog, "queries", lambda key: _questions(8))
    doc = split.write("finance_en")
    assert doc["n"] == {"train": 4, "test": 4} and doc["seed"] == split.SEED
    assert split.ids("finance_en", "test") == doc["test"]
    with pytest.raises(FileExistsError):
        split.write("finance_en")


# ---- fusion weights -------------------------------------------------------------


def test_weighted_rrf() -> None:
    text_, visual = [1, 2, 3], [3, 4, 1]
    assert [c for c, _ in fuse.rrf([text_, visual], k=4, weights=[0, 1])] == visual
    heavy = [c for c, _ in fuse.rrf([text_, visual], k=4, weights=[0.1, 1])]
    assert heavy[0] == 3  # visual's #1, also in text, beats text's #1
    assert fuse.rrf([text_, visual], k=4) == fuse.rrf([text_, visual], k=4, weights=[1, 1])
    with pytest.raises(ValueError):
        fuse.rrf([text_, visual], weights=[1])


# ---- a run, end to end, with a fake pipeline ------------------------------------


@pytest.fixture()
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    qs = _questions(20)
    by_id = {q["query_id"]: q for q in qs}
    monkeypatch.setattr(split, "SPLITS_DIR", tmp_path / "splits")
    monkeypatch.setattr(catalog, "queries", lambda key: qs)
    monkeypatch.setattr(catalog, "question", lambda key, qid: by_id.get(qid))
    monkeypatch.setenv("VIDORE_RUNS_DIR", str(tmp_path / "runs"))
    split.write("finance_en")
    monkeypatch.setattr(pipeline, "status", lambda key: {})
    monkeypatch.setattr(runs.RunConfig, "models", lambda self: {"visual": "fake", "reranker": None})

    def compute(key: str, query: str, stages: Any, depth: int, weights: Any, st: Any) -> Any:
        i = int(query.split()[-1])
        full = 1000 + i
        # text finds the answer at #3 for even questions and misses it for odd ones;
        # visual always ranks it #1
        text_ = [7, 8, full, 9] if i % 2 == 0 else [7, 8, 9]
        visual = [full, 7, 8]
        lists = {
            "text": [(c, 1.0) for c in text_],
            "visual": [(c, 1.0) for c in visual],
            "fused": fuse.rrf([text_, visual], k=50, weights=list(weights)),
        }
        return lists, {s: {"ms": 1} for s in lists}

    monkeypatch.setattr(pipeline, "compute", compute)
    return tmp_path


def test_a_run_writes_its_folder_with_summary_denominators_and_slices(fake: Path) -> None:
    cfg = runs.RunConfig("finance_en", "train", stages=("text", "visual", "fused"))
    rid = runs.execute(cfg, log=False)
    doc = runs.get(rid)
    assert doc["status"] == "finished" and doc["n"] == 10 and doc["progress"]["done"] == 10
    assert doc["split_fingerprint"] == split.fingerprint("finance_en")
    s = doc["summary"]
    assert s["visual"]["ndcg_10"]["mean"] < 1.0  # odd questions' part page is never found
    assert s["visual"]["hit_full_10"] == {"mean": 1.0, "n": 10}
    assert s["visual"]["first_full_rank"] == {"median": 1, "found": 10, "n": 10}
    evens = sum(1 for q in split.ids("finance_en", "train") if q % 2 == 0)
    assert s["text"]["hit_full_10"]["mean"] == pytest.approx(evens / 10)
    assert s["text"]["first_full_rank"]["found"] == evens
    dims = doc["slices"]
    assert set(dims) == {"query type", "content", "format", "author"}
    assert sum(r["n"] for r in dims["author"]) == 10
    recs = runs.queries(rid)
    assert (
        len(recs) == 10 and recs[0]["stages"]["visual"]["ranking"][0] == 1000 + recs[0]["query_id"]
    )


def test_the_test_split_is_guarded(fake: Path) -> None:
    cfg = runs.RunConfig("finance_en", "test", stages=("text", "visual", "fused"))
    with pytest.raises(runs.TestSplitGuardError):
        runs.execute(cfg, log=False)
    runs.execute(cfg, final=True, log=False)
    with pytest.raises(runs.TestSplitGuardError):  # once per configuration
        runs.execute(cfg, final=True, log=False)
    runs.execute(cfg, final=True, force=True, log=False)
    with pytest.raises(runs.TestSplitGuardError):
        runs.execute(runs.RunConfig("finance_en", "test", limit=3), final=True, log=False)
    with pytest.raises(ValueError):
        runs.execute(runs.RunConfig("finance_en", "all"), log=False)


def test_a_failed_question_marks_the_run_failed(
    fake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: Any) -> Any:
        raise RuntimeError("model fell over")

    monkeypatch.setattr(pipeline, "compute", boom)
    with pytest.raises(RuntimeError):
        runs.execute(runs.RunConfig("finance_en", "train", stages=("text",)), log=False)
    (r,) = runs.list_runs()
    assert r["status"] == "failed" and "model fell over" in r["error"]


# ---- the API --------------------------------------------------------------------


def test_runs_api_lists_drills_and_puts_final_runs_on_the_board(
    fake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pages, "page_index", lambda key: {})
    stages = ("text", "visual", "fused")
    train = runs.execute(runs.RunConfig("finance_en", "train", stages=stages), log=False)
    test = runs.execute(runs.RunConfig("finance_en", "test", stages=stages), final=True, log=False)
    c = TestClient(app_module.create_app())
    listed = c.get("/api/runs").json()
    assert [r["run_id"] for r in listed] == sorted([train, test], reverse=True)
    assert "slices" not in listed[0]
    assert c.get(f"/api/runs/{train}").json()["slices"]["author"]
    rows = c.get(f"/api/runs/{train}/queries").json()
    qid = rows[0]["query_id"]
    assert set(rows[0]["stages"]) == set(stages)
    one = c.get(f"/api/runs/{train}/queries/{qid}").json()
    assert one["gold"][0]["grade"] == 2 and one["stages"]["visual"]["top"][0]["grade"] == 2
    assert one["stages"]["visual"]["gold_ranks"][str(1000 + qid)] == 1
    assert c.get(f"/api/runs/{train}/queries/99999").status_code == 404
    assert c.get("/api/runs/nope").status_code == 404
    ours = app_module._ours(runs.list_runs())
    assert [o["run_id"] for o in ours] == [test]
    info = c.get("/api/split/finance_en").json()
    assert info["n"] == {"train": 10, "test": 10}
    assert sum(s["train"] + s["test"] for s in info["strata"]) == 20
    r = c.post("/api/runs", json={"dataset": "finance_en", "split": "test", "stages": list(stages)})
    assert r.status_code == 409 and "final" in r.json()["detail"]


def test_run_ids_cannot_escape_the_runs_folder(fake: Path) -> None:
    with pytest.raises(KeyError):
        runs.get("../splits")
    assert json.loads(split.path("finance_en").read_text())["dataset"] == "finance_en"
