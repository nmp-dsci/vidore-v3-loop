"""The retrieval pipeline, offline: fusion maths, and a text-only run over a fake index."""

from __future__ import annotations

from typing import Any

import pytest

from vidore_loop.retrieval import fuse, pipeline, text


def test_rrf_rewards_pages_both_channels_like() -> None:
    fused = fuse.rrf([[1, 2, 3], [3, 4, 1]], k=4)
    ids = [cid for cid, _ in fused]
    assert ids[:2] == [1, 3]  # in both lists
    assert set(ids) == {1, 2, 3, 4}
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_is_deterministic_on_ties() -> None:
    assert [c for c, _ in fuse.rrf([[5], [4]], k=2)] == [4, 5]


@pytest.fixture()
def fake(monkeypatch: pytest.MonkeyPatch) -> None:
    q = {
        "query": "uninsured deposits",
        "pages": [{"corpus_id": 7, "score": 2}, {"corpus_id": 9, "score": 1}],
    }
    monkeypatch.setattr(pipeline.catalog, "question", lambda key, qid: q if qid == 1 else None)
    monkeypatch.setattr(
        pipeline,
        "status",
        lambda key: {
            "text": True,
            "visual": False,
            "visual_complete": False,
            "visual_stats": None,
            "reranker": False,
        },
    )
    monkeypatch.setattr(pipeline, "_heavy_ok", lambda: False)
    monkeypatch.setattr(text, "search", lambda key, query, k=100: [(3, 9.0), (7, 8.0), (5, 1.0)])
    index: dict[int, dict[str, Any]] = {c: {"doc_id": "bank", "page": c} for c in (3, 5, 7, 9)}
    monkeypatch.setattr(pipeline.pages, "page_index", lambda key: index)
    monkeypatch.setattr(
        pipeline.pages,
        "page_meta",
        lambda key, cid: {"doc_id": "bank", "markdown": "uninsured deposits table"},
    )


def test_text_only_run_scores_against_gold(fake: None) -> None:
    out = pipeline.run("finance_en", query_id=1)
    st = out["stages"]
    assert "unavailable" in st["visual"] and "unavailable" in st["reranked"]
    t = st["text"]
    assert [h["corpus_id"] for h in t["hits"]] == [3, 7, 5]
    assert [h["grade"] for h in t["hits"]] == [0, 2, 0]
    assert t["hits"][1]["terms"] == ["uninsured", "deposits"]
    m = t["metrics"]
    assert (
        m["first_full_rank"] == 2 and m["recall_10"] == pytest.approx(0.5) and not m["complete_10"]
    )
    assert t["gold_ranks"] == {"7": 2, "9": None}
    assert st["fused"]["inputs"] == ["text"]
    assert out["deltas"]["text→fused"] == {"gained": [], "lost": []}


def test_free_text_has_no_gold_and_no_metrics(fake: None) -> None:
    out = pipeline.run("finance_en", query="deposits")
    assert out["gold"] == [] and "metrics" not in out["stages"]["text"]
    assert out["stages"]["text"]["hits"][0]["grade"] is None


def test_bad_inputs(fake: None) -> None:
    with pytest.raises(KeyError):
        pipeline.run("finance_en", query_id=999)
    with pytest.raises(ValueError):
        pipeline.run("finance_en", query="  ")
