"""The viewer's API, offline: the catalog and page store are replaced with fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from vidore_loop.data import catalog, pages
from vidore_loop.data import leaderboard as board
from vidore_loop.serving import app as app_module

QUERIES = [
    {
        "query_id": 181,
        "query": "What were Morgan Stanley's total deposits in 2024?",
        "query_types": ["extractive", "numerical"],
        "query_format": "question",
        "content_type": ["Table", "Text"],
        "query_generator": "human",
        "query_generation_pipeline": None,
        "query_type_for_generation": None,
        "source_type": None,
        "answer": "$376,007 million",
        "raw_answers": ["$376,007 million"],
        "n_pages": 2,
        "n_full": 1,
    }
]
QRELS = {
    181: [
        {
            "corpus_id": 2037,
            "score": 2,
            "content_type": ["Table"],
            "boxes": [{"annotator": 1, "x1": 1, "y1": 2, "x2": 3, "y2": 4}],
        },
        {"corpus_id": 2225, "score": 1, "content_type": ["Text"], "boxes": []},
    ]
}


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(catalog, "queries", lambda key: QUERIES if key == "finance_en" else [])
    monkeypatch.setattr(catalog, "qrels", lambda key: QRELS)
    monkeypatch.setattr(catalog, "documents", lambda key: [])
    checks = tmp_path / "hand_checks.jsonl"
    checks.write_text(
        json.dumps({"dataset": "finance_en", "query_id": 181, "verdict": "holds"}) + "\n"
    )
    monkeypatch.setattr(catalog, "HAND_CHECKS", checks)
    img = tmp_path / "2037.jpg"
    img.write_bytes(b"\xff\xd8fake")
    meta: dict[str, Any] = {
        "corpus_id": 2037,
        "doc_id": "morgan_stanley_2024",
        "page": 119,
        "width": 1700,
        "height": 2200,
        "markdown": "Deposits",
    }
    monkeypatch.setattr(pages, "page_meta", lambda key, cid: meta if cid == 2037 else None)
    monkeypatch.setattr(pages, "image_path", lambda key, cid: img if cid == 2037 else None)
    return TestClient(app_module.create_app())


def test_health_and_datasets(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    ds = client.get("/api/datasets").json()
    assert [d["key"] for d in ds] == list(app_module.ENGLISH_ORDER)
    assert ds[0]["answerable_blind_pct"] == 31.7


def test_non_english_dataset_is_refused(client: TestClient) -> None:
    assert client.get("/api/datasets/finance_fr").status_code == 404


def test_query_list_and_one_question(client: TestClient) -> None:
    rows = client.get("/api/datasets/finance_en/queries").json()
    assert rows[0]["query_id"] == 181 and rows[0]["hand_check"] == "holds"
    assert "answer" not in rows[0]  # the list stays light; the answer is on the question
    q = client.get("/api/datasets/finance_en/queries/181").json()
    assert q["answer"] == "$376,007 million"
    assert [p["corpus_id"] for p in q["pages"]] == [2037, 2225]
    assert q["hand_check"]["verdict"] == "holds"
    assert client.get("/api/datasets/finance_en/queries/999").status_code == 404


def test_dataset_summary(client: TestClient) -> None:
    s = client.get("/api/datasets/finance_en").json()["summary"]
    assert s["n_queries"] == 1 and s["hand_checked"] == 1
    assert dict(s["by_type"]) == {"extractive": 1, "numerical": 1}


def test_pages(client: TestClient) -> None:
    assert client.get("/api/pages/finance_en/2037").json()["page"] == 119
    r = client.get("/api/pages/finance_en/2037/image")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/api/pages/finance_en/1/image").status_code == 404


def test_leaderboard_from_committed_files(client: TestClient) -> None:
    b = client.get("/api/leaderboard").json()
    assert b["datasets"] == list(board.ENGLISH)
    assert b["pipelines"][0]["mean"] == 74.6
    top = b["models"][0]
    assert top["model"] == "tencent/EVIE-8B" and top["mean"] == 71.7
    assert all(len(m["scores"]) == 5 for m in b["models"])
