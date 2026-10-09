"""The API behind the viewer, and the viewer itself when `frontend/dist` exists.

Most routes read the pinned datasets (through the local cache) or a committed
file. Two do more: `POST /api/search` runs the retrieval models (M2), and
`POST /api/runs` starts a scored run in a background thread of this process (M3),
which writes its folder under `runs/`. No route calls a paid model.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vidore_loop.config import FRONTEND_DIST, settings
from vidore_loop.data import catalog, pages, registry
from vidore_loop.data import leaderboard as board

# English datasets in the order the plan ranks them (s00 §3): commercial value first
ENGLISH_ORDER = ("finance_en", "pharmaceuticals", "industrial", "hr", "computer_science")


class SearchIn(BaseModel):
    """One question through the retrieval stages: a benchmark question by id, or free text."""

    dataset: str
    query: str | None = None
    query_id: int | None = None
    stages: list[str] = ["text", "visual", "fused", "reranked"]
    rerank_depth: int = 20


class RunIn(BaseModel):
    """A scored run (M3): a pipeline configuration on one split."""

    dataset: str
    split: str = "train"
    stages: list[str] = ["text", "visual", "fused", "reranked"]
    rerank_depth: int = 20
    weights: list[float] = [1.0, 1.0]
    note: str = ""
    limit: int | None = None
    final: bool = False
    force: bool = False


def _dataset(key: str) -> registry.Dataset:
    if key not in ENGLISH_ORDER:
        raise HTTPException(
            404, f"{key!r} is not one of the English datasets: {', '.join(ENGLISH_ORDER)}"
        )
    return registry.get(key)


def create_app() -> FastAPI:
    app = FastAPI(title="vidore-v3-loop")
    s = settings()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "code_sha": s.code_sha, "datasets": list(ENGLISH_ORDER)}

    @app.get("/api/datasets")
    def datasets() -> list[dict[str, Any]]:
        out = []
        for key in ENGLISH_ORDER:
            d = registry.get(key)
            out.append(
                {
                    "key": d.key,
                    "domain": d.domain,
                    "repo_id": d.repo_id,
                    "revision": d.revision,
                    "documents": d.documents,
                    "pages": d.pages,
                    "queries": d.queries,
                    "answerable_blind_pct": board.boards()["answers"]["answerable_blind_pct"][
                        board.ENGLISH.index(key)
                    ],
                }
            )
        return out

    @app.get("/api/datasets/{key}")
    def dataset(key: str) -> dict[str, Any]:
        d = _dataset(key)
        return {
            "key": d.key,
            "domain": d.domain,
            "repo_id": d.repo_id,
            "revision": d.revision,
            "paper_revision": d.paper_revision,
            "pages": d.pages,
            "documents": catalog.documents(key),
            "summary": catalog.summary(key),
        }

    @app.get("/api/datasets/{key}/queries")
    def queries(key: str) -> list[dict[str, Any]]:
        _dataset(key)
        checks = catalog.hand_checks()
        return [
            {
                k: q[k]
                for k in (
                    "query_id",
                    "query",
                    "query_types",
                    "query_format",
                    "content_type",
                    "query_generator",
                    "n_pages",
                    "n_full",
                )
            }
            | {"hand_check": (checks.get((key, q["query_id"])) or {}).get("verdict")}
            for q in catalog.queries(key)
        ]

    @app.get("/api/datasets/{key}/queries/{query_id}")
    def question(key: str, query_id: int) -> dict[str, Any]:
        _dataset(key)
        q = catalog.question(key, query_id)
        if q is None:
            raise HTTPException(404, f"no English query {query_id} in {key}")
        return q

    @app.get("/api/pages/{key}/{corpus_id}")
    def page(key: str, corpus_id: int) -> dict[str, Any]:
        _dataset(key)
        meta = pages.page_meta(key, corpus_id)
        if meta is None:
            raise HTTPException(404, f"no page {corpus_id} in {key}")
        return meta

    @app.get("/api/pages/{key}/{corpus_id}/image")
    def page_image(key: str, corpus_id: int) -> FileResponse:
        _dataset(key)
        path = pages.image_path(key, corpus_id)
        if path is None:
            raise HTTPException(404, f"no page {corpus_id} in {key}")
        return FileResponse(
            path, media_type="image/jpeg", headers={"cache-control": "max-age=86400"}
        )

    @app.get("/api/retrieval/status")
    def retrieval_status(dataset: str = "finance_en") -> dict[str, Any]:
        _dataset(dataset)
        from vidore_loop.retrieval import pipeline

        return pipeline.status(dataset)

    @app.post("/api/search")
    def search(body: SearchIn) -> dict[str, Any]:
        _dataset(body.dataset)
        from vidore_loop.retrieval import pipeline

        try:
            return pipeline.run(
                body.dataset,
                query=body.query,
                query_id=body.query_id,
                stages=tuple(st for st in body.stages if st in pipeline.STAGES),
                rerank_depth=max(1, min(body.rerank_depth, 50)),
            )
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/api/leaderboard")
    def leaderboard() -> dict[str, Any]:
        from vidore_loop.eval import runs

        return {**board.leaderboard(), "ours": _ours(runs.list_runs())}

    @app.get("/api/split/{key}")
    def split_info(key: str) -> dict[str, Any]:
        _dataset(key)
        from vidore_loop.eval import split

        try:
            doc = split.load(key)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        by_q = {int(q["query_id"]): q for q in catalog.queries(key)}
        strata: dict[str, dict[str, int]] = {}
        for which in split.SPLITS:
            for qid in doc[which]:
                name = split.stratum(by_q[int(qid)])
                strata.setdefault(name, {"train": 0, "test": 0})[which] += 1
        return {
            "dataset": key,
            "seed": doc["seed"],
            "stratified_by": doc["stratified_by"],
            "n": doc["n"],
            "fingerprint": split.fingerprint(key),
            "strata": [{"stratum": k, **v} for k, v in sorted(strata.items())],
        }

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        from vidore_loop.eval import runs

        return [{k: v for k, v in r.items() if k != "slices"} for r in runs.list_runs()]

    @app.post("/api/runs")
    def start_run(body: RunIn) -> dict[str, Any]:
        _dataset(body.dataset)
        from vidore_loop.eval import runs
        from vidore_loop.retrieval import pipeline

        if len(body.weights) != 2:
            raise HTTPException(422, "weights are [text, visual]")
        cfg = runs.RunConfig(
            dataset=body.dataset,
            split=body.split,
            stages=tuple(st for st in body.stages if st in pipeline.STAGES),
            rerank_depth=max(1, min(body.rerank_depth, 50)),
            weights=(body.weights[0], body.weights[1]),
            note=body.note,
            limit=body.limit or None,
        )
        try:
            return {"run_id": runs.start(cfg, final=body.final, force=body.force)}
        except runs.TestSplitGuardError as e:
            raise HTTPException(409, str(e)) from e
        except runs.RunBusyError as e:
            raise HTTPException(409, str(e)) from e
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        from vidore_loop.eval import runs

        try:
            return runs.get(run_id)
        except KeyError as e:
            raise HTTPException(404, f"no run {run_id}") from e

    @app.get("/api/runs/{run_id}/queries")
    def run_queries(run_id: str) -> list[dict[str, Any]]:
        from vidore_loop.eval import runs

        try:
            recs = runs.queries(run_id)
        except KeyError as e:
            raise HTTPException(404, f"no run {run_id}") from e
        keep = ("ndcg_10", "recall_10", "first_rank", "first_full_rank", "hit_full_10")
        return [
            {
                "query_id": r["query_id"],
                "query": r["query"],
                "query_types": r["query_types"],
                "content_type": r["content_type"],
                "query_format": r["query_format"],
                "author": "person" if r["query_generator"] == "human" else "synthetic",
                "n_gold": len(r["gold"]),
                "n_full": sum(1 for g in r["gold"].values() if g == 2),
                "stages": {st: {k: v["metrics"][k] for k in keep} for st, v in r["stages"].items()},
            }
            for r in recs
        ]

    @app.get("/api/runs/{run_id}/queries/{query_id}")
    def run_query(run_id: str, query_id: int) -> dict[str, Any]:
        from vidore_loop.eval import runs

        try:
            recs = runs.queries(run_id)
            doc = runs.get(run_id)
        except KeyError as e:
            raise HTTPException(404, f"no run {run_id}") from e
        rec = next((r for r in recs if r["query_id"] == query_id), None)
        if rec is None:
            raise HTTPException(404, f"query {query_id} is not in run {run_id}")
        key = doc["dataset"]
        index = pages.page_index(key)
        gold = {int(c): g for c, g in rec["gold"].items()}

        def page(cid: int, rank: int | None = None) -> dict[str, Any]:
            e = index.get(cid, {})
            return {
                "corpus_id": cid,
                "rank": rank,
                "doc_id": e.get("doc_id"),
                "page": e.get("page"),
                "grade": gold.get(cid, 0),
            }

        stages = {}
        for st, v in rec["stages"].items():
            ranking = v["ranking"]
            pos = {cid: i for i, cid in enumerate(ranking, start=1)}
            stages[st] = {
                **{k: v[k] for k in v if k != "ranking"},
                "top": [page(cid, i) for i, cid in enumerate(ranking[:10], start=1)],
                "gold_ranks": {str(cid): pos.get(cid) for cid in gold},
            }
        return {
            **{k: rec[k] for k in rec if k not in ("stages", "gold")},
            "run_id": run_id,
            "dataset": key,
            "gold": [page(c) for c in sorted(gold, key=lambda c: (-gold[c], c))],
            "stages": stages,
        }

    _mount_frontend(app)
    return app


def _ours(all_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Our leaderboard rows: every finished final run on the test split, newest first."""
    out = []
    for r in all_runs:
        if r.get("split") != "test" or not r.get("final") or r.get("status") != "finished":
            continue
        out.append(
            {
                "run_id": r["run_id"],
                "dataset": r["dataset"],
                "label": r["label"],
                "n": r["n"],
                "models": r["models"],
                "config": r["config"],
                "ndcg_10": {s: m["ndcg_10"]["mean"] for s, m in (r.get("summary") or {}).items()},
                "costs": r.get("costs"),
            }
        )
    return out


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built SPA from this process when `frontend/dist` exists (in dev, Vite serves it)."""
    if not FRONTEND_DIST.is_dir():
        return
    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")
    index = FRONTEND_DIST / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str, request: Request) -> Any:
        if path.startswith("api/"):
            raise HTTPException(404)
        candidate = (FRONTEND_DIST / path).resolve()
        if path and candidate.is_file() and FRONTEND_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        # the bundles are content-hashed; index.html must be re-checked so a rebuild shows up
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
