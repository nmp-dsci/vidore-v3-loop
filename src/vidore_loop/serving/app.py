"""The API behind the viewer, and the viewer itself when `frontend/dist` exists.

Every route reads the pinned datasets (through the local cache) or a committed
file. No route calls a model and none writes, except the page cache, which fills
itself from the Hub the first time a page is opened.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from vidore_loop.config import FRONTEND_DIST, settings
from vidore_loop.data import catalog, pages, registry
from vidore_loop.data import leaderboard as board

# English datasets in the order the plan ranks them (s00 §3): commercial value first
ENGLISH_ORDER = ("finance_en", "pharmaceuticals", "industrial", "hr", "computer_science")


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

    @app.get("/api/leaderboard")
    def leaderboard() -> dict[str, Any]:
        return board.leaderboard()

    _mount_frontend(app)
    return app


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
        return FileResponse(index)
