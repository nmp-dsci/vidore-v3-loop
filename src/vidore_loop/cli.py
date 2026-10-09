"""`vidoreloop` — every Make target is a thin wrapper over one of these commands."""

from __future__ import annotations

from collections import Counter

import typer
from rich.console import Console
from rich.table import Table

from vidore_loop.data import registry
from vidore_loop.eval import metrics

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

# The smoke's offline ranking: one query, graded qrels, a known NDCG@10.
SMOKE_QRELS: dict[int, int] = {11: 2, 12: 1, 13: 1}
SMOKE_RANKING: list[int] = [11, 99, 12, 98, 97]


@app.command()
def datasets() -> None:
    """The eight public datasets, their size and pinned revision."""
    t = Table("key", "domain", "docs lang", "docs", "pages", "queries", "×6 langs", "revision")
    for d in registry.DATASETS:
        t.add_row(
            d.key,
            d.domain,
            d.doc_language,
            str(d.documents),
            str(d.pages),
            str(d.queries),
            str(d.queries_all_languages),
            d.revision[:7],
        )
    total_pages = sum(d.pages for d in registry.DATASETS)
    total_q = sum(d.queries for d in registry.DATASETS)
    t.add_row("total", "", "", "", str(total_pages), str(total_q), str(total_q * 6), "")
    console.print(t)


@app.command()
def download(
    dataset: str = typer.Option("all", help="a dataset key, or 'all'"),
    corpus: bool = typer.Option(False, help="also fetch page images (0.4–1.5 GB per dataset)"),
) -> None:
    """Fetch datasets at their pinned revision into data/cache/."""
    from vidore_loop.data.loader import download as fetch

    keys = [d.key for d in registry.DATASETS] if dataset == "all" else [registry.get(dataset).key]
    for key in keys:
        rows = fetch(key, corpus=corpus)
        console.print(f"{key}: " + ", ".join(f"{c} {n}" for c, n in rows.items()))


@app.command()
def stats(dataset: str = typer.Argument(..., help="a dataset key")) -> None:
    """Queries by language, type and format, and qrels grades, for one dataset."""
    from vidore_loop.data.loader import load

    queries = load(dataset, "queries")
    qrels = load(dataset, "qrels")
    by_lang = Counter(queries["language"])
    by_type = Counter(t for ts in queries["query_types"] for t in ts)
    by_format = Counter(queries["query_format"])
    grades = Counter(qrels["score"])
    console.print(f"[bold]{dataset}[/] — {len(queries)} queries, {len(qrels)} qrels")
    for name, c in (("language", by_lang), ("type", by_type), ("format", by_format)):
        console.print(f"  {name}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    console.print("  qrels grade: " + ", ".join(f"{k} {v}" for k, v in sorted(grades.items())))


@app.command()
def pages(
    dataset: str = typer.Option(..., help="a dataset key"),
) -> None:
    """Prefetch every page image and its OCR text into data/cache/pages/<key>/ (≈10 s per 230 pages)."""
    from vidore_loop.data import pages as page_store

    written = page_store.prefetch(registry.get(dataset).key)
    console.print(f"{dataset}: {written} pages written")


@app.command()
def index(
    dataset: str = typer.Option(..., help="a dataset key"),
    stage: str = typer.Option("text", help="text | visual | reranker (downloads the model)"),
    limit: int = typer.Option(0, help="visual only: encode the first N pages (a timing run)"),
    model: str = typer.Option("evie", help="visual only: evie | tomoro"),
) -> None:
    """Build a retrieval index into data/cache/index/<key>/<stage>/."""
    key = registry.get(dataset).key
    if stage == "text":
        from vidore_loop.retrieval import text

        console.print(f"{key}: {text.build(key)} pages in the text index")
    elif stage == "visual":
        from vidore_loop.retrieval import visual

        console.print(visual.build(key, model_key=model, limit=limit or None))
    elif stage == "reranker":
        from vidore_loop.retrieval import rerank

        rerank.model()
        console.print(f"{rerank.MODEL_ID} downloaded and loaded")
    else:
        raise typer.BadParameter("stage is text, visual or reranker")


@app.command()
def serve(port: int = 8083, host: str = "127.0.0.1", reload: bool = False) -> None:
    """The viewer's API, and the built frontend when frontend/dist exists."""
    from pathlib import Path

    import uvicorn

    uvicorn.run(
        "vidore_loop.serving.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(Path(__file__).parent)] if reload else None,
    )


@app.command()
def smoke() -> None:
    """Zero-LLM proof the project logs to the central MLflow: scores a fixed ranking."""
    from vidore_loop.tracking.mlflow_log import log_smoke

    ndcg = metrics.ndcg_at_k(SMOKE_RANKING, SMOKE_QRELS, k=10)
    recall = metrics.recall_at_k(SMOKE_RANKING, SMOKE_QRELS, k=10)
    console.print(f"ndcg@10 {ndcg:.4f} · recall@10 {recall:.4f}")
    console.print(f"run_id {log_smoke(ndcg, recall)}")


@app.command("split")
def make_split(dataset: str = typer.Option(..., help="a dataset key")) -> None:
    """Write our fixed train / test split of a dataset's English questions (once)."""
    from vidore_loop.eval import split

    doc = split.write(registry.get(dataset).key)
    console.print(
        f"{split.path(doc['dataset'])}: train {doc['n']['train']} · test {doc['n']['test']}"
    )


@app.command("eval")
def evaluate(
    dataset: str = typer.Option(..., help="a dataset key"),
    split_name: str = typer.Option("train", "--split", help="train | test (test needs --final)"),
    stages: str = typer.Option("text,visual,fused,reranked", help="comma-separated stages"),
    depth: int = typer.Option(20, help="how many fused pages the reranker reads"),
    w_text: float = typer.Option(1.0, help="fusion weight of the text channel"),
    w_visual: float = typer.Option(1.0, help="fusion weight of the visual channel"),
    final: bool = typer.Option(
        False, help="score the test split, once, for a configuration chosen on train"
    ),
    force: bool = typer.Option(False, help="repeat a configuration already scored on test"),
    limit: int = typer.Option(0, help="score only the first N questions (a smoke; train only)"),
    note: str = typer.Option("", help="why this run exists"),
    in_process: bool = typer.Option(
        False, help="run here even when the viewer is up (loads a second copy of the models)"
    ),
    port: int = typer.Option(8083, help="the viewer's port"),
) -> None:
    """Score a pipeline configuration on a split: a run folder under runs/, logged to MLflow.

    When the viewer is running, the run executes inside it (one copy of the models, the
    viewer's GPU lock) and this command follows its progress."""
    import json
    import time
    import urllib.error
    import urllib.request

    from vidore_loop.eval import runs

    cfg = runs.RunConfig(
        dataset=registry.get(dataset).key,
        split=split_name,
        stages=tuple(s.strip() for s in stages.split(",") if s.strip()),
        rerank_depth=depth,
        weights=(w_text, w_visual),
        note=note,
        limit=limit or None,
    )
    base = f"http://127.0.0.1:{port}"
    viewer_up = False
    if not in_process:
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=2):  # noqa: S310 - localhost
                viewer_up = True
        except (urllib.error.URLError, OSError):
            viewer_up = False
    if not viewer_up:
        rid = runs.execute(
            cfg,
            final=final,
            force=force,
            on_progress=lambda p: console.print(f"{p.done}/{p.n} · {p.seconds:.0f}s", end="\r"),
        )
    else:
        body = {
            **{k: v for k, v in cfg.__dict__.items() if k not in ("stages", "weights")},
            "stages": list(cfg.stages),
            "weights": list(cfg.weights),
            "final": final,
            "force": force,
        }
        req = urllib.request.Request(
            f"{base}/api/runs", json.dumps(body).encode(), {"content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - localhost
                rid = json.load(r)["run_id"]
        except urllib.error.HTTPError as e:
            raise typer.BadParameter(json.load(e).get("detail", str(e))) from e
        console.print(f"running in the viewer: {rid}")
        while True:
            time.sleep(5)
            doc = runs.get(rid)
            p = doc.get("progress", {})
            console.print(f"{p.get('done', 0)}/{p.get('n', '?')} · {doc['status']}", end="\r")
            if doc["status"] != "running":
                break
    doc = runs.get(rid)
    console.print(
        f"\n{rid}: {doc['status']}" + (f" ({doc.get('error')})" if doc.get("error") else "")
    )
    if doc.get("summary"):
        t = Table("stage", "NDCG@10", "Recall@10", "Recall@50", "full hit@10", "complete@50", "MRR")
        for s, m in doc["summary"].items():
            t.add_row(
                s,
                *(
                    f"{m[k]['mean']:.3f}" if m[k]["mean"] is not None else "—"
                    for k in (
                        "ndcg_10",
                        "recall_10",
                        "recall_50",
                        "hit_full_10",
                        "complete_50",
                        "mrr",
                    )
                ),
            )
        console.print(t)
    if doc.get("mlflow_error"):
        console.print(f"[yellow]not logged to MLflow: {doc['mlflow_error']}[/yellow]")


@app.command("runs")
def list_runs() -> None:
    """Every scored run, newest first."""
    from vidore_loop.eval import runs

    t = Table("run", "split", "n", "status", "NDCG@10 per stage")
    for r in runs.list_runs():
        nd = " · ".join(
            f"{s[0]} {m['ndcg_10']['mean']:.3f}" for s, m in (r.get("summary") or {}).items()
        )
        t.add_row(r["run_id"], r["split"], str(r["n"]), r["status"], nd)
    console.print(t)


if __name__ == "__main__":
    app()
