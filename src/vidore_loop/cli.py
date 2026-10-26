"""`vidoreloop` — every Make target is a thin wrapper over one of these commands."""

from __future__ import annotations

from collections import Counter
from typing import Any

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
def make_split(
    dataset: str = typer.Option(..., help="a dataset key"),
    extend: bool = typer.Option(False, help="deal the holdout into train and test (50/50)"),
) -> None:
    """Write our fixed train / test split of a dataset's English questions (once), or grow it."""
    from vidore_loop.eval import split

    key = registry.get(dataset).key
    doc = split.extend(key) if extend else split.write(key)
    n = doc["n"]
    console.print(
        f"{split.path(key)}: train {n['train']} · test {n['test']} · holdout {n.get('holdout', 0)}"
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
    question_encoder: str = typer.Option(
        "", help="a tuned question side of the visual model, by name (`tune-fit`); runs here"
    ),
) -> None:
    """Score a pipeline configuration on a split: a run folder under runs/, logged to MLflow.

    When the viewer is running, the run executes inside it (one copy of the models, the
    viewer's GPU lock) and this command follows its progress. A run with a tuned question
    encoder always runs here: the viewer's process may predate the encoder."""
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
        question_encoder=question_encoder or None,
    )
    base = f"http://127.0.0.1:{port}"
    viewer_up = False
    if not in_process and not question_encoder:
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


@app.command("tune")
def tune_cv(
    dataset: str = typer.Option(..., help="a dataset key"),
    settings_: str = typer.Option(
        "head-3e-5,head-1e-4,layers4-1e-5,layers4-3e-5", "--settings", help="comma-separated"
    ),
    epochs: int = typer.Option(5, help="epochs per fold, each scored on the held-out report"),
    resume: str = typer.Option("", help="carry on a stopped sweep by its runs/tune id"),
) -> None:
    """Cross-validate question-side fine-tuning of EVIE by annual report (plan s07): train
    only, the stored pages untouched, $0. Writes runs/tune/<id>/ and logs it to MLflow."""
    from vidore_loop.retrieval import tune

    names = [n.strip() for n in settings_.split(",") if n.strip()]
    unknown = [n for n in names if n not in tune.SETTINGS]
    if unknown:
        raise typer.BadParameter(f"unknown settings {unknown}; known: {list(tune.SETTINGS)}")
    rid = tune.cross_validate(
        registry.get(dataset).key, names, epochs, say=console.print, resume=resume or None
    )
    doc = tune.cv_run(rid)
    t = Table("setting", "best epoch", "NDCG@10", "gain", "95% interval")
    for name, by_epoch in doc["results"].items():
        e, r = max(by_epoch.items(), key=lambda kv: kv[1]["gain"])
        t.add_row(
            name, e, f"{r['ndcg_10']:.4f}", f"{r['gain']:+.4f}", f"[{r['lo']:+.4f}, {r['hi']:+.4f}]"
        )
    console.print(t)
    b = doc["best"]
    verdict = "passes" if b["passes"] else "does not pass"
    console.print(
        f"{rid}: EVIE {doc['base']['ndcg_10']:.4f} on {doc['base']['n']} held-out questions; "
        f"best {b['setting']} at epoch {b['epoch']}, {b['gain']:+.4f} [{b['lo']:+.4f}, "
        f"{b['hi']:+.4f}]: {verdict} the gate (+{tune.GATE:g}, interval above 0)"
    )


@app.command("tune-fit")
def tune_fit(
    dataset: str = typer.Option(..., help="a dataset key"),
    cv_run: str = typer.Option(..., help="the sweep (runs/tune/<id>) whose best setting to fit"),
    name: str = typer.Option("", help="the encoder's name (default: setting and epochs)"),
    force: bool = typer.Option(False, help="fit even though the sweep did not clear the gate"),
) -> None:
    """Retrain the sweep's best setting on every clean train pair; saves the tuned question
    tensors under data/cache/models/ (never committed). Refused unless the gate was cleared."""
    from vidore_loop.retrieval import tune

    try:
        out = tune.fit(registry.get(dataset).key, cv_run, name, force)
    except RuntimeError as e:
        raise typer.BadParameter(str(e)) from e
    console.print(f"saved {out}: score it with `vidoreloop eval ... --question-encoder {out}`")


@app.command("tune-compare")
def tune_compare(
    dataset: str = typer.Option(..., help="a dataset key"),
    base: str = typer.Option(..., help="the run as EVIE ships (e.g. the champion test run)"),
    tuned: str = typer.Option(..., help="the same configuration with the tuned encoder"),
) -> None:
    """Two scored runs question by question: the gain overall and on the questions whose gold
    pages a train question shares, against those with unseen gold pages."""
    from vidore_loop.retrieval import tune

    t = Table("questions", "n", "EVIE", "tuned", "gain", "95% interval")
    for label, r in tune.compare(base, tuned, registry.get(dataset).key).items():
        t.add_row(
            label,
            str(r["n"]),
            f"{r['base']:.4f}",
            f"{r['tuned']:.4f}",
            f"{r['gain']:+.4f}",
            f"[{r['lo']:+.4f}, {r['hi']:+.4f}]",
        )
    console.print(t)


@app.command("answer")
def answer_run(
    dataset: str = typer.Option(..., help="a dataset key"),
    split_name: str = typer.Option("train", "--split", help="train | test (test needs --final)"),
    modes: str = typer.Option("retrieved,oracle", help="settings: retrieved, oracle"),
    k: int = typer.Option(5, help="pages given in the retrieved setting"),
    source_run: str = typer.Option(
        "", help="the retrieval run to answer from (default: latest with the viewer's weights)"
    ),
    model: str = typer.Option("sonnet", help="the answering model: haiku | sonnet | opus | an id"),
    effort: str = typer.Option("medium", help="the answering model's effort: low … max"),
    judge_model: str = typer.Option("sonnet", help="the judge model (M5 S1 chose sonnet)"),
    judge_effort: str = typer.Option("medium", help="the judge's effort"),
    workers: int = typer.Option(4, help="questions answered at once"),
    final: bool = typer.Option(False, help="answer the test split, once"),
    force: bool = typer.Option(False, help="repeat a configuration already scored on test"),
    limit: int = typer.Option(0, help="answer only the first N questions (a smoke; train only)"),
    resume: str = typer.Option("", help="carry on an interrupted answer run by its id"),
    reuse: bool = typer.Option(
        True, help="copy answers already given the same pages under the same setup"
    ),
    note: str = typer.Option("", help="why this run exists"),
) -> None:
    """Answer every question of a split in each setting with Claude (subscription), then judge.

    Writes a run folder under runs/ and logs it to MLflow. Calls Claude about twice per
    question per setting: check the subscription has room before a full split."""
    from vidore_loop.eval import answers

    cfg = answers.AnswerConfig(
        dataset=registry.get(dataset).key,
        split=split_name,
        modes=tuple(m.strip() for m in modes.split(",") if m.strip()),
        k=k,
        source_run=source_run or None,
        model=model,
        effort=effort,
        judge_model=judge_model,
        judge_effort=judge_effort,
        workers=workers,
        note=note,
        limit=limit or None,
    )
    rid = answers.execute(
        cfg,
        final=final,
        force=force,
        resume=resume or None,
        reuse=reuse,
        on_progress=lambda d, n: console.print(f"{d}/{n}", end="\r"),
    )
    doc = runs_get(rid)
    console.print(f"\n{rid}: {doc['status']}")
    t = Table("setting", "accuracy", "Correct", "Partial", "Incorrect", "errors", "tokens/answer")
    for m, row in (doc.get("summary") or {}).items():
        if m == "retrieval_cost":
            continue
        v = row["verdicts"]
        acc = row["raw"]["mean"]
        t.add_row(
            m,
            "—" if acc is None else f"{acc * 100:.1f}% of {row['raw']['n']}",
            str(v["Correct"]),
            str(v["Partially Correct"]),
            str(v["Incorrect"]),
            str(row["errors"]),
            str(row["answer_tokens"] or "—"),
        )
    console.print(t)
    tot = doc.get("totals") or {}
    console.print(
        f"{tot.get('calls', 0)} calls · {tot.get('reused', 0)} answers reused · "
        f"{tot.get('input_tokens', 0):,} input tokens · "
        f"≈${tot.get('cost_usd', 0)} at per-token prices (the subscription pays)"
    )
    if doc.get("mlflow_error"):
        console.print(f"[yellow]not logged to MLflow: {doc['mlflow_error']}[/yellow]")


@app.command("agent")
def agent_run(
    dataset: str = typer.Option(..., help="a dataset key"),
    split_name: str = typer.Option("train", "--split", help="train | test (test needs --final)"),
    model: str = typer.Option("sonnet", help="the agent's model: haiku | sonnet | opus | an id"),
    effort: str = typer.Option("medium", help="the agent's effort: low … max"),
    judge_model: str = typer.Option("sonnet", help="the judge model"),
    judge_effort: str = typer.Option("medium", help="the judge's effort"),
    max_searches: int = typer.Option(5, help="searches the agent may make per question"),
    max_views: int = typer.Option(10, help="pages the agent may view per question"),
    workers: int = typer.Option(4, help="questions at once"),
    final: bool = typer.Option(False, help="run the test split, once"),
    force: bool = typer.Option(False, help="repeat a configuration already scored on test"),
    limit: int = typer.Option(0, help="only the first N questions (a smoke; train only)"),
    note: str = typer.Option("", help="why this run exists"),
) -> None:
    """The M5 agent on every question of a split: it searches, reads pages and answers.

    Scores its final 10 pages (NDCG@10) and its answer (the judge). Loads the visual model
    for search; one Agent SDK session per question on the subscription."""
    from vidore_loop.eval import agents

    cfg = agents.AgentConfig(
        dataset=registry.get(dataset).key,
        split=split_name,
        model=model,
        effort=effort,
        judge_model=judge_model,
        judge_effort=judge_effort,
        max_searches=max_searches,
        max_views=max_views,
        workers=workers,
        note=note,
        limit=limit or None,
    )
    rid = agents.execute(
        cfg, final=final, force=force, on_progress=lambda d, n: console.print(f"{d}/{n}", end="\r")
    )
    doc = runs_get(rid)
    s = doc.get("summary") or {}
    r, a = s.get("retrieval") or {}, s.get("answers") or {}
    console.print(f"\n{rid}: {doc['status']} · {doc['models']['answer']} ({cfg.effort})")
    t = Table(
        "NDCG@10",
        "recall@10",
        "accuracy",
        "cited gold",
        "searches",
        "views",
        "s / question",
        "errors",
    )
    acc = a.get("raw", {}).get("mean")
    t.add_row(
        f"{r['ndcg_10']['mean'] * 100:.1f}"
        if r.get("ndcg_10", {}).get("mean") is not None
        else "—",
        f"{r['recall_10']['mean'] * 100:.1f}"
        if r.get("recall_10", {}).get("mean") is not None
        else "—",
        "—" if acc is None else f"{acc * 100:.1f}% of {a['raw']['n']}",
        "—"
        if s.get("cited_gold", {}).get("mean") is None
        else f"{s['cited_gold']['mean'] * 100:.0f}%",
        str(s.get("searches")),
        str(s.get("views")),
        str(s.get("seconds")),
        str(s.get("errors")),
    )
    console.print(t)
    tot = doc.get("totals") or {}
    console.print(
        f"{tot.get('calls', 0)} calls · {tot.get('input_tokens', 0):,} input tokens · "
        f"≈${tot.get('cost_usd', 0)} at per-token prices (the subscription pays)"
    )
    if doc.get("mlflow_error"):
        console.print(f"[yellow]not logged to MLflow: {doc['mlflow_error']}[/yellow]")


@app.command("models")
def models(
    aliases: str = typer.Option("haiku,sonnet,opus", help="aliases to resolve, comma-separated"),
) -> None:
    """What each model alias resolves to today: one tiny call per alias (subscription)."""
    from vidore_loop.answer import claude

    t = Table("alias", "model id")
    for a in (x.strip() for x in aliases.split(",") if x.strip()):
        t.add_row(a, claude.resolve(a))
    console.print(t)


@app.command("rejudge")
def rejudge_run(
    run: str = typer.Option(..., help="the finished answer run whose answers are judged again"),
    judge_model: str = typer.Option("haiku", help="the judge to check: haiku | sonnet | opus"),
    judge_effort: str = typer.Option("high", help="its effort: low … max"),
    workers: int = typer.Option(4, help="verdicts at once"),
    final: bool = typer.Option(False, help="re-judge a test run (reporting, once)"),
    limit: int = typer.Option(0, help="only the first N questions (a smoke)"),
) -> None:
    """Judge an answer run's answers again with another model and compare the verdicts (M5 S1).

    Never re-answers. Writes runs/judge_checks/<id>/ and logs it to MLflow."""
    from vidore_loop.eval import rejudge

    cfg = rejudge.JudgeCheckConfig(
        source_run=run,
        judge_model=judge_model,
        judge_effort=judge_effort,
        workers=workers,
        limit=limit or None,
    )
    cid = rejudge.execute(
        cfg, final=final, on_progress=lambda d, n: console.print(f"{d}/{n}", end="\r")
    )
    doc = rejudge.get(cid)
    s = doc["summary"]
    console.print(f"\n{cid}: {doc['status']} · {doc['models']['judge']} ({cfg.judge_effort})")
    t = Table("scope", "n", "agree (binary)", "agree (3 levels)", "kappa", "accuracy shift")
    for scope, row in [("all", s["all"]), *s["modes"].items()]:
        t.add_row(
            scope,
            str(row["n"]),
            f"{row['binary_agree'] * 100:.1f}%",
            f"{row['three_level_agree'] * 100:.1f}%",
            f"{row['kappa']:.3f}",
            f"{row['shift'] * 100:+.1f} pts",
        )
    console.print(t)
    p = s["people"]
    if p["n"]:
        console.print(
            f"against {p['n']} labels: source {p['source'] * 100:.1f}% · check {p['check'] * 100:.1f}%"
        )
    tot = doc["totals"]
    console.print(
        f"{tot['calls']} calls · {s['errors']} errors · "
        f"≈${tot['cost_usd']} at per-token prices (the subscription pays)"
    )
    if doc.get("mlflow_error"):
        console.print(f"[yellow]not logged to MLflow: {doc['mlflow_error']}[/yellow]")


def runs_get(rid: str) -> dict[str, Any]:
    from vidore_loop.eval import runs

    return runs.get(rid)


@app.command("runs")
def list_runs() -> None:
    """Every scored run, newest first."""
    from vidore_loop.eval import runs

    t = Table("run", "split", "n", "status", "per stage / setting")
    for r in runs.list_runs():
        summ = r.get("summary") or {}
        if r.get("kind") == "answer":
            cells = " · ".join(
                f"{m} {v['raw']['mean'] * 100:.1f}%"
                for m, v in summ.items()
                if m != "retrieval_cost" and v["raw"]["mean"] is not None
            )
        elif r.get("kind") == "agent":
            ndcg = ((summ.get("retrieval") or {}).get("ndcg_10") or {}).get("mean")
            acc = ((summ.get("answers") or {}).get("raw") or {}).get("mean")
            cells = (f"agent ndcg {ndcg:.3f} · " if ndcg is not None else "agent · ") + (
                f"{acc * 100:.1f}%" if acc is not None else "—"
            )
        else:
            cells = " · ".join(f"{s[0]} {m['ndcg_10']['mean']:.3f}" for s, m in summ.items())
        t.add_row(r["run_id"], r["split"], str(r["n"]), r["status"], cells)
    console.print(t)


if __name__ == "__main__":
    app()
