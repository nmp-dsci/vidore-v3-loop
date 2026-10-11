"""Every run's configuration as one standard table: the same sections, rows and order for
a retrieval run and an answer run, so two runs can be read side by side.

Sections: data · retrieval · answer · judge · provenance. A row that does not apply to
a run says so ("—, retrieval only") rather than disappearing, so the shape never changes.
An answer run's retrieval rows are read from the retrieval run it answered from.
"""

from __future__ import annotations

from typing import Any

from vidore_loop.retrieval.pipeline import STAGES

Row = dict[str, Any]  # {k, v, link?, mono?}
STAGE_NAMES = {
    "text": "A · text",
    "visual": "B · visual",
    "fused": "C · fused",
    "reranked": "D · reranked",
}
NOT_ANSWERED = "—, retrieval only"


def _row(k: str, v: Any, link: str | None = None, mono: bool = False) -> Row:
    r: Row = {"k": k, "v": "—" if v is None or v == "" else str(v)}
    if link:
        r["link"] = link
    if mono:
        r["mono"] = True
    return r


def _fusion(cfg: dict[str, Any], rrf_k: Any) -> str:
    wt, wv = cfg.get("weights", [None, None])
    s = f"reciprocal rank fusion, text {wt:g} : visual {wv:g}, K = {rrf_k}"
    return s + (" (text off, so visual only)" if wt == 0 else "")


def retrieval_rows(r: dict[str, Any], stage: str | None = None) -> list[Row]:
    """A retrieval run's setup; with `stage` (an answer run's), only the stages up to it."""
    cfg, m = r.get("config", {}), r.get("models", {})
    stages = [s for s in STAGES if s in cfg.get("stages", [])]
    if stage in STAGES:
        stages = [s for s in stages if STAGES.index(s) <= STAGES.index(stage)]
    rr = "reranked" in stages
    chain = " → ".join(STAGE_NAMES[s] for s in stages)
    rows = [
        _row("pipeline", chain + (", the answers read the last" if stage else ", each scored")),
        _row("text channel", m.get("text") if "text" in stages else "off"),
        _row(
            "visual channel",
            f"{m['visual']} ({m.get('visual_dim')}-d vectors, MaxSim)"
            if m.get("visual") and "visual" in stages
            else "off",
            mono=True,
        ),
        _row("fusion", _fusion(cfg, r.get("rrf_k")) if "fused" in stages else "off"),
        _row("pools", f"each channel's top {r.get('channel_k')} → fused top {r.get('fused_k')}"),
        _row(
            "reranker",
            f"{m.get('reranker')} over the fused top {cfg.get('rerank_depth')}"
            if rr
            else ("not used" if stage else "none"),
            mono=rr,
        ),
    ]
    return rows


def describe(doc: dict[str, Any], source: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """[{section, rows}] for any run; `source` is an answer run's retrieval run, when it has one."""
    from vidore_loop.answer import judge

    cfg = doc.get("config", {})
    answer = doc.get("kind") == "answer"
    n = f"{doc.get('n')}" + (f" (the first {cfg['limit']} only)" if cfg.get("limit") else "")
    data = [
        _row("dataset", doc.get("dataset")),
        _row(
            "split", f"{doc.get('split')}" + (" · final, scored once" if doc.get("final") else "")
        ),
        _row("questions", n),
        _row(
            "split file",
            f"seed {doc.get('split_seed')} · {doc.get('split_fingerprint')}",
            mono=True,
        ),
    ]
    if answer:
        src_id = cfg.get("source_run")
        retrieval = [
            _row("source run", src_id, link=f"/runs/{src_id}" if src_id else None, mono=True),
            *(
                retrieval_rows(source, cfg.get("source_stage"))
                if source
                else [_row("retrieval", "none: the oracle setting reads the human-marked pages")]
            ),
        ]
        models = doc.get("models", {})
        modes = cfg.get("modes", [])
        given = []
        if "retrieved" in modes:
            given.append(
                f"retrieved: the top {cfg.get('k')} of {STAGE_NAMES.get(cfg.get('source_stage', ''), '')}"
            )
        if "oracle" in modes:
            given.append("oracle: every human-marked page")
        prompts = doc.get("prompts", {})
        answer_rows = [
            _row("answerer", "one Claude call per question and setting, no tools"),
            _row("model", f"{models.get('answer')} · effort {models.get('effort')}", mono=True),
            _row("pages given", " · ".join(given)),
            _row(
                "page format",
                cfg.get("modality", "").replace("+", " + ").replace("text", "OCR text"),
            ),
            _row("prompt", f"the paper's Figure 25 · {prompts.get('answer')}"),
            _row("in parallel", f"{cfg.get('workers')} questions at a time"),
        ]
        judge_rows = [
            _row(
                "model", f"{models.get('judge')} · effort {models.get('judge_effort')}", mono=True
            ),
            _row("prompt", f"the paper's Figure 24 · {prompts.get('judge')}"),
            _row("counts as right", f"{judge.VERDICTS[0]} only (the paper's binary label)"),
        ]
        billing = doc.get("billing")
    else:
        retrieval = retrieval_rows(doc)
        answer_rows = [_row("answerer", NOT_ANSWERED)]
        judge_rows = [_row("model", NOT_ANSWERED)]
        billing = "none: open models, locally"
    secs = doc.get("seconds")
    provenance = [
        _row("code", doc.get("code_sha"), mono=True),
        _row("config fingerprint", doc.get("fingerprint"), mono=True),
        _row(
            "ran",
            f"{str(doc.get('started_at', ''))[:16].replace('T', ' ')} UTC"
            + (f" · {secs / 60:.0f} min" if secs else ""),
        ),
        _row("billing", billing),
        _row(
            "MLflow",
            f"vidore-v3-loop/evals · {doc['mlflow_run_id']}"
            if doc.get("mlflow_run_id")
            else doc.get("mlflow_error", "not logged yet"),
            link=doc.get("mlflow_url"),
            mono=bool(doc.get("mlflow_run_id")),
        ),
        _row("note", cfg.get("note")),
    ]
    return [
        {"section": "data", "rows": data},
        {"section": "retrieval", "rows": retrieval},
        {"section": "answer", "rows": answer_rows},
        {"section": "judge", "rows": judge_rows},
        {"section": "provenance", "rows": provenance},
    ]
