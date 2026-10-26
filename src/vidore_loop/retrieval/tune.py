"""Question-side fine-tuning of the visual model (plan s07): the stored page vectors stay as built.

EVIE encodes pages and questions with one network. Only the question side learns here, so the
October index is never re-encoded and a tuned model is scored by the same MaxSim search:

- `head` (F0): the first 128 rows of the output head (2,560 → 128), 328K numbers.
- `layers` (F1): the last 4 language layers and that head, 446M numbers.

What it learns from (s07, the owner's call): each train question's gold pages, less every
page that is gold for a test question. Those pages are neither positives nor negatives, so no
gradient ever comes from a test gold page; on finance_en 304 of 702 pairs and 105 of 155
questions remain. A step scores the question against every stored page and lowers
−log(the softmax share of its gold pages), a full page (grade 2) counting twice a partial one.

A setting is chosen by cross-validation by annual report: a fold holds out one report's
single-report train questions, trains on the questions with no gold page in that report, and
never sees its pages, not even as negatives. The gate asks the best setting for +2 NDCG points
over EVIE as it ships, on the same questions, with a paired bootstrap 95% interval above 0;
below it the work stops and records "no gain", and `fit` refuses to save an encoder.

A sweep is a folder under runs/tune/ (run.json + cv.jsonl). A tuned encoder is the changed
tensors only, under data/cache/models/<key>/question/<name>/ (never committed), with a
meta.json whose hash joins a scored run's fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from vidore_loop.config import settings
from vidore_loop.data import catalog, pages
from vidore_loop.eval import metrics, split

MODEL_KEY = "evie"
GATE = 0.02  # +2 NDCG@10 points; metrics are on 0–1
SEED = 300
TAU = 1.0  # softmax temperature over raw MaxSim sums
BATCH = 8  # questions per optimiser step (gradients accumulated)
EPOCHS = 5  # each scored on the held-out report after every epoch
CLIP = 1.0
BOOTSTRAP = 10_000
HEAD = "1.linear."  # the Dense module after the Transformer (2,560 → 2,048)
LAYERS = "language_model.layers"
_SWAP = threading.Lock()


@dataclass(frozen=True)
class Setting:
    name: str
    part: str  # "head" (F0) | "layers" (F1)
    lr: float
    layers: int = 0  # language layers that learn, counted from the top ("layers" only)


SETTINGS: dict[str, Setting] = {
    s.name: s
    for s in (
        Setting("head-3e-5", "head", 3e-5),
        Setting("head-1e-4", "head", 1e-4),
        Setting("layers4-1e-5", "layers", 1e-5, 4),
        Setting("layers4-3e-5", "layers", 3e-5, 4),
    )
}


# ---- what it learns from ---------------------------------------------------------


@dataclass(frozen=True)
class Example:
    query_id: int
    query: str
    gold: dict[int, int]  # the gold pages it may learn from: test gold pages removed


@dataclass(frozen=True)
class Data:
    examples: list[Example]  # train questions left with at least one clean gold page
    blocked: frozenset[int]  # every test question's gold pages: never a positive or a negative
    pairs_all: int
    pairs_clean: int

    def fingerprint(self) -> str:
        doc = {
            "examples": [[e.query_id, sorted(e.gold.items())] for e in self.examples],
            "blocked": sorted(self.blocked),
        }
        return hashlib.sha256(json.dumps(doc).encode()).hexdigest()[:12]


def _gold(key: str, qid: int) -> dict[int, int]:
    return {int(p["corpus_id"]): int(p["score"]) for p in catalog.qrels(key).get(qid, [])}


def data(key: str) -> Data:
    """The train questions' clean (question, gold page) pairs and the pages kept out of training."""
    text = {int(q["query_id"]): q["query"] for q in catalog.queries(key)}
    blocked = frozenset(c for qid in split.ids(key, "test") for c in _gold(key, qid))
    examples, n_all = [], 0
    for qid in split.ids(key, "train"):
        g = _gold(key, qid)
        n_all += len(g)
        clean = {c: s for c, s in g.items() if c not in blocked and s > 0}
        if clean:
            examples.append(Example(qid, text[qid], clean))
    return Data(examples, blocked, n_all, sum(len(e.gold) for e in examples))


@dataclass(frozen=True)
class Fold:
    report: str
    held_out: list[int]  # this report's single-report train questions
    train: list[Example]  # questions with no gold page in this report
    blocked: frozenset[int]  # the test gold pages and every page of this report


def folds(key: str, d: Data | None = None) -> list[Fold]:
    """One fold per annual report; a question spanning reports is never held out."""
    d = d or data(key)
    index = pages.page_index(key)
    reports = {
        qid: {index[c]["doc_id"] for c in _gold(key, qid)} for qid in split.ids(key, "train")
    }
    out = []
    for report in sorted({r for rs in reports.values() for r in rs}):
        held = sorted(qid for qid, rs in reports.items() if rs == {report})
        if not held:
            continue
        train = [e for e in d.examples if report not in reports[e.query_id]]
        own = frozenset(c for c, e in index.items() if e["doc_id"] == report)
        out.append(Fold(report, held, train, d.blocked | own))
    return out


# ---- the gate ----------------------------------------------------------------------


def paired_bootstrap(
    base: list[float], tuned: list[float], n: int = BOOTSTRAP, seed: int = SEED
) -> dict[str, float]:
    """The mean gain (tuned − base) over the same questions and its 95% interval."""
    d = np.asarray(tuned, dtype=float) - np.asarray(base, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    means = np.sort(d[idx].mean(axis=1))
    return {
        "gain": round(float(d.mean()), 4),
        "lo": round(float(means[int(0.025 * n)]), 4),
        "hi": round(float(means[int(0.975 * n) - 1]), 4),
        "n": len(d),
    }


def passes(b: dict[str, float]) -> bool:
    return b["gain"] >= GATE and b["lo"] > 0


def gate_allows(cv: dict[str, Any], setting: str, epochs: int) -> bool:
    """A finished sweep whose best setting cleared the gate, at this setting and epoch count."""
    best = cv.get("best") or {}
    return bool(
        cv.get("status") == "finished"
        and best.get("passes")
        and best.get("setting") == setting
        and best.get("epoch") == epochs
    )


# ---- the model ---------------------------------------------------------------------


def info_nce(scores: Any, positives: dict[int, int], keep: Any, tau: float = TAU) -> Any:
    """−log of the softmax share of the gold pages among the pages a question may see.

    `positives` maps a page's row to its grade; the log-grade weight makes a full page (2)
    count twice a partial one (1), so the loss bottoms out at −log 2, not 0, when every
    full page holds the whole softmax. Rows where `keep` is False take no part at all."""
    import torch

    s = scores.masked_fill(~keep, float("-inf")) / tau
    rows = torch.tensor(list(positives), device=s.device)
    w = torch.tensor([float(g) for g in positives.values()], device=s.device).log()
    return -(torch.logsumexp(s[rows] + w, 0) - torch.logsumexp(s, 0))


def maxsim_train(q: Any, docs: Any, mask: Any, chunk: int = 256) -> Any:
    """MaxSim for training: each token's best patch is picked without gradients, then the
    picked patches are scored in fp32. Same scores as `visual.maxsim` (to fp16 rounding) and
    the same gradient; on MPS the backward of `max` over the fp16 grid now and then returned
    NaN for a finite loss (question 33, 2026-10-25), and this path has none."""
    import torch

    out = []
    for i in range(0, docs.shape[0], chunk):
        d = docs[i : i + chunk]
        with torch.no_grad():
            sim = torch.einsum("qd,ntd->nqt", q.to(d.dtype), d).float()
            best = sim.masked_fill(~mask[i : i + chunk, None, :], float("-inf")).argmax(dim=2)
        picked = torch.gather(d, 1, best[..., None].expand(-1, -1, d.shape[2])).float()
        out.append((picked * q[None]).sum(-1).sum(-1))
    return torch.cat(out)


def param_names(enc: Any, s: Setting) -> list[str]:
    """The parameters a setting trains: the head, plus the top `s.layers` language layers."""
    names = [n for n, _ in enc.named_parameters() if n.startswith(HEAD)]
    if s.part == "layers":
        prefix, layers = next((n, m) for n, m in enc.named_modules() if n.endswith(LAYERS))
        keep = tuple(f"{prefix}.{i}." for i in range(len(layers) - s.layers, len(layers)))
        names += [n for n, _ in enc.named_parameters() if n.startswith(keep)]
    if not names:
        raise RuntimeError(f"no parameters found for {s.name}")
    return names


def _free() -> None:
    """Return freed tensors to the system: the MPS allocator otherwise keeps them cached."""
    import gc

    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class Tuner:
    """EVIE with its question side trainable in place, the stored pages on the device, and the
    shipped weights kept, so every fold and setting starts from the model as it ships."""

    def __init__(self, key: str, trained: list[Setting]) -> None:
        from vidore_loop.retrieval import visual

        if visual.active(key) != MODEL_KEY:
            raise RuntimeError(f"{key} has no complete {MODEL_KEY} index")
        self.key = key
        self.enc = visual.model(MODEL_KEY)
        self.dim = visual.MODELS[MODEL_KEY].dim
        self.docs, self.mask, self.ids = visual._load(key, MODEL_KEY)
        self.row = {c: i for i, c in enumerate(self.ids)}
        self.named = dict(self.enc.named_parameters())
        for p in self.named.values():
            p.requires_grad_(False)
        wanted = {n for s in trained for n in param_names(self.enc, s)}
        self.base = {n: self.named[n].detach().clone() for n in wanted}
        self._lm: dict[int, dict[str, Any]] = {}

    def reset(self) -> None:
        import torch

        with torch.no_grad():
            for n, t in self.base.items():
                self.named[n].copy_(t)

    def trainable(self, s: Setting) -> int:
        """Numbers a setting can change (the head's rows past `dim` never reach a score)."""
        n = 0
        for name in param_names(self.enc, s):
            p = self.named[name]
            n += (
                (p.shape[1] * self.dim if p.ndim == 2 else self.dim)
                if name.startswith(HEAD)
                else p.numel()
            )
        return n

    def _features(self, text: str) -> dict[str, Any]:
        f = self.enc.preprocess([text], task="query")
        return {k: v.to(self.docs.device) if hasattr(v, "to") else v for k, v in f.items()}

    def _run(self, f: dict[str, Any], start: int = 0, stop: int | None = None) -> dict[str, Any]:
        """The encoder's modules from `start` to `stop`, run as its own forward runs them."""
        kwargs_of = getattr(self.enc, "module_kwargs", None) or {}
        for name, module in list(self.enc.named_children())[start:stop]:
            keys = kwargs_of.get(name, [])
            wants = "task" in keys or "task" in getattr(module, "forward_kwargs", ())
            f = module(f, **({"task": "query"} if wants else {}))
        return f

    def _vectors(self, out: dict[str, Any]) -> Any:
        import torch.nn.functional as F  # noqa: N812

        te = out["token_embeddings"][0][out["attention_mask"][0].bool()]
        return F.normalize(te[:, : self.dim].float(), dim=-1)

    def question(self, ex: Example, s: Setting) -> Any:
        """A train question's vectors with gradients; the head setting reuses the frozen
        Transformer's output, computed once per question."""
        import torch

        if s.part != "head":
            return self._vectors(self.enc(self._features(ex.query), task="query"))
        if ex.query_id not in self._lm:
            with torch.no_grad():
                out = self._run(self._features(ex.query), 0, 1)
            self._lm[ex.query_id] = {
                k: v.detach() if hasattr(v, "detach") else v for k, v in out.items()
            }
        return self._vectors(self._run(dict(self._lm[ex.query_id]), 1))

    def check(self, text: str) -> float:
        """The training path against `encode_query`: the lowest cosine over the vectors."""
        import torch

        from vidore_loop.retrieval import visual

        ref = torch.from_numpy(
            visual._cut(self.enc.encode_query([text], show_progress_bar=False)[0], self.dim)
        )
        with torch.no_grad():
            full = self._vectors(self.enc(self._features(text), task="query")).cpu()
            split = self._vectors(self._run(self._run(self._features(text), 0, 1), 1)).cpu()
        return float(min((full * ref.float()).sum(-1).min(), (split * ref.float()).sum(-1).min()))

    def keep(self, blocked: frozenset[int]) -> Any:
        import torch

        keep = torch.ones(len(self.ids), dtype=torch.bool, device=self.docs.device)
        rows = [self.row[c] for c in blocked if c in self.row]
        if rows:
            keep[torch.tensor(rows, device=self.docs.device)] = False
        return keep

    def train(
        self,
        examples: list[Example],
        blocked: frozenset[int],
        s: Setting,
        epochs: int,
        after_epoch: Callable[[int, float, int], None] | None = None,
    ) -> None:
        """Train in place from the current weights: each question's gradient, if finite, is
        added to fp32 master copies; every BATCH questions AdamW steps on their mean, clipped.
        `after_epoch` gets the epoch, its mean loss and the questions skipped as non-finite."""
        import torch

        from vidore_loop.retrieval import visual

        names = param_names(self.enc, s)
        params = [self.named[n] for n in names]
        masters = [p.detach().float().clone().requires_grad_(True) for p in params]
        for m in masters:
            m.grad = torch.zeros_like(m)  # one fp32 buffer per tensor, reused every step
        # per-tensor updates: the foreach path builds whole-model temporaries (≈2 GB here)
        opt = torch.optim.AdamW(masters, lr=s.lr, weight_decay=0.0, foreach=False)
        keep = self.keep(blocked)
        order = list(examples)
        rng = random.Random(SEED)
        for p in params:
            p.requires_grad_(True)
        try:
            for epoch in range(1, epochs + 1):
                _free()
                free = visual.free_memory_pct()
                if free is not None and free < visual.MIN_FREE_PCT:
                    raise visual.LowMemoryError(
                        f"{free}% memory free: stopped before epoch {epoch}"
                    )
                rng.shuffle(order)
                losses, skipped, added = [], 0, 0
                for i, ex in enumerate(order, start=1):
                    scores = maxsim_train(self.question(ex, s), self.docs, self.mask)
                    rows = {self.row[c]: g for c, g in ex.gold.items() if c in self.row}
                    loss = info_nce(scores, rows, keep)
                    loss.backward()
                    losses.append(float(loss.detach()))
                    if self._accumulate(params, masters):
                        added += 1
                    else:
                        skipped += 1
                    if (i % BATCH == 0 or i == len(order)) and added:
                        self._step(names, params, masters, opt, added)
                        added = 0
                if after_epoch:
                    after_epoch(epoch, sum(losses) / len(losses), skipped)
        finally:
            for p in params:
                p.requires_grad_(False)
                p.grad = None
            opt.state.clear()
            masters.clear()
            _free()

    def _accumulate(self, params: list[Any], masters: list[Any]) -> bool:
        """Add one question's gradient to the fp32 buffers when every element is finite
        (one sum per tensor, checked in one sync); clear it either way."""
        import torch

        sums = [torch.sum(p.grad, dtype=torch.float32) for p in params if p.grad is not None]
        ok = not sums or bool(torch.isfinite(torch.stack(sums)).all())
        for p, m in zip(params, masters, strict=True):
            if ok and p.grad is not None:
                m.grad.add_(p.grad)
            p.grad = None
        return ok

    def _step(
        self, names: list[str], params: list[Any], masters: list[Any], opt: Any, n: int
    ) -> None:
        import torch

        for name, m in zip(names, masters, strict=True):
            if name.startswith(HEAD):
                m.grad[self.dim :] = 0  # rows past the cut never reach a score
            m.grad.div_(n)
        torch.nn.utils.clip_grad_norm_(masters, CLIP)
        opt.step()
        opt.zero_grad(set_to_none=False)
        with torch.no_grad():
            for p, m in zip(params, masters, strict=True):
                p.copy_(m)
            sums = torch.stack([torch.sum(m, dtype=torch.float32) for m in masters])
        if not bool(torch.isfinite(sums).all()):
            raise FloatingPointError("a weight went non-finite after a step; the run stops")

    def ndcg(self, qids: list[int]) -> dict[int, float]:
        """NDCG@10 of each question with the weights as they stand, through `visual.search`."""
        from vidore_loop.retrieval import pipeline, visual

        out = {}
        for qid in qids:
            q = catalog.question(self.key, qid)
            assert q is not None, qid
            ranking = [c for c, _ in visual.search(self.key, q["query"], pipeline.CHANNEL_K)]
            out[qid] = float(metrics.report(ranking, _gold(self.key, qid))["ndcg_10"])
        return out


# ---- a sweep -----------------------------------------------------------------------


def tune_dir() -> Path:
    from vidore_loop.eval import runs

    return runs.runs_dir() / "tune"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(path)


def base_revision() -> str | None:
    """The snapshot of the base model on disk (it is not pinned; s06 E1)."""
    from huggingface_hub import try_to_load_from_cache

    from vidore_loop.retrieval import visual

    p = try_to_load_from_cache(visual.MODELS[MODEL_KEY].model_id, "config.json")
    return Path(p).parent.name if isinstance(p, str) else None


def cross_validate(
    key: str,
    names: list[str],
    epochs: int = EPOCHS,
    log: bool = True,
    say: Callable[[str], None] = print,
    resume: str | None = None,
) -> str:
    """Sweep settings across the report folds; writes runs/tune/<id>/ and logs it to MLflow.
    `resume` carries on a stopped sweep: its base scores and every (setting, fold) that ran all
    its epochs are kept, and the rest is run again."""
    from vidore_loop.eval import runs
    from vidore_loop.retrieval import visual
    from vidore_loop.tracking import mlflow_log

    chosen = [SETTINGS[n] for n in names]
    if log:
        mlflow_log.preflight()
    d = data(key)
    fs = folds(key, d)
    held = [qid for f in fs for qid in f.held_out]
    kept: list[dict[str, Any]] = []
    if resume:
        doc = cv_run(resume)
        rid, folder = resume, tune_dir() / resume
        same = (
            doc["dataset"] == key
            and [x["name"] for x in doc["settings"]] == names
            and doc["epochs"] == epochs
            and doc["data"]["fingerprint"] == d.fingerprint()
        )
        if not same or doc["status"] == "finished":
            raise ValueError(f"{resume} is finished or ran other settings, epochs or data")
        lines = [json.loads(x) for x in (folder / "cv.jsonl").read_text().splitlines() if x]
        runs_of: dict[tuple[str, str], set[int]] = {}
        for x in lines[1:]:
            runs_of.setdefault((x["setting"], x["fold"]), set()).add(x["epoch"])
        done = {k for k, es in runs_of.items() if es == set(range(1, epochs + 1))}
        kept = lines[:1] + [x for x in lines[1:] if (x["setting"], x["fold"]) in done]
        doc.update(status="running", error=None, finished_at=None)
        doc.setdefault("resumed_at", []).append(_now())
        doc["skipped"] = sum(x.get("skipped", 0) for x in kept[1:])
    else:
        rid = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{key}_cv"
        folder = tune_dir() / rid
        folder.mkdir(parents=True, exist_ok=True)
        doc = {
            "run_id": rid,
            "kind": "tune-cv",
            "dataset": key,
            "split": "train",
            "model": visual.MODELS[MODEL_KEY].model_id,
            "base_revision": base_revision(),
            "dim": visual.MODELS[MODEL_KEY].dim,
            "settings": [asdict(x) for x in chosen],
            "epochs": epochs,
            "batch": BATCH,
            "tau": TAU,
            "clip": CLIP,
            "seed": SEED,
            "gate": GATE,
            "data": {
                "pairs_all": d.pairs_all,
                "pairs_clean": d.pairs_clean,
                "questions": len(d.examples),
                "blocked_pages": len(d.blocked),
                "fingerprint": d.fingerprint(),
            },
            "folds": [
                {"report": f.report, "held_out": len(f.held_out), "train_questions": len(f.train)}
                for f in fs
            ],
            "split_fingerprint": split.fingerprint(key),
            "code_sha": runs.code_version(),
            "started_at": _now(),
            "finished_at": None,
            "status": "running",
            "results": None,
            "best": None,
            "skipped": 0,
            "mlflow_run_id": None,
        }
    _write(folder / "run.json", doc)
    t0 = time.time()
    scored: dict[str, dict[int, dict[int, float]]] = {
        x.name: {e: {} for e in range(1, epochs + 1)} for x in chosen
    }
    for x in kept[1:]:
        scored[x["setting"]][x["epoch"]].update({int(q): v for q, v in x["ndcg"].items()})
    finished = {(x["setting"], x["fold"]) for x in kept[1:]}
    try:
        tuner = Tuner(key, chosen)
        doc["check_cosine"] = round(tuner.check(d.examples[0].query), 5)
        doc["trainable"] = {x.name: tuner.trainable(x) for x in chosen}
        if kept:
            base = {int(q): v for q, v in kept[0]["ndcg"].items()}
            (folder / "cv.jsonl").write_text("".join(json.dumps(x) + "\n" for x in kept))
        with (folder / "cv.jsonl").open("a" if kept else "w") as out:
            if not kept:
                ts = time.time()
                base = tuner.ndcg(held)
                out.write(json.dumps({"setting": "base", "epoch": 0, "ndcg": base}) + "\n")
                say(
                    f"base: NDCG@10 {np.mean(list(base.values())):.4f} over {len(base)} held-out"
                    f" questions ({time.time() - ts:.0f}s)"
                )
            for s in chosen:
                for f in fs:
                    if (s.name, f.report) in finished:
                        continue
                    tuner.reset()
                    ts = time.time()

                    def after(
                        epoch: int,
                        loss: float,
                        skipped: int,
                        f: Fold = f,
                        s: Setting = s,
                        ts: float = ts,
                    ) -> None:
                        got = tuner.ndcg(f.held_out)
                        scored[s.name][epoch].update(got)
                        line = {
                            "setting": s.name,
                            "fold": f.report,
                            "epoch": epoch,
                            "loss": round(loss, 4),
                            "skipped": skipped,
                            "ndcg": got,
                        }
                        out.write(json.dumps(line) + "\n")
                        out.flush()
                        doc["skipped"] = doc.get("skipped", 0) + skipped
                        gain = np.mean([got[q] - base[q] for q in got])
                        say(
                            f"{s.name} · {f.report} · epoch {epoch}: loss {loss:.3f}, Δ {gain:+.4f}"
                            + (f", {skipped} skipped as non-finite" if skipped else "")
                            + f" ({time.time() - ts:.0f}s)"
                        )

                    tuner.train(f.train, f.blocked, s, epochs, after)
            tuner.reset()
    except BaseException as e:
        doc.update(status="failed", error=f"{type(e).__name__}: {e}", finished_at=_now())
        _write(folder / "run.json", doc)
        raise
    results: dict[str, Any] = {}
    best: dict[str, Any] | None = None
    for s in chosen:
        results[s.name] = {}
        for epoch, got in scored[s.name].items():
            b = paired_bootstrap([base[q] for q in held], [got[q] for q in held])
            results[s.name][str(epoch)] = {
                "ndcg_10": round(float(np.mean([got[q] for q in held])), 4),
                **b,
            }
            if best is None or b["gain"] > best["gain"]:
                best = {"setting": s.name, "epoch": epoch, **b}
    assert best is not None
    best["passes"] = passes(best)
    doc.update(
        status="finished",
        finished_at=_now(),
        seconds=round(doc.get("seconds", 0) + time.time() - t0, 1),
        base={"ndcg_10": round(float(np.mean([base[q] for q in held])), 4), "n": len(held)},
        results=results,
        best=best,
    )
    _write(folder / "run.json", doc)
    if log:
        try:
            doc["mlflow_run_id"] = mlflow_log.log_tune_run(doc, folder)
        except Exception as e:  # the folder is the record; a logging failure must not lose it
            doc["mlflow_error"] = f"{type(e).__name__}: {e}"
        _write(folder / "run.json", doc)
    return rid


def cv_run(run_id: str) -> dict[str, Any]:
    p = tune_dir() / run_id / "run.json"
    if "/" in run_id or ".." in run_id or not p.exists():
        raise KeyError(run_id)
    doc: dict[str, Any] = json.loads(p.read_text())
    return doc


# ---- a tuned encoder ---------------------------------------------------------------


def question_dir(key: str) -> Path:
    return settings().hf_home / "models" / key / "question"


def fit(key: str, cv_run_id: str, name: str = "", force: bool = False) -> str:
    """Retrain the sweep's best setting on every clean train pair and save the changed tensors."""
    from safetensors.torch import save_file

    from vidore_loop.eval import runs
    from vidore_loop.retrieval import visual

    cv = cv_run(cv_run_id)
    best = cv.get("best") or {}
    s, epochs = SETTINGS[best["setting"]], int(best["epoch"])
    if not force and not gate_allows(cv, s.name, epochs):
        raise RuntimeError(
            f"{cv_run_id} did not clear the gate (+{GATE * 100:g} NDCG, interval above 0)"
        )
    name = name or f"{s.name}-e{epochs}"
    out = question_dir(key) / name
    if (out / "meta.json").exists():
        raise FileExistsError(f"{out} exists; pick another --name")
    d = data(key)
    tuner = Tuner(key, [s])
    tuner.reset()
    t0 = time.time()
    curve: list[dict[str, Any]] = []
    tuner.train(
        d.examples,
        d.blocked,
        s,
        epochs,
        lambda e, loss, skipped: curve.append(
            {"epoch": e, "loss": round(loss, 4), "skipped": skipped}
        ),
    )
    tensors = {n: tuner.named[n].detach().cpu().contiguous() for n in param_names(tuner.enc, s)}
    tuner.reset()
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "weights.safetensors"))
    sha = hashlib.sha256((out / "weights.safetensors").read_bytes()).hexdigest()
    meta = {
        "name": name,
        "dataset": key,
        "model_key": MODEL_KEY,
        "model": visual.MODELS[MODEL_KEY].model_id,
        "base_revision": base_revision(),
        "setting": asdict(s),
        "epochs": epochs,
        "cv_run": cv_run_id,
        "cv_best": best,
        "data": {
            "pairs_clean": d.pairs_clean,
            "questions": len(d.examples),
            "fingerprint": d.fingerprint(),
        },
        "curve": curve,
        "tensors": len(tensors),
        "sha256": sha,
        "seconds": round(time.time() - t0, 1),
        "code_sha": runs.code_version(),
        "created_at": _now(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return name


def describe(key: str, name: str) -> dict[str, Any]:
    """What a run's fingerprint records of a tuned encoder: its name, setting and hash."""
    p = question_dir(key) / name / "meta.json"
    if "/" in name or ".." in name or not p.exists():
        raise KeyError(f"no tuned question encoder {name!r} for {key}")
    m = json.loads(p.read_text())
    return {
        "name": m["name"],
        "setting": m["setting"]["name"],
        "epochs": m["epochs"],
        "sha256": m["sha256"][:12],
    }


@cache
def _tensors(key: str, name: str, device: str) -> dict[str, Any]:
    from safetensors.torch import load_file

    return {
        n: t.to(device)
        for n, t in load_file(str(question_dir(key) / name / "weights.safetensors")).items()
    }


@contextmanager
def applied(enc: Any, key: str, model_key: str, name: str | None) -> Iterator[None]:
    """A tuned encoder's question weights in place of the shipped ones for the block; the
    stored page vectors were made with the shipped weights and are untouched."""
    if not name:
        yield
        return
    from vidore_loop.retrieval import visual

    meta = describe(key, name)
    if model_key != MODEL_KEY:
        raise RuntimeError(f"{meta['name']} tunes {MODEL_KEY}, not {model_key}")
    tensors = _tensors(key, name, visual.device())
    named = dict(enc.named_parameters())
    with _SWAP:
        saved = {n: named[n].data for n in tensors}
        try:
            for n, t in tensors.items():
                named[n].data = t
            yield
        finally:
            for n, t in saved.items():
                named[n].data = t


def compare(base_run: str, tuned_run: str, key: str) -> dict[str, Any]:
    """Two scored runs on the same split, question by question: the gain overall and on the
    questions that share a gold page with train (the leak's slice) and those that don't."""
    from vidore_loop.eval import runs

    def by_q(rid: str) -> dict[int, float]:
        return {
            int(r["query_id"]): float(r["stages"]["visual"]["metrics"]["ndcg_10"])
            for r in runs.queries(rid)
        }

    a, b = by_q(base_run), by_q(tuned_run)
    if set(a) != set(b):
        raise ValueError("the two runs scored different questions")
    train_gold = {c for qid in split.ids(key, "train") for c in _gold(key, qid)}
    shared = [q for q in a if set(_gold(key, q)) & train_gold]
    unseen = [q for q in a if q not in set(shared)]
    out = {}
    for label, qs in (
        ("all", sorted(a)),
        ("shares a train gold page", shared),
        ("unseen gold pages", unseen),
    ):
        out[label] = {
            "base": round(float(np.mean([a[q] for q in qs])), 4),
            "tuned": round(float(np.mean([b[q] for q in qs])), 4),
            **paired_bootstrap([a[q] for q in qs], [b[q] for q in qs]),
        }
    return out
