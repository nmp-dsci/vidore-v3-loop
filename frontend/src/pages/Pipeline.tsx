import { useEffect, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import {
  type Health,
  type Hit,
  type LiveAnswer,
  type Metrics,
  type Question,
  type QueryRow,
  type RetrievalStatus,
  type SearchOut,
  STAGES,
  STAGE_LABEL,
  type Stage,
  datasetLabel,
  pageImage,
  post,
  useGet,
} from '../lib/api';
import { PageView } from '../lib/page';
import { Busy, DatasetChips, Loading } from '../lib/ui';
import { AnswerText, VerdictChip } from './Review';

/**
 * `/pipeline?dataset=&qid=` or `?q=` — one question through the RAG pipeline: every
 * retrieval stage's top 10, scored against the gold pages when the question has them (M2),
 * then, on request, Claude's answer from the last stage's top 5, judged against the
 * reference (M4). The URL holds the question, so a run is a link.
 */
export function Pipeline() {
  const [sp] = useSearchParams();
  const nav = useNavigate();
  const dataset = sp.get('dataset') ?? 'finance_en';
  const qid = sp.get('qid');
  const qtext = sp.get('q') ?? '';
  const depth = Number(sp.get('depth') ?? 20);
  const [draft, setDraft] = useState(qtext);
  const [on, setOn] = useState<Record<Stage, boolean>>({
    text: true,
    visual: true,
    fused: true,
    reranked: true,
  });
  const [res, setRes] = useState<SearchOut | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<number | null>(null); // when the search started
  const [open, setOpen] = useState<number | null>(null);
  const status = useGet<RetrievalStatus>(`/api/retrieval/status?dataset=${dataset}`);
  const question = useGet<Question>(qid ? `/api/datasets/${dataset}/queries/${qid}` : null);
  const bench = useGet<QueryRow[]>(`/api/datasets/${dataset}/queries`);

  useEffect(() => setDraft(qtext), [qtext]);
  useEffect(() => {
    if (!qid && !qtext) {
      setRes(null);
      return;
    }
    let alive = true;
    setBusy(Date.now());
    setErr(null);
    setOpen(null);
    post<SearchOut>('/api/search', {
      dataset,
      query_id: qid ? Number(qid) : null,
      query: qid ? null : qtext,
      stages: STAGES.filter((s) => on[s]),
      rerank_depth: depth,
    })
      .then((r) => alive && setRes(r))
      .catch((e: Error) => alive && setErr(e.message))
      .finally(() => alive && setBusy(null));
    return () => {
      alive = false;
    };
  }, [dataset, qid, qtext, on, depth]);

  const runFree = (e: React.FormEvent) => {
    e.preventDefault();
    if (draft.trim()) nav(`/pipeline?dataset=${dataset}&q=${encodeURIComponent(draft.trim())}`);
  };
  const shown = STAGES.filter((s) => on[s] && res?.stages[s]);
  const goldBoxes = (cid: number) => question.data?.pages.find((p) => p.corpus_id === cid)?.boxes ?? [];

  return (
    <>
      <p className="label">RAG pipeline · retrieval (M2) · the answer column lands in M4</p>
      <h1>
        One question through every stage, <em>scored against the gold pages</em>
      </h1>
      <p className="lead">
        Pick a benchmark question or type your own to see every stage's top 10, scored against the gold pages.
      </p>
      <DatasetChips current={dataset} hrefFor={(d) => `/pipeline?dataset=${d}`} />
      <StatusLine s={status.data} dataset={dataset} />

      <QuestionPicker
        rows={bench.data}
        qid={qid}
        onPick={(id) => {
          const next = new URLSearchParams({ dataset });
          if (id) next.set('qid', id);
          if (sp.get('depth')) next.set('depth', sp.get('depth') as string);
          nav(`/pipeline?${next.toString()}`);
        }}
      />
      <form className="row askrow" onSubmit={runFree}>
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={`…or ask ${datasetLabel(dataset)} anything, e.g. “Wells Fargo CET1 ratio 2024”`}
          aria-label="your question"
        />
        <button className="btn" type="submit" disabled={!draft.trim()}>
          <span>Run</span>
        </button>
      </form>
      <div className="row stagetogs" role="group" aria-label="stages">
        {STAGES.map((s) => (
          <button
            key={s}
            type="button"
            className={`tog ${on[s] ? 'on' : ''}`}
            aria-pressed={on[s]}
            onClick={() => setOn({ ...on, [s]: !on[s] })}
          >
            <span>{STAGE_LABEL[s][0]}</span>
          </button>
        ))}
        <label className="pick small">
          <span>rerank the top</span>
          <select
            value={depth}
            onChange={(e) => {
              const next = new URLSearchParams(sp);
              next.set('depth', e.target.value);
              nav(`/pipeline?${next.toString()}`, { replace: true });
            }}
            aria-label="rerank depth"
          >
            {[10, 20, 50].map((d) => (
              <option key={d} value={d}>
                {d} pages (≈{Math.round(d * 3.2)} s the first time)
              </option>
            ))}
          </select>
        </label>
      </div>

      {qid && question.data && (
        <div className="qcard">
          <span className="label">
            benchmark question <Link to={`/datasets/${dataset}/${qid}`}>q{qid}</Link> ·{' '}
            {question.data.query_types.join(' · ')}
          </span>
          <p className="qbig">{question.data.query}</p>
          <p className="small">
            <b>Gold answer:</b> {question.data.answer}
          </p>
        </div>
      )}
      {busy != null && (
        <>
          <Busy
            label={`Running ${STAGES.filter((s) => on[s]).length} stages`}
            since={busy}
            detail={
              on.reranked
                ? `A new question takes up to ≈${Math.round(depth * 3.2)} s while the reranker reads its top ${depth} pages.`
                : undefined
            }
          />
          <SkeletonCols stages={STAGES.filter((s) => on[s])} />
        </>
      )}
      {err && <Loading error={err} />}

      {res && busy == null && (
        <>
          {res.gold.length > 0 && <GoldStrip k={dataset} res={res} stages={shown} onOpen={setOpen} />}
          {res.gold.length > 0 && <MetricsTable res={res} stages={shown} />}
          {res.gold.length === 0 && (
            <p className="small muted">Free-text questions have no gold pages, so judge the hits yourself.</p>
          )}
          <AnswerPanel dataset={dataset} qid={qid} qtext={qtext} res={res} stages={shown} />
          <div
            className="stagecols"
            style={{
              gridTemplateColumns: `repeat(${Math.max(1, shown.length)}, minmax(0, 1fr))`,
            }}
          >
            {shown.map((s) => (
              <StageColumn key={s} k={dataset} s={s} out={res.stages[s]!} open={open} onOpen={setOpen} />
            ))}
          </div>
          {Object.keys(res.deltas).length > 0 && <Deltas res={res} />}
          {open != null && (
            <section className="hitopen" aria-label="the open hit">
              <div className="row">
                <span className="label">hit · corpus {open}</span>
                <button type="button" className="linkish small" onClick={() => setOpen(null)}>
                  close
                </button>
              </div>
              <WhyRanked res={res} cid={open} />
              <PageView
                k={dataset}
                corpusId={open}
                boxes={goldBoxes(open)}
                grade={res.gold.length ? (res.gold.find((g) => g.corpus_id === open)?.grade ?? 0) : null}
              />
            </section>
          )}
        </>
      )}
    </>
  );
}

function StatusLine({ s, dataset }: { s: RetrievalStatus | null; dataset: string }) {
  if (!s) return null;
  const v = s.visual_stats;
  return (
    <p className="small statusline">
      <span className={`status ${s.text ? 'ok' : 'no'}`}>text index</span>{' '}
      <span className={`status ${s.visual_complete ? 'ok' : s.visual ? 'warn' : 'no'}`}>
        {s.visual_building ? 'building · ' : ''}
        visual index
        {v ? ` · ${v.model.split('/')[1]} · ${v.pages} pages · ${v.seconds_per_page} s/page on ${v.device}` : ''}
        {s.visual && !s.visual_complete ? ' (partial)' : ''}
      </span>{' '}
      <span className={`status ${s.reranker ? 'ok' : 'no'}`}>
        reranker {s.reranker ? 'ready' : 'not ready (zerank-2 not downloaded)'}
      </span>
      {!s.text && (
        <span className="muted">
          {' '}
          · build it: <code>make index DATASET={dataset} STAGE=text</code>
        </span>
      )}
    </p>
  );
}

const pct = (x: number) => `${(x * 100).toFixed(1)}`;
const yes = (b: boolean) => (b ? '● yes' : '○ no');
const ROWS: [string, (m: Metrics) => string, string][] = [
  ['NDCG@10', (m) => pct(m.ndcg_10), 'the board metric: gold near the top, full-answer pages double'],
  ['Recall@1', (m) => pct(m.recall_1), 'share of gold pages found in the top 1'],
  ['Recall@5', (m) => pct(m.recall_5), '… top 5: what a 5-page answer step can see'],
  ['Recall@10', (m) => pct(m.recall_10), '… top 10'],
  ['Recall@50', (m) => pct(m.recall_50), '… top 50: after fusion, the reranker’s ceiling'],
  ['Recall@100', (m) => pct(m.recall_100), '… top 100'],
  ['Complete@10', (m) => yes(m.complete_10), 'every gold page in the top 10'],
  ['Complete@50', (m) => yes(m.complete_50), 'every gold page in the top 50'],
  ['Full-answer hit@5', (m) => yes(m.hit_full_5), 'a grade-2 page in the top 5'],
  ['Full-answer hit@10', (m) => yes(m.hit_full_10), 'a grade-2 page in the top 10'],
  ['Precision@5', (m) => pct(m.precision_5), 'share of the top 5 that is gold'],
  ['Precision@10', (m) => pct(m.precision_10), 'share of the top 10 that is gold'],
  ['MRR', (m) => m.mrr.toFixed(3), '1 / rank of the first gold page'],
  ['First gold rank', (m) => (m.first_rank == null ? 'not found' : `#${m.first_rank}`), ''],
  ['First full-answer rank', (m) => (m.first_full_rank == null ? 'not found' : `#${m.first_full_rank}`), ''],
  ['MAP@10', (m) => m.map_10.toFixed(3), 'precision at every gold page’s position'],
];

function MetricsTable({ res, stages }: { res: SearchOut; stages: Stage[] }) {
  const g = res.gold;
  const full = g.filter((x) => x.grade === 2).length;
  return (
    <>
      <h2>
        Scored against {g.length} gold {g.length === 1 ? 'page' : 'pages'} ({full} full answer, {g.length - full} part)
      </h2>
      <div className="tw">
        <table className="metrics">
          <thead>
            <tr>
              <th>metric</th>
              {stages.map((s) => (
                <th key={s} className="num">
                  {STAGE_LABEL[s][0]}
                </th>
              ))}
              <th>what it measures</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map(([name, fmt, what]) => (
              <tr key={name} className={name === 'NDCG@10' ? 'cur' : ''}>
                <td className="sub nw">{name}</td>
                {stages.map((s) => {
                  const m = res.stages[s]?.metrics;
                  return (
                    <td key={s} className="num">
                      {m ? fmt(m) : '—'}
                    </td>
                  );
                })}
                <td className="small muted">{what}</td>
              </tr>
            ))}
            <tr>
              <td className="sub nw">Latency</td>
              {stages.map((s) => (
                <td key={s} className="num">
                  {res.stages[s]?.ms != null ? `${res.stages[s]!.ms} ms` : '—'}
                </td>
              ))}
              <td className="small muted">this stage’s own time for this question</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="small muted">Percentages are of this question’s gold pages, computed as in the Runs tab.</p>
    </>
  );
}

function StageColumn({
  k,
  s,
  out,
  open,
  onOpen,
}: {
  k: string;
  s: Stage;
  out: NonNullable<SearchOut['stages'][Stage]>;
  open: number | null;
  onOpen: (cid: number) => void;
}) {
  return (
    <section className="stagecol" aria-label={STAGE_LABEL[s][0]}>
      <header>
        <b>{STAGE_LABEL[s][0]}</b>
        <span className="small muted">{STAGE_LABEL[s][1]}</span>
        {out.weights && (
          <span className="small muted">
            weights{' '}
            {Object.entries(out.weights)
              .map(([c, w]) => `${c} ${w}`)
              .join(' · ')}
            {out.weights.text === 0 && ' (text off: it lowered NDCG@10 on train)'}
          </span>
        )}
        {out.ms != null && (
          <span className="mono small">{out.ms >= 1000 ? `${(out.ms / 1000).toFixed(1)} s` : `${out.ms} ms`}</span>
        )}
        {out.depth != null && (
          <span className="small muted">
            top {out.depth} reranked · {out.scored} scored now, {out.cached} from cache
          </span>
        )}
      </header>
      {out.unavailable ? (
        <p className="empty small">{out.unavailable}</p>
      ) : (
        <>
          {out.note && <p className="small warn-note">◐ {out.note}</p>}
          <ol className="hits">
            {(out.hits ?? []).map((h) => (
              <li key={h.corpus_id}>
                <HitCard k={k} h={h} on={open === h.corpus_id} onOpen={() => onOpen(h.corpus_id)} />
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  );
}

function HitCard({ k, h, on, onOpen }: { k: string; h: Hit; on: boolean; onOpen: () => void }) {
  const g = h.grade;
  return (
    <button type="button" className={`hit ${on ? 'on' : ''} ${g === 2 ? 'g2' : g === 1 ? 'g1' : ''}`} onClick={onOpen}>
      <span className="hrank mono">#{h.rank}</span>
      <img src={pageImage(k, h.corpus_id)} alt="" loading="lazy" />
      <span className="hmeta">
        <span className="mono small">
          {h.doc_id} p.{h.page}
        </span>
        {g != null && (
          <span className={`chip ${g === 2 ? 'ok' : g === 1 ? 'warn' : 'no'}`}>
            {g === 2 ? '● full · 2' : g === 1 ? '◐ part · 1' : '○ not gold'}
          </span>
        )}
        {h.terms && h.terms.length > 0 && (
          <span className="small muted">matched: {h.terms.slice(0, 6).join(', ')}</span>
        )}
      </span>
    </button>
  );
}

function Deltas({ res }: { res: SearchOut }) {
  return (
    <>
      <h3>What each stage changed in the top 10</h3>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>stage</th>
              <th>gold pages gained</th>
              <th>gold pages lost</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(res.deltas).map(([k, d]) => (
              <tr key={k}>
                <td className="sub nw">{k}</td>
                <td className="mono">{d.gained.length ? d.gained.join(', ') : '—'}</td>
                <td className="mono">{d.lost.length ? d.lost.join(', ') : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function WhyRanked({ res, cid }: { res: SearchOut; cid: number }) {
  const hit = STAGES.map((s) => res.stages[s]?.hits?.find((h) => h.corpus_id === cid)).find(Boolean);
  const gold = res.gold.find((g) => g.corpus_id === cid);
  return (
    <dl className="why">
      <dt>rank in each stage</dt>
      <dd className="mono">
        {STAGES.filter((s) => res.stages[s] && !res.stages[s]!.unavailable)
          .map((s) => {
            const r = hit?.ranks?.[s] ?? res.stages[s]?.gold_ranks?.[String(cid)];
            return `${STAGE_LABEL[s][0]}: ${r == null ? `not in top ${res.stages[s]?.n ?? '—'}` : `#${r}`}`;
          })
          .join('  ·  ')}
      </dd>
      <dt>score in each stage</dt>
      <dd className="mono">
        {STAGES.filter((s) => res.stages[s] && !res.stages[s]!.unavailable)
          .map((s) => {
            const v = hit?.scores?.[s];
            return `${STAGE_LABEL[s][0]}: ${v == null ? '—' : v.toFixed(s === 'fused' ? 4 : 2)}`;
          })
          .join('  ·  ')}
        <span className="small muted">
          {' '}
          (A: BM25 · B: visual MaxSim, summed over the question's tokens · C: RRF · D: zerank-2; — = not scored there)
        </span>
      </dd>
      {hit?.terms && (
        <>
          <dt>question words found on the page (BM25)</dt>
          <dd>{hit.terms.length ? hit.terms.join(', ') : 'none: it ranked on the visual channel alone'}</dd>
        </>
      )}
      <dt>gold</dt>
      <dd>{res.gold.length === 0 ? 'free-text question: no gold pages' : gold ? `yes, grade ${gold.grade}` : 'no'}</dd>
    </dl>
  );
}

/** Every gold page, and where each stage put it: coverage at a glance, including pages no stage found. */
function GoldStrip({
  k,
  res,
  stages,
  onOpen,
}: {
  k: string;
  res: SearchOut;
  stages: Stage[];
  onOpen: (cid: number) => void;
}) {
  return (
    <>
      <h3>The gold pages, and where each stage ranked them</h3>
      <div className="goldstrip">
        {res.gold.map((g) => (
          <button key={g.corpus_id} type="button" className={`gold g${g.grade}`} onClick={() => onOpen(g.corpus_id)}>
            <img src={pageImage(k, g.corpus_id)} alt="" loading="lazy" />
            <span className={`chip ${g.grade === 2 ? 'ok' : 'warn'}`}>
              {g.grade === 2 ? '● full · 2' : '◐ part · 1'}
            </span>
            <span className="granks mono small">
              {stages
                .filter((s) => res.stages[s]?.gold_ranks)
                .map((s) => {
                  const r = res.stages[s]!.gold_ranks![String(g.corpus_id)];
                  return (
                    <span key={s} className={r != null && r <= 10 ? 'in' : ''}>
                      {STAGE_LABEL[s][0].split(' · ')[0]} {r == null ? '—' : `#${r}`}
                    </span>
                  );
                })}
            </span>
          </button>
        ))}
      </div>
    </>
  );
}

/** Every benchmark question of the dataset, grouped by its set of query types. */
function QuestionPicker({
  rows,
  qid,
  onPick,
}: {
  rows: QueryRow[] | null;
  qid: string | null;
  onPick: (qid: string | null) => void;
}) {
  const groups = new Map<string, QueryRow[]>();
  for (const r of rows ?? []) {
    const k = [...r.query_types].sort().join(' + ') || 'untyped';
    groups.set(k, [...(groups.get(k) ?? []), r]);
  }
  const clip = (t: string) => (t.length > 110 ? `${t.slice(0, 107)}…` : t);
  const pickRandom = () => {
    if (rows?.length) onPick(String(rows[Math.floor(Math.random() * rows.length)].query_id));
  };
  return (
    <div className="row askrow pickrow">
      <select
        value={qid ?? ''}
        onChange={(e) => onPick(e.target.value || null)}
        disabled={!rows}
        aria-label="benchmark question"
      >
        <option value="">
          {rows ? `Pick one of the ${rows.length} benchmark questions…` : 'Loading the questions…'}
        </option>
        {[...groups.entries()]
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([k, qs]) => (
            <optgroup key={k} label={`${k} (${qs.length})`}>
              {qs.map((q) => (
                <option key={q.query_id} value={q.query_id}>
                  q{q.query_id} · {clip(q.query)}
                  {q.n_full ? '' : ' · no full-answer page'}
                </option>
              ))}
            </optgroup>
          ))}
      </select>
      <button className="tog" type="button" onClick={pickRandom} disabled={!rows?.length}>
        <span>Random</span>
      </button>
    </div>
  );
}

/** Placeholder stage columns that pulse while a search runs. */
function SkeletonCols({ stages }: { stages: Stage[] }) {
  return (
    <div
      className="stagecols"
      aria-hidden="true"
      style={{
        gridTemplateColumns: `repeat(${Math.max(1, stages.length)}, minmax(0, 1fr))`,
      }}
    >
      {stages.map((s) => (
        <section key={s} className="stagecol skel">
          <header>
            <b>
              <span className="busy-dot" />
              {STAGE_LABEL[s][0]}
            </b>
          </header>
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="skel-hit" />
          ))}
        </section>
      ))}
    </div>
  );
}

/** The end of the pipeline: Claude answers from the last stage's top 5, on request (it uses the subscription). */
function AnswerPanel({
  dataset,
  qid,
  qtext,
  res,
  stages,
}: {
  dataset: string;
  qid: string | null;
  qtext: string;
  res: SearchOut;
  stages: Stage[];
}) {
  const health = useGet<Health>('/api/health');
  const [ans, setAns] = useState<LiveAnswer | null>(null);
  const [since, setSince] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const last = stages[stages.length - 1];
  const top = (last ? (res.stages[last]?.hits ?? []) : []).slice(0, 5).map((h) => h.corpus_id);
  const key = `${dataset}|${qid}|${qtext}|${last}|${top.join(',')}`;
  useEffect(() => {
    setAns(null);
    setErr(null);
    setSince(null);
  }, [key]);
  if (!health.data?.claude.available || !last || top.length === 0) return null;
  const go = () => {
    setSince(Date.now());
    setErr(null);
    post<LiveAnswer>('/api/answer', {
      dataset,
      query_id: qid ? Number(qid) : null,
      query: qid ? null : qtext,
      corpus_ids: top,
    })
      .then(setAns)
      .catch((e: Error) => setErr(e.message))
      .finally(() => setSince(null));
  };
  return (
    <section aria-label="the answer">
      <h2>Answer</h2>
      {!ans && since == null && (
        <div className="row">
          <button type="button" className="btn" onClick={go}>
            <span>
              Ask Claude to answer from the top {top.length} of {STAGE_LABEL[last][0]}
            </span>
          </button>
          <span className="small muted">One call on the subscription, about 25k tokens.</span>
        </div>
      )}
      {since != null && <Busy label="Claude is reading the pages" since={since} />}
      {err && <p className="small warn-note">{err}</p>}
      {ans && (
        <div className="answercard">
          <span className="label">
            {ans.usage.model} · {ans.usage.input_tokens.toLocaleString('en-AU')} tokens ·{' '}
            {(ans.usage.ms / 1000).toFixed(1)} s{ans.judge ? '' : ' · free text, so not judged'}
          </span>
          <AnswerText text={ans.answer} />
          {ans.judge && (
            <>
              <VerdictChip v={ans.judge.verdict} />
              <p className="small">
                <b>Gold answer:</b> {ans.reference}
              </p>
              <p className="small muted">
                <b>Judge:</b> {ans.judge.explanation}
              </p>
            </>
          )}
        </div>
      )}
    </section>
  );
}
