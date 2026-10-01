import { useEffect, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import {
  type Hit,
  type Metrics,
  type Question,
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
import { DatasetChips, Loading } from '../lib/ui';

/**
 * `/pipeline?dataset=&qid=` or `?q=` — one question through the retrieval half of the RAG
 * pipeline (M2): every stage's top 10, scored against the gold pages when the question has
 * them. M4 adds the answer column. The URL holds the question, so a run is a link.
 */
export function Pipeline() {
  const [sp] = useSearchParams();
  const nav = useNavigate();
  const dataset = sp.get('dataset') ?? 'finance_en';
  const qid = sp.get('qid');
  const qtext = sp.get('q') ?? '';
  const [draft, setDraft] = useState(qtext);
  const [on, setOn] = useState<Record<Stage, boolean>>({ text: true, visual: true, fused: true, reranked: true });
  const [res, setRes] = useState<SearchOut | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState<number | null>(null);
  const status = useGet<RetrievalStatus>(`/api/retrieval/status?dataset=${dataset}`);
  const question = useGet<Question>(qid ? `/api/datasets/${dataset}/queries/${qid}` : null);

  useEffect(() => setDraft(qtext), [qtext]);
  useEffect(() => {
    if (!qid && !qtext) {
      setRes(null);
      return;
    }
    let alive = true;
    setBusy(true);
    setErr(null);
    setOpen(null);
    post<SearchOut>('/api/search', {
      dataset,
      query_id: qid ? Number(qid) : null,
      query: qid ? null : qtext,
      stages: STAGES.filter((s) => on[s]),
    })
      .then((r) => alive && setRes(r))
      .catch((e: Error) => alive && setErr(e.message))
      .finally(() => alive && setBusy(false));
    return () => {
      alive = false;
    };
  }, [dataset, qid, qtext, on]);

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
        Pick a question on the Datasets tab and press <b>Run retrieval</b>, or type your own here. Each stage returns its
        top 10. For a benchmark question, the human grades show which hits are right, and every metric is computed
        against the gold pages.
      </p>
      <DatasetChips current={dataset} hrefFor={(d) => `/pipeline?dataset=${d}`} />
      <StatusLine s={status.data} dataset={dataset} />

      <form className="row askrow" onSubmit={runFree}>
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={`Ask ${datasetLabel(dataset)} anything, e.g. “Wells Fargo CET1 ratio 2024”`}
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
      </div>

      {qid && question.data && (
        <div className="qcard">
          <span className="label">
            benchmark question <Link to={`/datasets/${dataset}/${qid}`}>q{qid}</Link> · {question.data.query_types.join(' · ')}
          </span>
          <p className="qbig">{question.data.query}</p>
          <p className="small">
            <b>Reference answer:</b> {question.data.answer}
          </p>
        </div>
      )}
      {busy && <p className="empty">Running… (the first query loads the models: up to a minute)</p>}
      {err && <Loading error={err} />}

      {res && !busy && (
        <>
          {res.gold.length > 0 && <GoldStrip k={dataset} res={res} stages={shown} onOpen={setOpen} />}
          {res.gold.length > 0 && <MetricsTable res={res} stages={shown} />}
          {res.gold.length === 0 && (
            <p className="small muted">A free-text question has no gold pages, so no metrics: judge the hits yourself.</p>
          )}
          <div className="stagecols" style={{ gridTemplateColumns: `repeat(${Math.max(1, shown.length)}, minmax(0, 1fr))` }}>
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
        visual index{v ? ` · ${v.pages} pages · ${v.seconds_per_page} s/page on ${v.device}` : ''}
        {s.visual && !s.visual_complete ? ' (partial)' : ''}
      </span>{' '}
      <span className={`status ${s.reranker ? 'ok' : 'no'}`}>reranker {s.reranker ? 'ready' : 'not ready (zerank-2 not downloaded)'}</span>
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
      <p className="small muted">
        Percentages are of this question’s gold pages. One function (<code>eval/metrics.py</code>, checked against{' '}
        <code>pytrec_eval</code>) computes these here and in M3’s runs.
      </p>
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
        {out.ms != null && <span className="mono small">{out.ms} ms</span>}
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
        {h.terms && h.terms.length > 0 && <span className="small muted">matched: {h.terms.slice(0, 6).join(', ')}</span>}
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
function GoldStrip({ k, res, stages, onOpen }: { k: string; res: SearchOut; stages: Stage[]; onOpen: (cid: number) => void }) {
  return (
    <>
      <h3>The gold pages, and where each stage ranked them</h3>
      <div className="goldstrip">
        {res.gold.map((g) => (
          <button key={g.corpus_id} type="button" className={`gold g${g.grade}`} onClick={() => onOpen(g.corpus_id)}>
            <img src={pageImage(k, g.corpus_id)} alt="" loading="lazy" />
            <span className={`chip ${g.grade === 2 ? 'ok' : 'warn'}`}>{g.grade === 2 ? '● full · 2' : '◐ part · 1'}</span>
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
