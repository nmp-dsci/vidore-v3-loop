import { useMemo, useState } from 'react';
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import {
  type Mean,
  type Metrics,
  type RankStat,
  type RunDetail,
  type RunMeta,
  type RunQuery,
  type RunQueryRow,
  type SplitInfo,
  type Stage,
  type StageSummary,
  STAGES,
  STAGE_LABEL,
  pageImage,
  pct1,
  useGet,
} from '../lib/api';
import { Loading } from '../lib/ui';

/** Every metric a run averages, in the order the RAG pipeline tab lists them. */
const ROWS: [string, string, 'mean' | 'rank', string][] = [
  ['ndcg_10', 'NDCG@10', 'mean', 'the board metric: gold near the top, full-answer pages double'],
  ['recall_1', 'Recall@1', 'mean', 'share of gold pages in the top 1'],
  ['recall_5', 'Recall@5', 'mean', '… top 5: what a 5-page answer step can see'],
  ['recall_10', 'Recall@10', 'mean', '… top 10'],
  ['recall_50', 'Recall@50', 'mean', '… top 50: the reranker’s ceiling'],
  ['recall_100', 'Recall@100', 'mean', '… top 100'],
  ['complete_10', 'Complete@10', 'mean', 'share of questions with every gold page in the top 10'],
  ['complete_50', 'Complete@50', 'mean', '… in the top 50'],
  ['hit_full_5', 'Full-answer hit@5', 'mean', 'share of questions with a grade-2 page in the top 5'],
  ['hit_full_10', 'Full-answer hit@10', 'mean', '… in the top 10'],
  ['precision_5', 'Precision@5', 'mean', 'share of the top 5 that is gold'],
  ['precision_10', 'Precision@10', 'mean', 'share of the top 10 that is gold'],
  ['mrr', 'MRR', 'mean', 'mean of 1 / rank of the first gold page'],
  ['map_10', 'MAP@10', 'mean', 'precision at every gold page’s position'],
  ['first_rank', 'First gold rank', 'rank', 'median over the questions where it was found'],
  ['first_full_rank', 'First full-answer rank', 'rank', 'median, questions with a grade-2 page'],
];
const SLICE_METRICS: [string, string][] = [
  ['ndcg_10', 'NDCG@10'],
  ['recall_10', 'Recall@10'],
  ['recall_50', 'Recall@50'],
  ['hit_full_10', 'Full-answer hit@10'],
  ['complete_50', 'Complete@50'],
  ['mrr', 'MRR'],
];

const stagesOf = (r: RunMeta): Stage[] => STAGES.filter((s) => r.config.stages.includes(s));
const short = (s: Stage) => STAGE_LABEL[s][0].split(' · ')[0];
const when = (iso: string) => iso.replace('T', ' ').slice(0, 16);
const meanOf = (st: StageSummary | undefined, k: string) => (st?.[k] as Mean | undefined)?.mean ?? null;
const fmtMetric = (k: string, v: number | null) =>
  v == null ? '—' : k === 'mrr' || k === 'map_10' ? v.toFixed(3) : pct1(v);

function StatusChip({ r }: { r: RunMeta }) {
  if (r.status === 'running')
    return (
      <span className="status warn pulsing">
        running {r.progress.done}/{r.progress.n}
      </span>
    );
  if (r.status === 'finished') return <span className="status ok">finished</span>;
  return (
    <span className="status err" title={r.error ?? ''}>
      {r.status}
    </span>
  );
}

function SplitChip({ r }: { r: RunMeta }) {
  return r.split === 'test' ? (
    <span className="chip warn" title="scored once, for a configuration chosen on train">
      test · final
    </span>
  ) : (
    <span className="chip">train</span>
  );
}

function configText(r: RunMeta) {
  const [wt, wv] = r.config.weights;
  const parts = [`fusion text ${wt} : visual ${wv}`];
  if (r.config.stages.includes('reranked')) parts.push(`rerank top ${r.config.rerank_depth}`);
  else parts.push('no reranker');
  if (r.config.limit) parts.push(`first ${r.config.limit} only`);
  return parts.join(' · ');
}

/** `/runs` — every scored run, and the split they are scored on. */
export function Runs() {
  const { data: runs, error } = useGet<RunMeta[]>('/api/runs');
  const { data: split } = useGet<SplitInfo>('/api/split/finance_en');
  if (!runs) return <Loading error={error} what="the runs" />;
  return (
    <>
      <p className="label">Runs · retrieval scored over our split (M3)</p>
      <h1>
        Every configuration, scored on <em>155 train questions</em>; the chosen one, once, on test
      </h1>
      <p className="lead">
        Each run scores a whole split; configurations are compared on train and only the chosen one is scored on test,
        once.
      </p>
      {split && <SplitPanel s={split} />}
      <p className="small rule">
        <b>Selection rule</b> (fixed in advance): the highest train NDCG@10 is scored on test, once.
      </p>
      <p className="small rule">
        <b>Finance outcome:</b> visual alone won on train (75.8 vs 75.4 with the reranker) and scored 70.0 on test.
      </p>

      <h2>Runs, newest first</h2>
      {runs.length === 0 ? (
        <p className="empty">
          No runs yet: start one with <code>make eval DATASET=finance_en</code>.
        </p>
      ) : (
        <div className="tw">
          <table className="runs fit">
            <thead>
              <tr>
                <th rowSpan={2}>run</th>
                <th rowSpan={2}>split</th>
                <th rowSpan={2} className="num">
                  questions
                </th>
                <th rowSpan={2}>configuration</th>
                <th colSpan={STAGES.length} className="group">
                  NDCG@10 per stage
                </th>
                <th rowSpan={2}>status</th>
              </tr>
              <tr>
                {STAGES.map((s) => (
                  <th key={s} className="num" title={`NDCG@10, ${STAGE_LABEL[s][1]}`}>
                    {short(s)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => {
                const vals = STAGES.map((s) => meanOf(r.summary?.[s], 'ndcg_10'));
                const best = Math.max(...vals.map((v) => v ?? -1));
                return (
                  <tr key={r.run_id}>
                    <td className="runcell">
                      <Link className="nw" to={`/runs/${r.run_id}`}>
                        {when(r.started_at)}
                      </Link>
                      {r.config.note && <div className="small muted">{r.config.note}</div>}
                    </td>
                    <td>
                      <SplitChip r={r} />
                    </td>
                    <td className="num">{r.n}</td>
                    <td className="small cfgcell">{configText(r)}</td>
                    {vals.map((v, i) => (
                      <td key={STAGES[i]} className={`num ${v != null && v === best ? 'best' : ''}`}>
                        {v == null ? '—' : pct1(v)}
                      </td>
                    ))}
                    <td>
                      <StatusChip r={r} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="small muted">
        A text (BM25) · B visual (EVIE) · C fused (RRF) · D reranked (zerank-2), NDCG@10 × 100.
      </p>
    </>
  );
}

function SplitPanel({ s }: { s: SplitInfo }) {
  return (
    <details className="splitpanel">
      <summary>
        The split: <b>{s.n.train}</b> train · <b>{s.n.test}</b> test of {s.dataset}'s {s.n.train + s.n.test} English
        questions · seed {s.seed} · stratified by {s.stratified_by} ({s.strata.length} groups) · file {s.fingerprint}
      </summary>
      <div className="tw">
        <table className="fit">
          <thead>
            <tr>
              <th>query types</th>
              <th className="num">train</th>
              <th className="num">test</th>
            </tr>
          </thead>
          <tbody>
            {s.strata.map((g) => (
              <tr key={g.stratum}>
                <td>{g.stratum.replaceAll('+', ' + ')}</td>
                <td className="num">{g.train}</td>
                <td className="num">{g.test}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

/** `/runs/:runId` — one run: its configuration, every metric per stage, the slices, every question. */
export function Run() {
  const { runId = '' } = useParams();
  const { data: r, error } = useGet<RunDetail>(`/api/runs/${runId}`);
  const { data: qs } = useGet<RunQueryRow[]>(r?.status === 'finished' ? `/api/runs/${runId}/queries` : null);
  if (!r) return <Loading error={error} what="the run" />;
  const stages = stagesOf(r);
  return (
    <>
      <p className="label">
        <Link to="/runs">Runs</Link> / {r.run_id}
      </p>
      <h1>
        {r.dataset} · {r.split} · <em>{configText(r)}</em>
      </h1>
      <div className="row">
        <SplitChip r={r} />
        <StatusChip r={r} />
        <span className="small muted">
          {r.n} questions · started {when(r.started_at)}
          {r.seconds != null && ` · ${Math.round(r.seconds / 60)} min`} · code {r.code_sha} · config {r.fingerprint}
        </span>
      </div>
      {r.config.note && <p className="small">{r.config.note}</p>}
      {r.error && <p className="empty">Failed: {r.error}</p>}
      <dl className="kv small runkv">
        <dt>visual model</dt>
        <dd className="mono">
          {r.models.visual} ({r.models.visual_dim}-d)
        </dd>
        <dt>reranker</dt>
        <dd className="mono">{r.models.reranker ?? '—'}</dd>
        <dt>pools</dt>
        <dd>
          each channel's top {r.channel_k} → fused top {r.fused_k} (RRF k = {r.rrf_k})
        </dd>
        <dt>split</dt>
        <dd>
          seed {r.split_seed} · file {r.split_fingerprint}
        </dd>
        <dt>MLflow</dt>
        <dd className="mono">
          {r.mlflow_run_id ? `vidore-v3-loop/evals · ${r.mlflow_run_id}` : (r.mlflow_error ?? 'not logged yet')}
        </dd>
      </dl>
      {r.summary && <SummaryTable r={r} stages={stages} />}
      {r.costs && <Costs r={r} stages={stages} />}
      {r.slices && <Slices slices={r.slices} stages={stages} />}
      {qs && <QueryTable r={r} rows={qs} stages={stages} />}
    </>
  );
}

function SummaryTable({ r, stages }: { r: RunMeta; stages: Stage[] }) {
  return (
    <>
      <h2>Every metric, per stage</h2>
      <div className="tw">
        <table className="metrics fit">
          <thead>
            <tr>
              <th>metric</th>
              {stages.map((s) => (
                <th key={s} className="num">
                  {STAGE_LABEL[s][0]}
                </th>
              ))}
              <th className="num">questions</th>
              <th>what it measures</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map(([k, name, kind, what]) => {
              const any = r.summary?.[stages[0]]?.[k];
              return (
                <tr key={k} className={k === 'ndcg_10' ? 'cur' : ''}>
                  <td className="sub">{name}</td>
                  {stages.map((s) => {
                    const v = r.summary?.[s]?.[k];
                    if (!v)
                      return (
                        <td key={s} className="num">
                          —
                        </td>
                      );
                    if (kind === 'rank') {
                      const rs = v as RankStat;
                      return (
                        <td key={s} className="num" title={`found for ${rs.found} of ${rs.n} questions`}>
                          {rs.median == null ? '—' : `#${rs.median}`}
                          <span className="muted small rankn">
                            ({rs.found}/{rs.n})
                          </span>
                        </td>
                      );
                    }
                    return (
                      <td key={s} className="num">
                        {fmtMetric(k, (v as Mean).mean)}
                      </td>
                    );
                  })}
                  <td className="num muted">{any ? (any as Mean).n : '—'}</td>
                  <td className="small muted">{what}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted">Means over the questions counted; ranks show the median and (found/total).</p>
    </>
  );
}

function Costs({ r, stages }: { r: RunMeta; stages: Stage[] }) {
  return (
    <p className="small muted">
      Time per question:{' '}
      {stages
        .map((s) => {
          const c = r.costs?.[s];
          if (!c || c.mean_ms == null) return `${short(s)} —`;
          const t = c.mean_ms >= 1000 ? `${(c.mean_ms / 1000).toFixed(1)} s` : `${c.mean_ms} ms`;
          return s === 'reranked'
            ? `${short(s)} ${t} (${c.pages_scored} pages scored, ${c.pages_cached} from cache)`
            : `${short(s)} ${t}`;
        })
        .join(' · ')}
    </p>
  );
}

function Slices({
  slices,
  stages,
}: {
  slices: Record<string, { value: string; n: number; stages: Partial<Record<Stage, StageSummary>> }[]>;
  stages: Stage[];
}) {
  const dims = Object.keys(slices);
  const [dim, setDim] = useState(dims[0]);
  const [metric, setMetric] = useState('ndcg_10');
  const rows = slices[dim] ?? [];
  return (
    <>
      <h2>Sliced</h2>
      <div className="row">
        <div className="seg" role="group" aria-label="slice by">
          {dims.map((d) => (
            <button
              key={d}
              type="button"
              className={`tog ${d === dim ? 'on' : ''}`}
              aria-pressed={d === dim}
              onClick={() => setDim(d)}
            >
              <span>{d}</span>
            </button>
          ))}
        </div>
        <label className="pick small">
          <span>metric</span>
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            {SLICE_METRICS.map(([k, n]) => (
              <option key={k} value={k}>
                {n}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="tw">
        <table className="fit">
          <thead>
            <tr>
              <th>{dim}</th>
              <th className="num">questions</th>
              {stages.map((s) => (
                <th key={s} className="num">
                  {STAGE_LABEL[s][0]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const vals = stages.map((s) => meanOf(row.stages[s], metric));
              const best = Math.max(...vals.map((v) => v ?? -1));
              return (
                <tr key={row.value}>
                  <td>{row.value}</td>
                  <td className="num">{row.n}</td>
                  {vals.map((v, i) => (
                    <td key={stages[i]} className={`num ${v != null && v === best ? 'best' : ''}`}>
                      {fmtMetric(metric, v)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        A question with two types counts in both rows, so small groups move a lot.
      </p>
    </>
  );
}

type SortKey = 'id' | Stage;

function QueryTable({ r, rows, stages }: { r: RunMeta; rows: RunQueryRow[]; stages: Stage[] }) {
  const [sp, setSp] = useSearchParams();
  const nav = useNavigate();
  const last = stages[stages.length - 1];
  const sort = (sp.get('sort') as SortKey | null) ?? last;
  const filter = sp.get('f') ?? '';
  const shown = useMemo(() => {
    const f = filter.trim().toLowerCase();
    const out = rows.filter(
      (q) =>
        !f ||
        String(q.query_id) === f ||
        q.query.toLowerCase().includes(f) ||
        q.query_types.some((t) => t.includes(f)) ||
        q.content_type.some((t) => t.toLowerCase().includes(f)),
    );
    return out.sort((a, b) =>
      sort === 'id'
        ? a.query_id - b.query_id
        : (a.stages[sort]?.ndcg_10 ?? 0) - (b.stages[sort]?.ndcg_10 ?? 0) || a.query_id - b.query_id,
    );
  }, [rows, sort, filter]);
  const set = (k: string, v: string) => {
    const next = new URLSearchParams(sp);
    if (v) next.set(k, v);
    else next.delete(k);
    setSp(next, { replace: true });
  };
  return (
    <>
      <h2>Every question ({rows.length})</h2>
      <div className="row">
        <input
          type="text"
          className="grow"
          value={filter}
          placeholder="filter: words, a query id, a type (multi-hop), a content type (chart)"
          onChange={(e) => set('f', e.target.value)}
          aria-label="filter questions"
        />
        <span className="small muted">
          {shown.length} shown · sorted by{' '}
          {sort === 'id' ? 'query id' : `${STAGE_LABEL[sort as Stage][0]} NDCG@10, worst first`}
        </span>
      </div>
      <div className="tw">
        <table className="qtable fit">
          <thead>
            <tr>
              <th>
                <button type="button" className="linkish" onClick={() => set('sort', 'id')}>
                  q
                </button>
              </th>
              <th>question</th>
              <th className="num">gold</th>
              {stages.map((s) => (
                <th key={s} className="num">
                  <button type="button" className="linkish" onClick={() => set('sort', s)} title="sort worst first">
                    {short(s)} NDCG
                  </button>
                </th>
              ))}
              <th className="num" title={`first full-answer page, ${STAGE_LABEL[last][0]}`}>
                {short(last)} full at
              </th>
            </tr>
          </thead>
          <tbody>
            {shown.map((q) => (
              <tr key={q.query_id} className="click" onClick={() => nav(`/runs/${r.run_id}/${q.query_id}`)}>
                <td className="mono">
                  <Link to={`/runs/${r.run_id}/${q.query_id}`}>q{q.query_id}</Link>
                </td>
                <td className="wrap">
                  {q.query}
                  <div className="small muted">
                    {q.query_types.join(' · ')} · {q.content_type.join(', ')} · {q.author}
                  </div>
                </td>
                <td className="num">
                  {q.n_gold}
                  <span className="muted small"> ({q.n_full} full)</span>
                </td>
                {stages.map((s) => {
                  const v = q.stages[s]?.ndcg_10;
                  return (
                    <td key={s} className={`num ${v == null ? '' : v >= 0.7 ? 'good' : v < 0.3 ? 'poor' : ''}`}>
                      {v == null ? '—' : pct1(v)}
                    </td>
                  );
                })}
                <td className="num">
                  {q.n_full === 0 ? (
                    <span className="muted">no full page</span>
                  ) : q.stages[last]?.first_full_rank ? (
                    `#${q.stages[last]?.first_full_rank}`
                  ) : (
                    <span className="poor">not in top 100</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

/** `/runs/:runId/:qid` — one question's stored rankings in a run, against its gold pages. */
export function RunQuestion() {
  const { runId = '', qid = '' } = useParams();
  const { data: q, error } = useGet<RunQuery>(`/api/runs/${runId}/queries/${qid}`);
  const { data: r } = useGet<RunMeta>(`/api/runs/${runId}`);
  if (!q || !r) return <Loading error={error} what="the question" />;
  const stages = STAGES.filter((s) => q.stages[s]);
  const depth = r.config.rerank_depth;
  return (
    <>
      <p className="label">
        <Link to="/runs">Runs</Link> / <Link to={`/runs/${runId}`}>{runId}</Link> / q{q.query_id}
      </p>
      <h1 className="qh1">{q.query}</h1>
      <div className="row">
        <span className="small muted">
          {q.query_types.join(' · ')} · {q.content_type.join(', ')} · {q.query_format} ·{' '}
          {q.query_generator === 'human' ? 'written by a person' : 'synthetic'}
        </span>
        <Link className="btn" to={`/pipeline?dataset=${q.dataset}&qid=${q.query_id}&depth=${depth}`}>
          <span>Run it live in the RAG pipeline →</span>
        </Link>
        <Link to={`/datasets/${q.dataset}/${q.query_id}`} className="small">
          the question and its evidence →
        </Link>
      </div>
      <p className="small muted">
        Rankings stored by this run ({configText(r)}); the live tab may differ if the settings have changed.
      </p>

      <h3>The gold pages, and where each stage ranked them</h3>
      <div className="goldstrip">
        {q.gold.map((g) => (
          <div key={g.corpus_id} className={`gold g${g.grade}`}>
            <img src={pageImage(q.dataset, g.corpus_id)} alt="" loading="lazy" />
            <span className={`chip ${g.grade === 2 ? 'ok' : 'warn'}`}>
              {g.grade === 2 ? '● full · 2' : '◐ part · 1'}
            </span>
            <span className="granks mono small">
              {stages.map((s) => {
                const rk = q.stages[s]?.gold_ranks[String(g.corpus_id)];
                return (
                  <span key={s} className={rk != null && rk <= 10 ? 'in' : ''}>
                    {short(s)} {rk == null ? '—' : `#${rk}`}
                  </span>
                );
              })}
            </span>
            <span className="small muted">
              {g.doc_id} p.{g.page}
            </span>
          </div>
        ))}
      </div>

      <div className="tw">
        <table className="metrics fit">
          <thead>
            <tr>
              <th>metric</th>
              {stages.map((s) => (
                <th key={s} className="num">
                  {STAGE_LABEL[s][0]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(
              [
                ['NDCG@10', (m) => pct1(m.ndcg_10)],
                ['Recall@10', (m) => pct1(m.recall_10)],
                ['Recall@50', (m) => pct1(m.recall_50)],
                ['First full-answer rank', (m) => (m.first_full_rank == null ? '—' : `#${m.first_full_rank}`)],
                ['MRR', (m) => m.mrr.toFixed(3)],
              ] as [string, (m: Metrics) => string][]
            ).map(([name, fmt]) => (
              <tr key={name} className={name === 'NDCG@10' ? 'cur' : ''}>
                <td className="sub nw">{name}</td>
                {stages.map((s) => (
                  <td key={s} className="num">
                    {fmt(q.stages[s]!.metrics)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div
        className="stagecols"
        style={{
          gridTemplateColumns: `repeat(${Math.max(1, stages.length)}, minmax(0, 1fr))`,
        }}
      >
        {stages.map((s) => (
          <section key={s} className="stagecol" aria-label={STAGE_LABEL[s][0]}>
            <header>
              <b>{STAGE_LABEL[s][0]}</b>
              <span className="small muted">{STAGE_LABEL[s][1]}</span>
            </header>
            <ol className="hits">
              {q.stages[s]!.top.map((h) => (
                <li key={h.corpus_id}>
                  <div className={`hit ${h.grade === 2 ? 'g2' : h.grade === 1 ? 'g1' : ''}`}>
                    <span className="hrank mono">#{h.rank}</span>
                    <img src={pageImage(q.dataset, h.corpus_id)} alt="" loading="lazy" />
                    <span className="hmeta">
                      <span className="mono small">
                        {h.doc_id} p.{h.page}
                      </span>
                      <span className={`chip ${h.grade === 2 ? 'ok' : h.grade === 1 ? 'warn' : 'no'}`}>
                        {h.grade === 2 ? '● full · 2' : h.grade === 1 ? '◐ part · 1' : '○ not gold'}
                      </span>
                    </span>
                  </div>
                </li>
              ))}
            </ol>
          </section>
        ))}
      </div>
    </>
  );
}
