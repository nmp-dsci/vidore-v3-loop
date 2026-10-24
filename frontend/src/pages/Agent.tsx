import { Link, useParams } from 'react-router-dom';
import {
  type AgentPage,
  type AgentQuestion,
  type AgentRun,
  type AgentStep,
  type Mean,
  pageImage,
  pct1,
  useGet,
} from '../lib/api';
import { Kpi, Loading } from '../lib/ui';
import { AnswerText, VerdictChip } from './Review';

const when = (iso: string) => iso.replace('T', ' ').slice(0, 16);
const pctMean = (m: Mean | undefined | null) => (m?.mean == null ? '—' : `${pct1(m.mean)}%`);
const ndcgOf = (r: AgentRun) => (r.summary?.retrieval?.ndcg_10 as Mean | undefined) ?? null;

function RunStatus({ r }: { r: AgentRun }) {
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

/** A page as a grade chip: the gold grade is a word and a glyph, never colour alone. */
function Grade({ g }: { g: number }) {
  return (
    <span className={`chip ${g === 2 ? 'ok' : g === 1 ? 'warn' : 'no'}`}>
      {g === 2 ? '● full · 2' : g === 1 ? '◐ part · 1' : '○ not gold'}
    </span>
  );
}

function Hit({ k, p, rank, note }: { k: string; p: AgentPage; rank?: number; note?: string }) {
  return (
    <div className={`hit ${p.grade === 2 ? 'g2' : p.grade === 1 ? 'g1' : ''}`}>
      <span className="hrank mono">{rank != null ? `#${rank}` : ''}</span>
      <img src={pageImage(k, p.corpus_id)} alt="" loading="lazy" />
      <span className="hmeta">
        <span className="mono small">
          {p.doc_id} p.{p.page} · {p.corpus_id}
        </span>
        <Grade g={p.grade} />
        {note && <span className="small muted">{note}</span>}
      </span>
    </div>
  );
}

/** `/agent` — every agent run: its pages scored as a ranking, its answers scored by the judge. */
export function Agent() {
  const { data: rs, error } = useGet<AgentRun[]>('/api/runs?kind=agent');
  if (!rs) return <Loading error={error} what="the agent runs" />;
  return (
    <>
      <p className="label">Agent · search, read, then answer (M5)</p>
      <h1>
        The agent's runs, <em>both halves scored</em>
      </h1>
      <p className="lead">
        Claude searches our index and reads pages before it answers; its 10 pages are scored by NDCG@10 and its answer by
        the judge.
      </p>
      {rs.length === 0 ? (
        <p className="empty">
          No agent runs yet: start one with <code>uv run vidoreloop agent --dataset finance_en --limit 10</code>.
        </p>
      ) : (
        <div className="tw">
          <table className="runs fit">
            <thead>
              <tr>
                <th>run</th>
                <th>split</th>
                <th className="num">questions</th>
                <th>agent · judge</th>
                <th className="num">NDCG@10</th>
                <th className="num">% correct</th>
                <th className="num">cites a gold page</th>
                <th className="num">searches · views</th>
                <th className="num">s / question</th>
                <th>status</th>
              </tr>
            </thead>
            <tbody>
              {rs.map((r) => {
                const s = r.summary;
                return (
                  <tr key={r.run_id}>
                    <td className="runcell">
                      <Link className="nw" to={`/agent/${r.run_id}`}>
                        {when(r.started_at)}
                      </Link>
                      <div className="small muted">{r.label}</div>
                    </td>
                    <td>{r.split}</td>
                    <td className="num">{r.n}</td>
                    <td className="small mono">
                      {r.models.answer} · {r.models.effort}
                      <div className="muted">judge {r.models.judge}</div>
                    </td>
                    <td className="num">{pct1(ndcgOf(r)?.mean)}</td>
                    <td className="num">{pctMean(s?.answers?.raw)}</td>
                    <td className="num">{pctMean(s?.cited_gold)}</td>
                    <td className="num">
                      {s?.searches ?? '—'} · {s?.views ?? '—'}
                    </td>
                    <td className="num">{s?.seconds ?? '—'}</td>
                    <td>
                      <RunStatus r={r} />
                      {s?.errors ? <div className="small muted">{s.errors} errors</div> : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

/** `/agent/:runId` — one run: its headline numbers and every question's path length and verdict. */
export function AgentRunPage() {
  const { runId = '' } = useParams();
  const { data: r, error } = useGet<AgentRun>(`/api/agent-runs/${runId}`);
  if (!r) return <Loading error={error} what="the agent run" />;
  const s = r.summary;
  return (
    <>
      <p className="label">
        <Link to="/agent">Agent</Link> / {r.run_id}
      </p>
      <h1 className="qh1">
        {r.models.answer} · {r.models.effort} on {r.n} {r.split} questions
      </h1>
      <p className="small muted">
        judge {r.models.judge} · {r.models.judge_effort} · up to {r.config.max_searches} searches and{' '}
        {r.config.max_views} page views a question
        {r.mlflow_url && (
          <>
            {' · '}
            <a href={r.mlflow_url} target="_blank" rel="noreferrer">
              MLflow
            </a>
          </>
        )}
      </p>
      {s && (
        <div className="kpis">
          <Kpi n={pct1(ndcgOf(r)?.mean)} b="NDCG@10 of the agent's 10 pages" />
          <Kpi n={pctMean(s.answers?.raw)} b={`answers judged correct, of ${s.answers?.raw.n ?? 0}`} />
          <Kpi n={pctMean(s.cited_gold)} b="answers citing a gold page" />
          <Kpi n={`${s.searches ?? '—'} · ${s.views ?? '—'}`} b="searches · page views a question" />
          <Kpi
            n={`${s.seconds ?? '—'} s`}
            b={`a question · ≈$${r.totals?.cost_usd ?? '—'} API-equivalent in all`}
            tone={s.errors ? 'warn' : undefined}
          />
        </div>
      )}
      <div className="tw">
        <table className="runs fit">
          <thead>
            <tr>
              <th className="num">q</th>
              <th>question</th>
              <th className="num">NDCG@10</th>
              <th>verdict</th>
              <th className="num">searches · views</th>
              <th>cites gold</th>
            </tr>
          </thead>
          <tbody>
            {(r.questions ?? []).map((q) => (
              <tr key={q.query_id}>
                <td className="num mono">
                  <Link to={`/agent/${r.run_id}/${q.query_id}`}>{q.query_id}</Link>
                </td>
                <td className="wrap">
                  <Link to={`/agent/${r.run_id}/${q.query_id}`}>{q.query}</Link>
                </td>
                <td className="num">{pct1(q.ndcg_10)}</td>
                <td>{q.error ? <span className="chip">— error</span> : <VerdictChip v={q.verdict} />}</td>
                <td className="num">
                  {q.searches ?? '—'} · {q.views ?? '—'}
                </td>
                <td>{q.cited_gold == null ? '—' : q.cited_gold ? '● yes' : '○ no'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function Step({ k, s, i }: { k: string; s: AgentStep; i: number }) {
  if (s.tool === 'search')
    return (
      <li className="agent-step">
        <span className="hrank mono">{i}</span>
        <div className="agent-step-body">
          <b>search</b> <span className="mono small">“{s.query}”</span>
          {s.refused && <span className="small muted"> · refused, the search limit was reached</span>}
          {s.error && <span className="small muted"> · failed: {s.error}</span>}
          {s.results && (
            <div className="chips">
              {s.results.map((p, n) => (
                <span key={p.corpus_id} className={`chip ${p.grade === 2 ? 'ok' : p.grade === 1 ? 'warn' : ''}`}>
                  #{n + 1} {p.doc_id} p.{p.page}
                  {p.grade === 2 ? ' ●' : p.grade === 1 ? ' ◐' : ''}
                </span>
              ))}
            </div>
          )}
        </div>
      </li>
    );
  return (
    <li className="agent-step">
      <span className="hrank mono">{i}</span>
      <div className="agent-step-body">
        <b>view_page</b>
        {s.refused && <span className="small muted"> · refused, the page limit was reached</span>}
        {s.page && <Hit k={k} p={s.page} />}
      </div>
    </li>
  );
}

/** `/agent/:runId/:qid` — one question's path: each search and page viewed, then the reply. */
export function AgentQuestionPage() {
  const { runId = '', qid = '' } = useParams();
  const { data: q, error } = useGet<AgentQuestion>(`/api/agent-runs/${runId}/${qid}`);
  if (!q) return <Loading error={error} what="the question" />;
  const u = q.mode.usage?.answer;
  return (
    <>
      <p className="label">
        <Link to="/agent">Agent</Link> / <Link to={`/agent/${runId}`}>{runId}</Link> / q{q.query_id}
      </p>
      <h1 className="qh1">{q.query}</h1>
      <div className="row">
        <span className="small muted">{q.query_types.join(' · ')}</span>
        {q.prev != null && <Link to={`/agent/${runId}/${q.prev}`}>← q{q.prev}</Link>}
        {q.next != null && <Link to={`/agent/${runId}/${q.next}`}>q{q.next} →</Link>}
        <Link to={`/datasets/${q.dataset}/${q.query_id}`} className="small">
          the question and its evidence →
        </Link>
      </div>

      <div className="agent-answers">
        <section className="card">
          <span className="label">the agent's answer</span>
          {q.mode.error ? <p className="empty">No answer: {q.mode.error}</p> : <AnswerText text={q.mode.answer} />}
          <VerdictChip v={q.mode.verdict} />
          {q.mode.explanation && <p className="small muted">{q.mode.explanation}</p>}
        </section>
        <section className="card">
          <span className="label">the gold answer</span>
          <AnswerText text={q.reference} />
          {u && (
            <p className="small muted mono">
              {u.turns ?? '—'} turns · {u.input_tokens.toLocaleString()} tokens in · ≈${(u.cost_usd ?? 0).toFixed(3)}
            </p>
          )}
        </section>
      </div>

      <h3>The path, step by step</h3>
      {q.steps.length === 0 ? (
        <p className="small muted">The agent answered without a tool call.</p>
      ) : (
        <ol className="agent-steps">
          {q.steps.map((s, i) => (
            <Step key={i} k={q.dataset} s={s} i={i + 1} />
          ))}
        </ol>
      )}

      <h3>
        Its 10 pages · NDCG@10 {pct1(q.metrics?.ndcg_10)}
      </h3>
      <p className="small muted">
        The first {q.agent_ranked} it ranked itself; the rest are filled from its searches.
      </p>
      <ol className="hits agent-ranking">
        {q.ranking.map((p, i) => (
          <li key={p.corpus_id}>
            <Hit
              k={q.dataset}
              p={p}
              rank={i + 1}
              note={q.cited.some((c) => c.corpus_id === p.corpus_id) ? 'cited in the answer' : undefined}
            />
          </li>
        ))}
      </ol>

      <h3>The gold pages</h3>
      <ol className="hits agent-ranking">
        {q.gold.map((p) => {
          const at = q.ranking.findIndex((r) => r.corpus_id === p.corpus_id);
          return (
            <li key={p.corpus_id}>
              <Hit k={q.dataset} p={p} note={at >= 0 ? `the agent ranked it #${at + 1}` : 'not in its 10'} />
            </li>
          );
        })}
      </ol>
    </>
  );
}
