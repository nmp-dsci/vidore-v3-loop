import { Link } from 'react-router-dom';
import { type Board, type OurRow, type Stage, STAGES, STAGE_LABEL, datasetLabel, pct1, useGet } from '../lib/api';
import { Loading } from '../lib/ui';

const TOP_MODELS = 15;

const ourShortKind = (s: Stage) =>
  ({ text: 'text', visual: 'visual LI', fused: 'visual LI + text', reranked: 'visual LI + text + reranker' })[s];
function ourKind(s: Stage, o: OurRow) {
  const visual = o.models.visual?.split('/')[1] ?? 'visual';
  const [wt, wv] = o.config.weights;
  if (s === 'text') return 'BM25 on the OCR text';
  if (s === 'visual') return visual;
  const fused = `${visual} + BM25, fused ${wt}:${wv}`;
  return s === 'fused' ? fused : `${fused}, zerank-2 on the top ${o.config.rerank_depth}`;
}

/** `/leaderboard` — the two retrieval boards and the paper's answer and grounding tables. */
export function Leaderboard() {
  const { data: b, error } = useGet<Board>('/api/leaderboard');
  if (!b) return <Loading error={error} what="the boards" />;
  const heads = b.datasets.map((d) => (
    <th key={d} className="num">
      {datasetLabel(d)}
    </th>
  ));
  const best = (i: number, rows: { scores: (number | null)[] }[]) => Math.max(...rows.map((r) => r.scores[i] ?? -1));
  const pipeBest = b.datasets.map((_, i) => best(i, b.pipelines));
  const modelRows = b.models.slice(0, TOP_MODELS);
  const modelBest = b.datasets.map((_, i) => best(i, modelRows));
  return (
    <>
      <p className="label">Leaderboard · English queries · 5 public English datasets</p>
      <h1>
        An agent that re-reads the pages leads retrieval at <em>74.6 NDCG@10</em>; no board scores answers
      </h1>
      <p className="lead">
        Two retrieval boards (ViDoRe pipelines and MTEB single models) on English queries, plus the paper's answer and
        grounding tables.
      </p>

      <dl className="kinds">
        <dt>visual / text</dt>
        <dd>what the model reads: the page image (tables and charts as laid out), or the page's OCR text only</dd>
        <dt>LI</dt>
        <dd>
          late interaction: one vector per image patch or word; each question word is matched to its best patch (MaxSim)
        </dd>
        <dt>dense</dt>
        <dd>one vector per page, compared with one vector for the question</dd>
        <dt>+ reranker · + LLM agent</dt>
        <dd>
          a second model rereads the top 50–100 and reorders them; an agent also rewrites the query and searches again
        </dd>
      </dl>

      <h2>Pipelines — the ViDoRe pipeline board</h2>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>pipeline</th>
              <th>kind</th>
              {heads}
              <th className="num">mean of 5</th>
              <th>latency</th>
            </tr>
          </thead>
          <tbody>
            {b.pipelines.map((p, r) => (
              <tr key={p.name} className={r === 0 ? 'lead-row' : ''}>
                <td className="sub wrap">{p.name}</td>
                <td>{p.kind}</td>
                {p.scores.map((s, i) => (
                  <td key={i} className={`num ${s === pipeBest[i] ? 'best' : ''}`}>
                    {s.toFixed(1)}
                  </td>
                ))}
                <td className="num best">{p.mean.toFixed(1)}</td>
                <td className="small">{p.latency ?? '—'}</td>
              </tr>
            ))}
            {b.ours.flatMap((o) =>
              STAGES.filter((s) => o.ndcg_10[s] != null).map((s) => (
                <tr key={`${o.run_id}-${s}`} className="ours-row">
                  <td className="sub wrap">
                    <Link to={`/runs/${o.run_id}`}>ours · {STAGE_LABEL[s][0]}</Link>
                    <div className="small muted">
                      {ourKind(s, o)} · our test split, {o.n} questions
                    </div>
                  </td>
                  <td>{ourShortKind(s)}</td>
                  {b.datasets.map((d) => (
                    <td key={d} className="num">
                      {d === o.dataset ? pct1(o.ndcg_10[s]) : '—'}
                    </td>
                  ))}
                  <td className="num muted">—</td>
                  <td className="small">local, Apple M1 Pro</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        NDCG@10 on English queries, from illuin-tech/vidore-benchmark (last entry 2026-03-13).
      </p>
      {b.ours.length > 0 ? (
        <p className="small muted">
          Our rows are scored once on our test half, so they are close to, not exactly, like for like.
        </p>
      ) : (
        <p className="small muted">Our rows appear once a configuration chosen on train is scored on test.</p>
      )}

      <h2>
        Single models — MTEB ViDoRe(v3), top {TOP_MODELS} of {b.models.length}
      </h2>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th className="num">#</th>
              <th>model</th>
              {heads}
              <th className="num">mean of 5</th>
            </tr>
          </thead>
          <tbody>
            {modelRows.map((m, r) => (
              <tr key={m.model}>
                <td className="num">{r + 1}</td>
                <td className="sub">
                  {m.model}
                  <span className="path">rev {m.revision}</span>
                </td>
                {m.scores.map((s, i) => (
                  <td key={i} className={`num ${s === modelBest[i] ? 'best' : ''}`}>
                    {s.toFixed(1)}
                  </td>
                ))}
                <td className="num">{m.mean.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        English-query NDCG@10 from embeddings-benchmark/results (2026-09-26), saved in{' '}
        <code>docs/research/mteb_vidore3_scores_2026-09-28.json</code>.
      </p>

      <h2>Answers — the paper's end-to-end table</h2>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>pages given</th>
              <th>as</th>
              <th>generator</th>
              {heads}
              <th className="num">hard qs</th>
              <th className="num">all 8 sets</th>
            </tr>
          </thead>
          <tbody>
            {b.answers.rows.map((r, n) => (
              <tr key={n}>
                <td className="sub">{r.pages}</td>
                <td>{r.as}</td>
                <td>{r.generator}</td>
                {r.scores.map((s, i) => (
                  <td key={i} className="num">
                    {s == null ? '—' : `${s.toFixed(1)}%`}
                  </td>
                ))}
              </tr>
            ))}
            {(b.ours_answers ?? []).flatMap((o) =>
              (['retrieved', 'oracle'] as const)
                .filter((m) => o.accuracy[m] != null)
                .map((m) => (
                  <tr key={`${o.run_id}-${m}`} className="ours-row">
                    <td className="sub">
                      <Link to={`/review/${o.run_id}`}>ours · {m === 'retrieved' ? 'our top 5' : 'oracle'}</Link>
                    </td>
                    <td>image + text</td>
                    <td>{o.models.answer}</td>
                    {b.datasets.map((d) => (
                      <td key={d} className="num">
                        {d === o.dataset ? `${((o.accuracy[m] ?? 0) * 100).toFixed(1)}%` : '—'}
                      </td>
                    ))}
                    <td className="num">—</td>
                    <td className="num">—</td>
                  </tr>
                )),
            )}
          </tbody>
        </table>
      </div>
      <p className="small muted">% judged correct by {b.answers.judge} (arXiv 2601.08620 v2, Tables 3 and 22).</p>

      <h2>Grounding — boxes around the evidence</h2>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>who</th>
              {heads}
              <th className="num">overall</th>
            </tr>
          </thead>
          <tbody>
            {b.grounding.rows.map((r) => (
              <tr key={r.who}>
                <td className="sub">{r.who}</td>
                {r.scores.map((s, i) => (
                  <td key={i} className="num">
                    {s.toFixed(3)}
                  </td>
                ))}
                <td className="num">{r.overall.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">{b.grounding.metric} (arXiv 2601.08620 v2, Tables 20–21).</p>
    </>
  );
}
