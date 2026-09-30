import { type Board, datasetLabel, useGet } from '../lib/api';
import { Loading } from '../lib/ui';

const TOP_MODELS = 15;

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
        Retrieval has two boards: the ViDoRe pipeline board and MTEB's single-model board. Both are shown on English
        queries, where they agree (nemotron-colembed-vl-8b-v2 scores 69.1 on each). Answer accuracy and grounding exist
        only in the paper. Our own rows arrive in M3 (retrieval) and M4 (answers).
      </p>

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
          </tbody>
        </table>
      </div>
      <p className="small muted">NDCG@10 over pages, English queries. Source: github.com/illuin-tech/vidore-benchmark results/metrics (last entry 2026-03-13).</p>

      <h2>Single models — MTEB ViDoRe(v3), top {TOP_MODELS} of {b.models.length}</h2>
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
        English-query NDCG@10 from embeddings-benchmark/results (2026-09-26), committed as{' '}
        <code>docs/research/mteb_vidore3_scores_2026-09-28.json</code>. The official MTEB number averages six query
        languages and ten datasets, so it runs lower.
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
          </tbody>
        </table>
      </div>
      <p className="small muted">% judged correct. Judge: {b.answers.judge}. arXiv 2601.08620 v2 Table 3; the no-document row from Table 22.</p>

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
      <p className="small muted">{b.grounding.metric}. arXiv 2601.08620 v2 Tables 20–21.</p>
    </>
  );
}
