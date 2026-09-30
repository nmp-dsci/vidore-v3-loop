import { Link } from 'react-router-dom';
import { type DatasetCard, datasetLabel, fmtN, useGet } from '../lib/api';
import { Loading } from '../lib/ui';

const BLURB: Record<string, string> = {
  finance_en: '2024 10-K annual reports of six US banks. Tables carry most answers.',
  pharmaceuticals: 'FDA / CDER slide decks, 2017–2021. The most charts and infographics.',
  industrial: 'US Air Force technical orders: maintenance manuals with diagrams. The hardest set.',
  hr: 'European Commission labour and employment reports, 2024. Charts and tables.',
  computer_science: 'Two OpenStax textbooks. Most questions are answerable from memory.',
};

const MILESTONES: [string, string, 'ok' | 'warn' | 'no'][] = [
  ['M0', 'scaffold: pinned datasets, loader, NDCG@10, smoke, CI', 'ok'],
  ['M1', 'this viewer: datasets, questions, reference answers, evidence pages with human boxes; the board', 'warn'],
  ['M2', 'retrieval pipeline (BM25S, visual 4B, RRF, reranker) + Search tab', 'no'],
  ['M3', 'retrieval scored on our finance_en split + Runs tab, our row on the Leaderboard', 'no'],
  ['M4', 'Claude answers, judge and reference audit + Review tab', 'no'],
  ['M5', 'the agent + Agent tab', 'no'],
  ['M6', 'the self-improving loop + Optimise tab', 'no'],
  ['M7', 'grounding: our boxes against the human boxes', 'no'],
];

export function Overview() {
  const { data, error } = useGet<DatasetCard[]>('/api/datasets');
  return (
    <>
      <p className="label">vidore-v3-loop · ViDoRe V3, English datasets, one at a time</p>
      <h1>
        Five English corpora of real PDFs, where the answer often sits in <em>a table, chart or diagram</em>
      </h1>
      <p className="lead">
        Each question comes with human-graded relevant pages, human boxes around the evidence, and a reference answer.
        Review them here before anything is built on top. The plan is{' '}
        <code>.lavish/s00_vidore-v3-research-plan.html</code>; the first dataset is finance.
      </p>
      {!data ? (
        <Loading error={error} what="datasets" />
      ) : (
        <div className="cards">
          {data.map((d) => (
            <Link key={d.key} to={`/datasets/${d.key}`} className="card dscard">
              <span className="label">{d.key}</span>
              <h3>{datasetLabel(d.key)}</h3>
              <p className="small">{BLURB[d.key]}</p>
              <dl className="facts">
                <dt>English questions</dt>
                <dd className="mono">{fmtN(d.queries)}</dd>
                <dt>pages</dt>
                <dd className="mono">
                  {fmtN(d.pages)} in {d.documents} docs
                </dd>
                <dt>answerable blind</dt>
                <dd className="mono">{d.answerable_blind_pct}% of questions</dd>
              </dl>
            </Link>
          ))}
        </div>
      )}
      <p className="small muted">
        “Answerable blind” is the share of questions at least one of six LLMs answered with no documents (arXiv
        2601.08620 Table 22): a high share tests model memory, not the system.
      </p>

      <h2>Milestones — each ships a backend and the tab that shows it</h2>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>milestone</th>
              <th>what it delivers</th>
              <th>state</th>
            </tr>
          </thead>
          <tbody>
            {MILESTONES.map(([m, what, st]) => (
              <tr key={m} className={st === 'warn' ? 'cur' : ''}>
                <td className="sub mono">{m}</td>
                <td className="wrap">{what}</td>
                <td className="nw">
                  <span className={`status ${st}`}>{st === 'ok' ? 'done' : st === 'warn' ? 'in review' : 'designed'}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
