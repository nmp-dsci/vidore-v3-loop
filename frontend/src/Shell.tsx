import { NavLink, Outlet } from 'react-router-dom';
import { type Health, useGet } from './lib/api';

/**
 * The tabs in the order the story runs: what the benchmark is → who else has tried →
 * what we built and ran. The same slots as tau2-loop's and DataAgentBench's viewers.
 * A tab whose milestone has not landed is shown, greyed, with the milestone it waits on
 * (s00 §7b), so the viewer's shape is visible from M1.
 */
const NAV: [string, string][] = [
  ['/', 'Overview'],
  ['/datasets', 'Datasets & questions'],
  ['/pipeline', 'RAG pipeline'],
  ['/runs', 'Runs'],
  ['/leaderboard', 'Leaderboard'],
];
const LATER: [string, string][] = [
  ['Review', 'M4'],
  ['Agent', 'M5'],
  ['Optimise', 'M6'],
];

export function Shell() {
  const { data: health } = useGet<Health>('/api/health');
  return (
    <>
      <header className="top">
        <div className="in">
          <NavLink to="/" className="brand">
            <img src="/favicon.svg" alt="" width="22" height="22" />
            <span>
              vidore<b>-v3-loop</b>
            </span>
          </NavLink>
          <nav aria-label="Pages">
            {NAV.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => (isActive ? 'on' : '')}>
                {label}
              </NavLink>
            ))}
            {LATER.map(([label, m]) => (
              <span key={label} className="later" title={`lands in ${m}`} aria-disabled="true">
                {label} <small>{m}</small>
              </span>
            ))}
          </nav>
          <span className="mode" title="reads the pinned datasets and committed files; retrieval runs open models locally; never calls Claude">
            {health ? 'read only · no Claude' : '…'}
          </span>
        </div>
      </header>
      <main>
        <Outlet />
      </main>
      <footer>
        Data from the pinned ViDoRe V3 datasets; page images from SEC and FDA (public domain), USAF (distribution A),
        European Commission and OpenStax (CC BY 4.0) · build <code>{health?.code_sha ?? '…'}</code> ·{' '}
        <a href="https://github.com/nmp-dsci/vidore-v3-loop">Source</a> ·{' '}
        <a href="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3">ViDoRe V3</a>.
      </footer>
    </>
  );
}
