/** The pieces every page shares, so the number rules (DESIGN.md) live in one place. */
import { Link } from 'react-router-dom';
import { DATASETS, datasetLabel } from './api';

export function Kpi({ n, b, tone }: { n: string; b: string; tone?: 'ok' | 'warn' }) {
  return (
    <div className="kpi">
      <div className={`n ${tone ?? ''}`}>{n}</div>
      <div className="b">{b}</div>
    </div>
  );
}

export function Loading({ error, what }: { error: string | null; what?: string }) {
  return <p className="empty">{error ? `Could not load: ${error}` : `Loading${what ? ` ${what}` : ''}…`}</p>;
}

/** Every English dataset as a lozenge, one click into it. */
export function DatasetChips({ current }: { current?: string }) {
  return (
    <nav className="chips domainbar" aria-label="datasets">
      {DATASETS.map((d) => (
        <Link
          key={d}
          to={`/datasets/${d}`}
          className={`chip nav ${d === current ? 'on' : ''}`}
          aria-current={d === current ? 'page' : undefined}
        >
          {datasetLabel(d)}
        </Link>
      ))}
    </nav>
  );
}

/** A hand check's verdict as a word with its glyph, never colour alone. */
export function CheckChip({ verdict }: { verdict: 'holds' | 'off' | null | undefined }) {
  if (!verdict) return null;
  return verdict === 'holds' ? (
    <span className="chip ok">● reference holds</span>
  ) : (
    <span className="chip warn">◐ reference off</span>
  );
}
