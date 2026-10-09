/** The pieces every page shares, so the number rules (DESIGN.md) live in one place. */
import { useEffect, useState } from 'react';
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
  if (error) return <p className="empty">Could not load: {error}</p>;
  return <Busy label={`Loading${what ? ` ${what}` : ''}`} />;
}

/** Work in progress: an accent panel with a pulsing dot, a sliding bar and the seconds elapsed. */
export function Busy({ label, detail, since }: { label: string; detail?: string; since?: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (since == null) return;
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, [since]);
  return (
    <div className="busy" role="status" aria-live="polite">
      <span className="busy-dot" aria-hidden="true" />
      <b>{label}…</b>
      {since != null && <span className="mono">{Math.max(0, Math.floor((now - since) / 1000))} s</span>}
      {detail && <span className="small busy-detail">{detail}</span>}
      <span className="busy-bar" aria-hidden="true" />
    </div>
  );
}

/** Every English dataset as a lozenge, one click into it. */
export function DatasetChips({ current, hrefFor }: { current?: string; hrefFor?: (d: string) => string }) {
  return (
    <nav className="chips domainbar" aria-label="datasets">
      {DATASETS.map((d) => (
        <Link
          key={d}
          to={hrefFor ? hrefFor(d) : `/datasets/${d}`}
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
