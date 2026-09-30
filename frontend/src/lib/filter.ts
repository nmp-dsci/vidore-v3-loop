/**
 * The question list's filters and the page viewer's box maths, kept pure so
 * vitest checks them without a browser.
 */
import type { Box, QueryRow } from './api';

export type Filters = { text: string; type: string; content: string; format: string; author: string; check: string };
export const NO_FILTERS: Filters = { text: '', type: '', content: '', format: '', author: '', check: '' };

export function filterQueries(rows: QueryRow[], f: Filters): QueryRow[] {
  const text = f.text.trim().toLowerCase();
  return rows.filter(
    (q) =>
      (!text || q.query.toLowerCase().includes(text) || String(q.query_id) === text) &&
      (!f.type || q.query_types.includes(f.type)) &&
      (!f.content || q.content_type.includes(f.content)) &&
      (!f.format || q.query_format === f.format) &&
      (!f.author || q.query_generator === f.author) &&
      (!f.check || (f.check === 'any' ? q.hand_check != null : q.hand_check === f.check)),
  );
}

/** Distinct values of a field across the rows, most frequent first. */
export function facet(rows: QueryRow[], pick: (q: QueryRow) => string[]): [string, number][] {
  const n = new Map<string, number>();
  for (const q of rows) for (const v of pick(q)) n.set(v, (n.get(v) ?? 0) + 1);
  return [...n.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

/** Boxes arrive in the original page's pixels; one rect per box, ordered by annotator. */
export function boxRects(boxes: Box[]): { x: number; y: number; w: number; h: number; annotator: number }[] {
  return [...boxes]
    .sort((a, b) => a.annotator - b.annotator)
    .map((b) => ({
      x: Math.min(b.x1, b.x2),
      y: Math.min(b.y1, b.y2),
      w: Math.abs(b.x2 - b.x1),
      h: Math.abs(b.y2 - b.y1),
      annotator: b.annotator,
    }));
}

/** Filters ⇄ URL query string, so a filtered list is a link a reviewer can send. */
export function filtersFrom(sp: URLSearchParams): Filters {
  return {
    text: sp.get('q') ?? '',
    type: sp.get('type') ?? '',
    content: sp.get('content') ?? '',
    format: sp.get('format') ?? '',
    author: sp.get('author') ?? '',
    check: sp.get('check') ?? '',
  };
}
export function filtersTo(f: Filters): string {
  const sp = new URLSearchParams();
  const put = (k: string, v: string) => v && sp.set(k, v);
  put('q', f.text);
  put('type', f.type);
  put('content', f.content);
  put('format', f.format);
  put('author', f.author);
  put('check', f.check);
  const s = sp.toString();
  return s ? `?${s}` : '';
}
