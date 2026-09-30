import { describe, expect, it } from 'vitest';
import type { QueryRow } from './api';
import { NO_FILTERS, boxRects, facet, filterQueries, filtersFrom, filtersTo } from './filter';

const row = (id: number, over: Partial<QueryRow> = {}): QueryRow => ({
  query_id: id,
  query: `question ${id}`,
  query_types: ['extractive'],
  query_format: 'question',
  content_type: ['Table'],
  query_generator: 'human',
  n_pages: 1,
  n_full: 1,
  hand_check: null,
  ...over,
});

const rows = [
  row(1),
  row(2, { query_types: ['numerical', 'extractive'], content_type: ['Chart'], hand_check: 'off' }),
  row(3, { query_generator: 'sdg', query_format: 'keyword', query: 'uninsured deposits', hand_check: 'holds' }),
];

describe('filterQueries', () => {
  it('passes everything with no filters', () => {
    expect(filterQueries(rows, NO_FILTERS)).toHaveLength(3);
  });
  it('matches text case-insensitively, or an exact query id', () => {
    expect(filterQueries(rows, { ...NO_FILTERS, text: 'UNINSURED' }).map((q) => q.query_id)).toEqual([3]);
    expect(filterQueries(rows, { ...NO_FILTERS, text: '2' }).map((q) => q.query_id)).toEqual([2]);
  });
  it('filters by type membership, content, format, author and hand check', () => {
    expect(filterQueries(rows, { ...NO_FILTERS, type: 'numerical' }).map((q) => q.query_id)).toEqual([2]);
    expect(filterQueries(rows, { ...NO_FILTERS, content: 'Table' }).map((q) => q.query_id)).toEqual([1, 3]);
    expect(filterQueries(rows, { ...NO_FILTERS, format: 'keyword', author: 'sdg' }).map((q) => q.query_id)).toEqual([3]);
    expect(filterQueries(rows, { ...NO_FILTERS, check: 'any' }).map((q) => q.query_id)).toEqual([2, 3]);
    expect(filterQueries(rows, { ...NO_FILTERS, check: 'off' }).map((q) => q.query_id)).toEqual([2]);
  });
});

describe('facet', () => {
  it('counts values across rows, most frequent first', () => {
    expect(facet(rows, (q) => q.query_types)).toEqual([
      ['extractive', 3],
      ['numerical', 1],
    ]);
  });
});

describe('boxRects', () => {
  it('normalises inverted corners and orders by annotator', () => {
    expect(
      boxRects([
        { annotator: 2, x1: 50, y1: 60, x2: 10, y2: 20 },
        { annotator: 1, x1: 0, y1: 0, x2: 5, y2: 5 },
      ]),
    ).toEqual([
      { x: 0, y: 0, w: 5, h: 5, annotator: 1 },
      { x: 10, y: 20, w: 40, h: 40, annotator: 2 },
    ]);
  });
});

describe('filters in the URL', () => {
  it('round-trips, and an empty filter set is no query string', () => {
    const f = { ...NO_FILTERS, text: 'deposits', type: 'numerical', check: 'off' };
    expect(filtersFrom(new URLSearchParams(filtersTo(f).slice(1)))).toEqual(f);
    expect(filtersTo(NO_FILTERS)).toBe('');
  });
});
