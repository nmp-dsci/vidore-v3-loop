import { useEffect, useState } from 'react';

/** English datasets, in the order the plan ranks them (s00 §3): commercial value first. */
export const DATASETS = ['finance_en', 'pharmaceuticals', 'industrial', 'hr', 'computer_science'] as const;
export type DatasetKey = (typeof DATASETS)[number];

const LABELS: Record<string, string> = {
  finance_en: 'Finance',
  pharmaceuticals: 'Pharmaceuticals',
  industrial: 'Industrial',
  hr: 'HR',
  computer_science: 'Computer science',
};
export const datasetLabel = (k: string) => LABELS[k] ?? k;

export type Health = { status: string; code_sha: string; datasets: string[] };

export type DatasetCard = {
  key: string;
  domain: string;
  repo_id: string;
  revision: string;
  documents: number;
  pages: number;
  queries: number;
  answerable_blind_pct: number;
};

export type Doc = {
  doc_id: string;
  file_name: string;
  url: string;
  doc_type: string;
  doc_language: string;
  doc_year: string;
  visual_types: string[];
  pages: number;
  license: string;
};

export type Summary = {
  n_queries: number;
  by_type: [string, number][];
  by_content: [string, number][];
  by_format: [string, number][];
  by_generator: [string, number][];
  pages_per_query: { mean: number; median: number; max: number };
  hand_checked: number;
};

export type DatasetDetail = {
  key: string;
  domain: string;
  repo_id: string;
  revision: string;
  paper_revision: string;
  pages: number;
  documents: Doc[];
  summary: Summary;
};

export type QueryRow = {
  query_id: number;
  query: string;
  query_types: string[];
  query_format: string;
  content_type: string[];
  query_generator: string;
  n_pages: number;
  n_full: number;
  hand_check: 'holds' | 'off' | null;
};

export type Box = { annotator: number; x1: number; y1: number; x2: number; y2: number };
export type RelPage = { corpus_id: number; score: number; content_type: string[]; boxes: Box[] };
export type HandCheck = {
  dataset: string;
  query_id: number;
  verdict: 'holds' | 'off';
  checked_answer: string;
  note: string;
  checked_by: string;
  date: string;
};
export type Question = QueryRow & {
  answer: string;
  raw_answers: string[];
  query_generation_pipeline: string | null;
  query_type_for_generation: string | null;
  source_type: string | null;
  pages: RelPage[];
  hand_check: HandCheck | null;
};

export type PageMeta = {
  corpus_id: number;
  doc_id: string;
  page: number;
  width: number;
  height: number;
  markdown: string;
};

export type ModelRow = { model: string; revision: string; scores: number[]; mean: number };
export type PipelineRow = { name: string; kind: string; scores: number[]; mean: number; latency: string | null };
export type Board = {
  datasets: string[];
  models: ModelRow[];
  pipelines: PipelineRow[];
  answers: {
    columns: string[];
    judge: string;
    rows: { pages: string; as: string; generator: string; scores: (number | null)[] }[];
    answerable_blind_pct: number[];
  };
  grounding: { metric: string; rows: { who: string; scores: number[]; overall: number }[] };
  about: string;
};

export async function get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((j as { detail?: string }).detail ?? `${r.status} ${url}`);
  return j as T;
}

export function useGet<T>(url: string | null): { data: T | null; error: string | null; loading: boolean } {
  const [state, set] = useState<{ data: T | null; error: string | null; loading: boolean }>({
    data: null,
    error: null,
    loading: !!url,
  });
  useEffect(() => {
    if (!url) return;
    let alive = true;
    set({ data: null, error: null, loading: true });
    get<T>(url)
      .then((d) => alive && set({ data: d, error: null, loading: false }))
      .catch((e: Error) => alive && set({ data: null, error: e.message, loading: false }));
    return () => {
      alive = false;
    };
  }, [url]);
  return state;
}

export const pageImage = (key: string, corpusId: number) => `/api/pages/${key}/${corpusId}/image`;
export const fmtN = (n: number) => n.toLocaleString('en-AU');

export async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((j as { detail?: string }).detail ?? `${r.status} ${url}`);
  return j as T;
}

/** Retrieval (M2): one question through the stages. */
export const STAGES = ['text', 'visual', 'fused', 'reranked'] as const;
export type Stage = (typeof STAGES)[number];
export const STAGE_LABEL: Record<Stage, [string, string]> = {
  text: ['A · text', 'BM25S on the OCR markdown'],
  visual: ['B · visual', 'EVIE-4.5B on the page image'],
  fused: ['C · fused', 'reciprocal rank fusion → 50'],
  reranked: ['D · reranked', 'zerank-2 reads the 50'],
};

export type Metrics = {
  ndcg_10: number;
  map_10: number;
  mrr: number;
  first_rank: number | null;
  first_full_rank: number | null;
  precision_5: number;
  precision_10: number;
  hit_full_5: boolean;
  hit_full_10: boolean;
  complete_10: boolean;
  complete_50: boolean;
  n_relevant: number;
  n_full: number;
  recall_1: number;
  recall_5: number;
  recall_10: number;
  recall_50: number;
  recall_100: number;
};
export type Hit = {
  corpus_id: number;
  rank: number;
  score: number;
  doc_id: string | null;
  page: number | null;
  grade: number | null;
  ranks: Partial<Record<Stage, number | null>>;
  terms?: string[];
};
export type StageOut = {
  unavailable?: string;
  note?: string;
  ms?: number;
  inputs?: string[];
  model?: string;
  hits?: Hit[];
  n?: number;
  metrics?: Metrics;
  gold_ranks?: Record<string, number | null>;
};
export type SearchOut = {
  dataset: string;
  query: string;
  query_id: number | null;
  gold: { corpus_id: number; grade: number }[];
  stages: Partial<Record<Stage, StageOut>>;
  deltas: Record<string, { gained: number[]; lost: number[] }>;
};
export type RetrievalStatus = {
  text: boolean;
  visual: boolean;
  visual_complete: boolean;
  visual_stats: { pages: number; seconds_per_page: number; mean_vectors_per_page: number; device: string } | null;
  reranker: boolean;
};
