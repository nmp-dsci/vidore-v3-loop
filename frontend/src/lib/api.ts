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
