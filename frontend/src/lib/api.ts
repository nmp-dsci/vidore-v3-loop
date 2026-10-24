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

export type Health = {
  status: string;
  code_sha: string;
  datasets: string[];
  claude: { available: boolean; detail: string };
};

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

export type Box = {
  annotator: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};
export type RelPage = {
  corpus_id: number;
  score: number;
  content_type: string[];
  boxes: Box[];
};
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

export type ModelRow = {
  model: string;
  revision: string;
  scores: number[];
  mean: number;
};
export type PipelineRow = {
  name: string;
  kind: string;
  scores: number[];
  mean: number;
  latency: string | null;
};
export type Board = {
  datasets: string[];
  models: ModelRow[];
  pipelines: PipelineRow[];
  answers: {
    columns: string[];
    judge: string;
    rows: {
      pages: string;
      as: string;
      generator: string;
      scores: (number | null)[];
    }[];
    answerable_blind_pct: number[];
  };
  grounding: {
    metric: string;
    rows: { who: string; scores: number[]; overall: number }[];
  };
  about: string;
  ours: OurRow[];
  ours_answers: OurAnswerRow[];
  ours_agent?: OurAgentRow[];
};

export async function get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((j as { detail?: string }).detail ?? `${r.status} ${url}`);
  return j as T;
}

export function useGet<T>(url: string | null): {
  data: T | null;
  error: string | null;
  loading: boolean;
} {
  const [state, set] = useState<{
    data: T | null;
    error: string | null;
    loading: boolean;
  }>({
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
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((j as { detail?: string }).detail ?? `${r.status} ${url}`);
  return j as T;
}

/** Retrieval (M2): one question through the stages. */
export const STAGES = ['text', 'visual', 'fused', 'reranked'] as const;
export type Stage = (typeof STAGES)[number];
export const STAGE_LABEL: Record<Stage, [string, string]> = {
  text: ['A · text', 'BM25S on the OCR markdown'],
  visual: ['B · visual', 'late interaction on the page image'],
  fused: ['C · fused', 'reciprocal rank fusion → 50'],
  reranked: ['D · reranked', 'zerank-2 rereads the top of the fused list'],
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
  scores?: Partial<Record<Stage, number | null>>;
  terms?: string[];
};
export type StageOut = {
  unavailable?: string;
  note?: string;
  ms?: number;
  inputs?: string[];
  weights?: Record<string, number>;
  model?: string;
  depth?: number;
  scored?: number;
  cached?: number;
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
  visual_building: boolean;
  visual_stats: {
    model: string;
    pages: number;
    seconds_per_page: number | null;
    mean_vectors_per_page: number | null;
    device: string;
  } | null;
  reranker: boolean;
};

/** Scored runs (M3): a pipeline configuration over our train or test split. */
export type Mean = { mean: number | null; n: number };
export type RankStat = { median: number | null; found: number; n: number };
export type StageSummary = Record<string, Mean | RankStat>;
export type SetupRow = { k: string; v: string; link?: string; mono?: boolean };
export type SetupSection = { section: string; rows: SetupRow[] };

export type RunMeta = {
  run_id: string;
  setup?: SetupSection[];
  dataset: string;
  split: 'train' | 'test';
  final: boolean;
  label: string;
  config: {
    stages: Stage[];
    rerank_depth: number;
    weights: [number, number];
    note: string;
    limit: number | null;
  };
  fingerprint: string;
  models: {
    visual: string | null;
    visual_dim: number | null;
    reranker: string | null;
    text: string;
  };
  channel_k: number;
  fused_k: number;
  rrf_k: number;
  split_seed: number;
  split_fingerprint: string;
  code_sha: string;
  started_at: string;
  finished_at: string | null;
  seconds?: number;
  status: 'running' | 'finished' | 'failed' | 'interrupted';
  error?: string;
  progress: { done: number; n: number; seconds?: number };
  n: number;
  summary: Partial<Record<Stage, StageSummary>> | null;
  costs: Partial<
    Record<
      Stage,
      {
        mean_ms: number | null;
        pages_scored?: number;
        pages_cached?: number;
        rerank_seconds?: number;
      }
    >
  > | null;
  mlflow_run_id: string | null;
  mlflow_error?: string;
};
export type SliceRow = {
  value: string;
  n: number;
  stages: Partial<Record<Stage, StageSummary>>;
};
export type RunDetail = RunMeta & { slices: Record<string, SliceRow[]> | null };
export type RunQueryRow = {
  query_id: number;
  query: string;
  query_types: string[];
  content_type: string[];
  query_format: string;
  author: string;
  n_gold: number;
  n_full: number;
  stages: Partial<
    Record<
      Stage,
      {
        ndcg_10: number;
        recall_10: number;
        first_rank: number | null;
        first_full_rank: number | null;
        hit_full_10: boolean;
      }
    >
  >;
};
export type RunPage = {
  corpus_id: number;
  rank: number | null;
  doc_id: string | null;
  page: number | null;
  grade: number;
};
export type RunQuery = {
  run_id: string;
  dataset: string;
  query_id: number;
  query: string;
  query_types: string[];
  content_type: string[];
  query_format: string;
  query_generator: string;
  gold: RunPage[];
  stages: Partial<
    Record<
      Stage,
      {
        metrics: Metrics;
        ms: number | null;
        top: RunPage[];
        gold_ranks: Record<string, number | null>;
        scored?: number;
        cached?: number;
        depth?: number;
      }
    >
  >;
};
export type SplitInfo = {
  dataset: string;
  seed: number;
  stratified_by: string;
  n: { train: number; test: number; holdout: number };
  fingerprint: string;
  strata: { stratum: string; train: number; test: number; holdout: number }[];
};
export type OurRow = {
  run_id: string;
  dataset: string;
  label: string;
  n: number;
  models: RunMeta['models'];
  config: RunMeta['config'];
  ndcg_10: Partial<Record<Stage, number | null>>;
  all?: Partial<Record<Stage, Mean>>; // with its train twin: every question, as the boards score
};
export const pct1 = (x: number | null | undefined) => (x == null ? '—' : (x * 100).toFixed(1));

// ---- M4: answers, the judge, the audit ----------------------------------------

export const MODES = ['retrieved', 'oracle'] as const;
export type Mode = (typeof MODES)[number];
export const MODE_LABEL: Record<Mode, [string, string]> = {
  retrieved: ['retrieved', 'the top 5 pages of our retrieval'],
  oracle: ['oracle', 'the human-marked pages: the ceiling'],
};
export type Verdict = 'Correct' | 'Partially Correct' | 'Incorrect';
export type ModeSummary = {
  raw: Mean;
  audited: Mean;
  verdicts: Record<Verdict, number>;
  errors: number;
  answer_tokens: number | null;
  ms: number | null;
  full_given?: Mean;
  correct_when_full_given?: Mean;
  correct_when_full_missed?: Mean;
};
export type AnswerSummary = Partial<Record<Mode, ModeSummary>> & {
  retrieval_cost?: { raw: number | null; audited: number | null };
};
export type AnswerRun = {
  run_id: string;
  setup?: SetupSection[];
  kind: 'answer';
  dataset: string;
  split: string;
  final: boolean;
  status: string;
  n: number;
  label: string;
  started_at: string;
  finished_at: string | null;
  progress: { done: number; n: number; seconds?: number };
  error?: string;
  config: {
    modes: Mode[];
    k: number;
    source_run: string | null;
    source_stage: string;
    modality: string;
    note: string;
    limit: number | null;
  };
  models: { answer: string; effort: string; judge: string; judge_effort: string };
  summary: AnswerSummary | null;
  audited?: AnswerSummary;
  totals?: { calls: number; input_tokens: number; output_tokens: number; cost_usd: number };
  calibration?: { labelled: number; agree: number; target: number };
  disputed?: number;
  mlflow_run_id?: string | null;
};
export type AnswerRow = {
  query_id: number;
  query: string;
  query_types: string[];
  n_full: number;
  reference: 'disputed' | 'holds' | null;
  modes: Partial<
    Record<
      Mode,
      {
        verdict: Verdict | null;
        correct: boolean | null;
        person: boolean | null;
        full_given?: boolean;
        error?: string;
      }
    >
  >;
};
export type GivenPage = { corpus_id: number; doc_id: string | null; page: number | null; grade: number };
export type JudgeLabel = { correct: boolean; judge_correct: boolean; agrees: boolean; note: string; at: string };
export type ModeAnswer = {
  answer: string | null;
  verdict: Verdict | null;
  correct: boolean | null;
  explanation?: string;
  error?: string;
  full_given?: boolean;
  pages: GivenPage[];
  label: JudgeLabel | null;
  usage?: { answer: Usage; judge: Usage };
};
export type Usage = { input_tokens: number; output_tokens: number; ms: number; cost_usd: number | null; model: string };
export type ReferenceState = { status: 'disputed' | 'holds'; note: string; by: string; at: string; source: string };
export type AnswerQuestion = {
  run_id: string;
  dataset: string;
  query_id: number;
  query: string;
  answer: string;
  raw_answers: string[];
  query_types: string[];
  reference: ReferenceState | null;
  gold: GivenPage[];
  modes: Partial<Record<Mode, ModeAnswer>>;
  prev: number | null;
  next: number | null;
};
export type LiveAnswer = {
  answer: string;
  pages: number[] | null;
  usage: Usage;
  reference?: string;
  judge?: { verdict: Verdict; correct: boolean; explanation: string; usage: Usage };
};
export type OurAnswerRow = {
  run_id: string;
  dataset: string;
  n: number;
  models: { answer: string; judge: string };
  accuracy: Partial<Record<Mode, number | null>>;
  all?: Partial<Record<Mode, Mean>>;
};

// ---- M5: the agent -------------------------------------------------------------

export type AgentSummary = {
  retrieval: StageSummary | null;
  answers: ModeSummary | null;
  cited_gold: Mean;
  cited_full: Mean;
  searches: number | null;
  views: number | null;
  turns: number | null;
  seconds: number | null;
  errors: number;
};
export type AgentRun = {
  run_id: string;
  kind: 'agent';
  dataset: string;
  split: string;
  final: boolean;
  status: string;
  n: number;
  label: string;
  started_at: string;
  progress: { done: number; n: number; seconds?: number };
  error?: string;
  config: { max_searches: number; max_views: number; note: string; limit: number | null };
  models: { answer: string; effort: string; judge: string; judge_effort: string };
  summary: AgentSummary | null;
  totals?: { calls: number; input_tokens: number; output_tokens: number; cost_usd: number };
  mlflow_url?: string | null;
  questions?: AgentQuestionRow[];
};
export type AgentQuestionRow = {
  query_id: number;
  query: string;
  query_types: string[];
  n_full: number;
  ndcg_10: number | null;
  verdict: Verdict | null;
  correct: boolean | null;
  error?: string | null;
  searches: number | null;
  views: number | null;
  cited_gold: boolean | null;
};
export type AgentPage = { corpus_id: number; doc_id: string | null; page: number | null; grade: number };
export type AgentStep = {
  tool: 'search' | 'view_page';
  query?: string;
  results?: AgentPage[];
  page?: AgentPage;
  refused?: string;
  error?: string;
  ms?: number;
};
export type AgentQuestion = {
  run_id: string;
  dataset: string;
  query_id: number;
  query: string;
  reference: string | null;
  query_types: string[];
  gold: AgentPage[];
  steps: AgentStep[];
  ranking: AgentPage[];
  agent_ranked: number;
  metrics: Metrics | null;
  cited: AgentPage[];
  mode: {
    answer: string | null;
    verdict: Verdict | null;
    correct: boolean | null;
    explanation?: string;
    error?: string;
    usage?: { answer: Usage & { turns?: number }; judge?: Usage };
  };
  prev: number | null;
  next: number | null;
};
export type OurAgentRow = {
  run_id: string;
  dataset: string;
  n: number;
  models: { answer: string; judge: string };
  ndcg_10: number | null;
  accuracy: number | null;
  all?: { ndcg_10?: Mean; accuracy?: Mean };
};
