# Research sources (2026-09-28)

The committed sources behind the numbers in `.lavish/s00_vidore-v3-research-plan.html`.

- `mteb_vidore3_scores_2026-09-28.json` — every model's ViDoRe V3 score from
  [embeddings-benchmark/results](https://github.com/embeddings-benchmark/results) at the
  2026-09-26 commit. Key `<model>|<revision>|<v3|v3.1>` → task → `[mean NDCG@10 over the six
  query languages, English-query NDCG@10, languages, results commit, mteb version]`.
- `demo_questions.json` — the one English demo question per dataset shown in the plan, its
  relevant pages (corpus id, grade, doc, page, box count) and a hand check of each reference
  answer against the page (`_checks`).
- Paper tables (retrieval, rerankers, end-to-end accuracy, grounding) are quoted from
  [arXiv 2601.08620 v2](https://arxiv.org/html/2601.08620); the pipeline leaderboard from
  [illuin-tech/vidore-benchmark results/metrics](https://github.com/illuin-tech/vidore-benchmark/tree/main/results/metrics).
