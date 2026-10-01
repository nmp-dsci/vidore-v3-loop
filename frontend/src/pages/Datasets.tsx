import { useEffect, useMemo, useState } from 'react';
import { Link, useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import {
  type DatasetCard,
  type DatasetDetail,
  type Question,
  type QueryRow,
  datasetLabel,
  fmtN,
  pageImage,
  useGet,
} from '../lib/api';
import { NO_FILTERS, facet, filterQueries, filtersFrom, filtersTo, type Filters } from '../lib/filter';
import { CheckChip, DatasetChips, Kpi, Loading } from '../lib/ui';
import { PageView } from '../lib/page';

/** `/datasets` — the five English sets, one click into each. */
export function Datasets() {
  const { data, error } = useGet<DatasetCard[]>('/api/datasets');
  return (
    <>
      <p className="label">Datasets & questions</p>
      <h1>
        Pick a corpus, then read its questions <em>against the evidence</em>
      </h1>
      <p className="lead">
        English queries only. Each question opens beside its list with the reference answer, the human answers it was
        merged from, and every relevant page with the annotators' boxes drawn on it.
      </p>
      <DatasetChips />
      {!data ? (
        <Loading error={error} />
      ) : (
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>dataset</th>
                <th>domain</th>
                <th className="num">English questions</th>
                <th className="num">documents</th>
                <th className="num">pages</th>
                <th className="num">answerable blind</th>
                <th>pinned revision</th>
              </tr>
            </thead>
            <tbody>
              {data.map((d) => (
                <tr key={d.key}>
                  <td className="sub">
                    <Link to={`/datasets/${d.key}`}>{d.key}</Link>
                  </td>
                  <td>{d.domain}</td>
                  <td className="num">{fmtN(d.queries)}</td>
                  <td className="num">{d.documents}</td>
                  <td className="num">{fmtN(d.pages)}</td>
                  <td className="num">{d.answerable_blind_pct}%</td>
                  <td className="mono small">{d.revision.slice(0, 7)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

/** `/datasets/:key[/:qid]` — the question list, filters in the URL, one question open beside it. */
export function Dataset() {
  const { key = '', qid } = useParams();
  const [sp] = useSearchParams();
  const nav = useNavigate();
  const loc = useLocation();
  const detail = useGet<DatasetDetail>(`/api/datasets/${key}`);
  const list = useGet<QueryRow[]>(`/api/datasets/${key}/queries`);
  const filters = filtersFrom(sp);
  const rows = useMemo(() => (list.data ? filterQueries(list.data, filters) : []), [list.data, sp.toString()]); // eslint-disable-line react-hooks/exhaustive-deps
  const open = qid ? Number(qid) : null;
  const setFilters = (f: Filters) => nav(`/datasets/${key}${open != null ? `/${open}` : ''}${filtersTo(f)}`, { replace: true });
  const go = (id: number) => nav(`/datasets/${key}/${id}${loc.search}`);

  // keep the open question in view in its list, on a deep link as much as on j / k
  useEffect(() => {
    document.querySelector('.qrow.on')?.scrollIntoView({ block: 'nearest' });
  }, [open, list.data]);

  // j / k step through the filtered list, so a reviewer never reaches for the mouse
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest('input, select, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key !== 'j' && e.key !== 'k') return;
      if (!rows.length) return;
      const i = rows.findIndex((r) => r.query_id === open);
      const next = e.key === 'j' ? Math.min(rows.length - 1, i + 1) : Math.max(0, i - 1);
      go(rows[i < 0 ? 0 : next].query_id);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const all = list.data ?? [];
  const types = facet(all, (q) => q.query_types);
  const contents = facet(all, (q) => q.content_type);
  const formats = facet(all, (q) => [q.query_format]);
  const authors = facet(all, (q) => [q.query_generator]);
  const pos = rows.findIndex((r) => r.query_id === open);

  return (
    <>
      <div className="crumbs">
        <Link to="/datasets">Datasets</Link> / {key}
      </div>
      <DatasetChips current={key} />
      {detail.error && <Loading error={detail.error} />}
      {detail.data && <DatasetHead d={detail.data} />}

      <div className="review">
        <section className="qlist" aria-label="questions">
          <div className="filters">
            <input
              type="search"
              placeholder="Search the questions, or a query id"
              value={filters.text}
              onChange={(e) => setFilters({ ...filters, text: e.target.value })}
              aria-label="search questions"
            />
            <Pick label="type" value={filters.type} opts={types} onChange={(v) => setFilters({ ...filters, type: v })} />
            <Pick
              label="content"
              value={filters.content}
              opts={contents}
              onChange={(v) => setFilters({ ...filters, content: v })}
            />
            <Pick label="format" value={filters.format} opts={formats} onChange={(v) => setFilters({ ...filters, format: v })} />
            <Pick label="author" value={filters.author} opts={authors} onChange={(v) => setFilters({ ...filters, author: v })} />
            <Pick
              label="hand check"
              value={filters.check}
              opts={[
                ['any', all.filter((q) => q.hand_check).length],
                ['holds', all.filter((q) => q.hand_check === 'holds').length],
                ['off', all.filter((q) => q.hand_check === 'off').length],
              ]}
              onChange={(v) => setFilters({ ...filters, check: v })}
            />
            <span className="count">
              {rows.length} of {all.length}
            </span>
            {sp.toString() && (
              <button type="button" className="linkish small" onClick={() => setFilters(NO_FILTERS)}>
                clear
              </button>
            )}
          </div>
          {!list.data ? (
            <Loading error={list.error} what="questions (the first load of a dataset takes a few seconds)" />
          ) : (
            <ol className="qrows">
              {rows.map((q) => (
                <li key={q.query_id}>
                  <Link
                    to={`/datasets/${key}/${q.query_id}${loc.search}`}
                    className={`qrow ${q.query_id === open ? 'on' : ''}`}
                    aria-current={q.query_id === open ? 'true' : undefined}
                  >
                    <span className="mono qid">q{q.query_id}</span>
                    <span className="qtext">{q.query}</span>
                    <span className="qmeta">
                      {q.query_types.join(' · ')} · {q.content_type.filter((c) => !c.startsWith('N/A')).join(', ')} ·{' '}
                      {q.n_pages} {q.n_pages === 1 ? 'page' : 'pages'}
                      {q.hand_check && <> · reference {q.hand_check}</>}
                    </span>
                  </Link>
                </li>
              ))}
            </ol>
          )}
        </section>

        <section className="qopen" aria-label="the open question">
          {open == null ? (
            <p className="empty">
              Open a question on the left. <span className="mono">j</span> and <span className="mono">k</span> step
              through the filtered list.
            </p>
          ) : (
            <QuestionView
              k={key}
              qid={open}
              pos={pos >= 0 ? `${pos + 1} of ${rows.length}` : null}
              prev={pos > 0 ? rows[pos - 1].query_id : null}
              next={pos >= 0 && pos < rows.length - 1 ? rows[pos + 1].query_id : null}
            />
          )}
        </section>
      </div>
    </>
  );
}

function Pick({
  label,
  value,
  opts,
  onChange,
}: {
  label: string;
  value: string;
  opts: [string, number][];
  onChange: (v: string) => void;
}) {
  return (
    <label className="pick">
      <span className="sr">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)} aria-label={label}>
        <option value="">any {label}</option>
        {opts.map(([v, n]) => (
          <option key={v} value={v}>
            {v} ({n})
          </option>
        ))}
      </select>
    </label>
  );
}

function DatasetHead({ d }: { d: DatasetDetail }) {
  const s = d.summary;
  const human = s.by_generator.find(([g]) => g === 'human')?.[1] ?? 0;
  return (
    <>
      <h1 className="dshead">
        {datasetLabel(d.key)}: {fmtN(s.n_queries)} English questions over <em>{fmtN(d.pages)} pages</em>
      </h1>
      <div className="kpis">
        <Kpi n={fmtN(s.n_queries)} b={`English questions; ${human} written by people, ${s.n_queries - human} synthetic`} />
        <Kpi n={String(s.pages_per_query.median)} b={`relevant pages per question, median (mean ${s.pages_per_query.mean}, max ${s.pages_per_query.max})`} />
        <Kpi n={String(d.documents.length)} b={`documents, ${fmtN(d.pages)} pages`} />
        <Kpi
          n={`${s.hand_checked}`}
          b={`references hand-checked of ${fmtN(s.n_queries)}; the full audit is M4`}
          tone={s.hand_checked ? 'warn' : undefined}
        />
      </div>
      <details className="docs">
        <summary>
          The {d.documents.length} documents, their licences, and the dataset's revision
        </summary>
        <p className="small muted">
          <code>{d.repo_id}</code> at <code>{d.revision.slice(0, 7)}</code>; the paper's end-to-end runs used{' '}
          <code>{d.paper_revision.slice(0, 7)}</code>.
        </p>
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>document</th>
                <th>type</th>
                <th>year</th>
                <th className="num">pages</th>
                <th>visual content</th>
                <th>licence</th>
              </tr>
            </thead>
            <tbody>
              {d.documents.map((doc) => (
                <tr key={doc.doc_id}>
                  <td className="sub">
                    <a href={doc.url} target="_blank" rel="noreferrer">
                      {doc.doc_id}
                    </a>
                  </td>
                  <td>{doc.doc_type}</td>
                  <td className="mono">{doc.doc_year}</td>
                  <td className="num">{doc.pages}</td>
                  <td>{doc.visual_types.join(', ')}</td>
                  <td className="wrap small">{doc.license.length > 90 ? `${doc.license.slice(0, 88)}…` : doc.license}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}

function QuestionView({
  k,
  qid,
  pos,
  prev,
  next,
}: {
  k: string;
  qid: number;
  pos: string | null;
  prev: number | null;
  next: number | null;
}) {
  const loc = useLocation();
  const { data: q, error } = useGet<Question>(`/api/datasets/${k}/queries/${qid}`);
  const [page, setPage] = useState<number | null>(null);
  useEffect(() => setPage(null), [k, qid]);
  if (!q) return <Loading error={error} what={`question ${qid}`} />;
  const current = q.pages.find((p) => p.corpus_id === page) ?? q.pages[0];
  const raws = q.raw_answers.filter((r) => r.trim());
  return (
    <article className="question">
      <div className="qnav">
        <span className="label">
          q{q.query_id} {pos && <>· {pos} in this list</>}
        </span>
        <span className="qnav-btns">
          {prev != null ? <Link to={`/datasets/${k}/${prev}${loc.search}`}>← previous (k)</Link> : <span />}
          {next != null && <Link to={`/datasets/${k}/${next}${loc.search}`}>next (j) →</Link>}
        </span>
      </div>
      <div className="row runrow">
        <Link className="btn" to={`/pipeline?dataset=${k}&qid=${q.query_id}`}>
          Run retrieval on this question →
        </Link>
        <span className="small muted">opens the RAG pipeline tab with every stage scored against these gold pages</span>
      </div>
      <p className="qbig">{q.query}</p>
      <div className="chips">
        {q.query_types.map((t) => (
          <span key={t} className="chip">
            {t}
          </span>
        ))}
        <span className="chip">{q.query_format}</span>
        {q.content_type
          .filter((c) => !c.startsWith('N/A'))
          .map((c) => (
            <span key={c} className="chip">
              {c.toLowerCase()}
            </span>
          ))}
        <span className="chip">{q.query_generator === 'human' ? 'written by a person' : 'synthetic (sdg)'}</span>
        <CheckChip verdict={q.hand_check?.verdict} />
      </div>

      <dl className="answer">
        <dt>reference answer</dt>
        <dd className="ref">{q.answer}</dd>
        {raws.length > 0 && (
          <>
            <dt>
              {raws.length === 1 ? 'the human answer' : `the ${raws.length} human answers`} it was merged from
            </dt>
            <dd>
              <ol className="raws">
                {raws.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ol>
            </dd>
          </>
        )}
        {q.hand_check && (
          <>
            <dt>hand check · {q.hand_check.date}</dt>
            <dd className={`check ${q.hand_check.verdict}`}>
              <b>{q.hand_check.verdict === 'holds' ? 'Holds.' : `Off. On the page: ${q.hand_check.checked_answer}`}</b>{' '}
              {q.hand_check.note}
            </dd>
          </>
        )}
        {q.query_generator !== 'human' && (
          <>
            <dt>how it was generated</dt>
            <dd className="small">
              {[q.query_generation_pipeline, q.query_type_for_generation, q.source_type && `from ${q.source_type}`]
                .filter(Boolean)
                .join(' · ')}
            </dd>
          </>
        )}
      </dl>

      <h3>
        Evidence — {q.pages.length} relevant {q.pages.length === 1 ? 'page' : 'pages'}, {q.n_full} with the full
        answer
      </h3>
      <div className="thumbs" role="tablist" aria-label="relevant pages">
        {q.pages.map((p) => (
          <button
            key={p.corpus_id}
            type="button"
            role="tab"
            aria-selected={p.corpus_id === current?.corpus_id}
            className={`thumb ${p.corpus_id === current?.corpus_id ? 'on' : ''}`}
            onClick={() => setPage(p.corpus_id)}
          >
            <img src={pageImage(k, p.corpus_id)} alt={`page ${p.corpus_id}`} loading="lazy" />
            <span className={`grade g${p.score}`}>{p.score === 2 ? 'full · 2' : 'part · 1'}</span>
          </button>
        ))}
      </div>
      {current && <PageView k={k} corpusId={current.corpus_id} boxes={current.boxes} grade={current.score} />}
    </article>
  );
}
