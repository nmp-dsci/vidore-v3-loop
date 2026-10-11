import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import {
  type AnswerQuestion,
  type AnswerRow,
  type AnswerRun,
  type GivenPage,
  type Mean,
  MODES,
  MODE_LABEL,
  type Mode,
  type ModeAnswer,
  type Question,
  type Verdict,
  get,
  pageImage,
  pct1,
  post,
  useGet,
} from '../lib/api';
import { PageView } from '../lib/page';
import { SetupTable } from '../lib/setup';
import { Busy, Kpi, Loading } from '../lib/ui';

const when = (iso: string) => iso.replace('T', ' ').slice(0, 16);
const pctMean = (m: Mean | undefined) => (m?.mean == null ? '—' : `${pct1(m.mean)}%`);
const modesOf = (r: AnswerRun): Mode[] => MODES.filter((m) => r.config.modes.includes(m));

/** An answer's text, with its markdown **bold** shown as bold and nothing else interpreted. */
export function AnswerText({ text }: { text: string | null }) {
  const parts = (text ?? '').split(/(\*\*[^*]+\*\*)/g);
  return (
    <p className="answer-text">
      {parts.map((p, i) => (p.startsWith('**') && p.endsWith('**') ? <b key={i}>{p.slice(2, -2)}</b> : p))}
    </p>
  );
}

/** A verdict as a glyph and a word, never colour alone. */
export function VerdictChip({ v, person }: { v: Verdict | null | undefined; person?: boolean | null }) {
  const overruled = person != null && v != null && person !== (v === 'Correct');
  const chip =
    v === 'Correct' ? (
      <span className="chip ok">● correct</span>
    ) : v === 'Partially Correct' ? (
      <span className="chip warn">◐ partial</span>
    ) : v === 'Incorrect' ? (
      <span className="chip bad">✕ incorrect</span>
    ) : (
      <span className="chip">— error</span>
    );
  return (
    <span className="verdict">
      {chip}
      {overruled && <span className="small muted"> you: {person ? 'correct' : 'not correct'}</span>}
    </span>
  );
}

function Glyph({ m }: { m: AnswerRow['modes'][Mode] }) {
  if (!m) return <span className="muted">·</span>;
  const g = m.verdict === 'Correct' ? '●' : m.verdict === 'Partially Correct' ? '◐' : m.verdict ? '✕' : '—';
  const cls = m.verdict === 'Correct' ? 'good' : m.verdict === 'Partially Correct' ? 'poor' : '';
  const over = m.person != null && m.person !== m.correct;
  return (
    <span className={`glyph ${cls}`} title={`${m.verdict ?? m.error ?? ''}${over ? ' · you overruled the judge' : ''}`}>
      {g}
      {over && <sup>you {m.person ? '●' : '✕'}</sup>}
    </span>
  );
}

function RunStatus({ r }: { r: AnswerRun }) {
  if (r.status === 'running')
    return (
      <span className="status warn pulsing">
        running {r.progress.done}/{r.progress.n}
      </span>
    );
  if (r.status === 'finished') return <span className="status ok">finished</span>;
  return (
    <span className="status err" title={r.error ?? ''}>
      {r.status}
    </span>
  );
}

/** `/review` — every answer run: its retrieved score, its ceiling and the gap between them. */
export function Review() {
  const { data: rs, error } = useGet<AnswerRun[]>('/api/runs?kind=answer');
  if (!rs) return <Loading error={error} what="the answer runs" />;
  return (
    <>
      <p className="label">Review · answers judged against the gold answer (M4)</p>
      <h1>
        Claude's answers, <em>judged and audited</em>
      </h1>
      <p className="lead">
        Each run answers every question from our top 5 pages and from the human-marked pages, and a judge scores each
        answer against the gold answer.
      </p>
      {rs.length === 0 ? (
        <p className="empty">
          No answer runs yet: start one with <code>make answers DATASET=finance_en LIMIT=5</code>.
        </p>
      ) : (
        <div className="tw">
          <table className="runs fit">
            <thead>
              <tr>
                <th rowSpan={2}>run</th>
                <th rowSpan={2}>split</th>
                <th rowSpan={2} className="num">
                  questions
                </th>
                <th colSpan={MODES.length} className="group">
                  % correct, audited
                </th>
                <th rowSpan={2} className="num">
                  retrieval cost
                </th>
                <th rowSpan={2}>status</th>
              </tr>
              <tr>
                {MODES.map((m) => (
                  <th key={m} className="num" title={MODE_LABEL[m][1]}>
                    {MODE_LABEL[m][0]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rs.map((r) => {
                const a = r.audited ?? r.summary ?? {};
                const cost = a.retrieval_cost?.audited;
                return (
                  <tr key={r.run_id}>
                    <td className="runcell">
                      <Link className="nw" to={`/review/${r.run_id}`}>
                        {when(r.started_at)}
                      </Link>
                      <div className="small muted">
                        {r.config.note || r.label} · {r.models.answer}
                      </div>
                    </td>
                    <td>
                      <span className={`chip ${r.split === 'test' ? 'warn' : ''}`}>
                        {r.split}
                        {r.final ? ' · final' : ''}
                      </span>
                    </td>
                    <td className="num">{r.n}</td>
                    {MODES.map((m) => (
                      <td key={m} className="num" title={`raw (the judge alone): ${pctMean(a[m]?.raw)}`}>
                        {pctMean(a[m]?.audited)}
                      </td>
                    ))}
                    <td className="num">{cost == null ? '—' : `${(cost * 100).toFixed(1)} pts`}</td>
                    <td>
                      <RunStatus r={r} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="small muted">
        Retrieval cost is the oracle's accuracy minus the retrieved one: what our retrieval loses.
      </p>
    </>
  );
}

type Lens = 'all' | 'retrieval' | 'generation' | 'disputed' | 'labelled';
const LENSES: [Lens, string, string][] = [
  ['all', 'all', 'every question'],
  ['retrieval', 'retrieval misses', 'right from the human pages, not from ours'],
  ['generation', 'generation misses', 'not right even from the human pages'],
  ['disputed', 'disputed', 'the gold answer is disputed'],
  ['labelled', 'labelled', 'a person has checked the judge'],
];
const right = (m: AnswerRow['modes'][Mode]) => (m ? (m.person ?? m.correct) === true : false);

function inLens(r: AnswerRow, lens: Lens): boolean {
  switch (lens) {
    case 'retrieval':
      return right(r.modes.oracle) && !right(r.modes.retrieved);
    case 'generation':
      return !!r.modes.oracle && !right(r.modes.oracle);
    case 'disputed':
      return r.reference === 'disputed';
    case 'labelled':
      return Object.values(r.modes).some((m) => m?.person != null);
    default:
      return true;
  }
}

/** `/review/:runId` — one answer run: its three scores, the judge's calibration, every question. */
export function ReviewRun() {
  const { runId = '' } = useParams();
  const [r, setR] = useState<AnswerRun | null>(null);
  const [rows, setRows] = useState<AnswerRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const load = () =>
      Promise.all([get<AnswerRun>(`/api/runs/${runId}`), get<AnswerRow[]>(`/api/runs/${runId}/answers`)])
        .then(([doc, rs]) => {
          if (!alive) return;
          setR(doc);
          setRows(rs);
          if (doc.status === 'running') timer = window.setTimeout(load, 5000);
        })
        .catch((e: Error) => alive && setErr(e.message));
    load();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [runId]);
  if (!r || !rows) return <Loading error={err} what="the run" />;
  const modes = modesOf(r);
  const a = r.audited ?? r.summary ?? {};
  const ret = a.retrieved;
  const cal = r.calibration;
  return (
    <>
      <p className="label">
        <Link to="/review">Review</Link> / {r.run_id}
      </p>
      <h1 className="qh1">
        {r.dataset} · {r.split} · {r.n} questions answered by <em>{r.models.answer}</em>
      </h1>
      {r.status === 'running' && (
        <Busy
          label={`Answering ${r.progress.done} of ${r.progress.n} questions`}
          detail="This page refreshes every 5 s while the run goes."
        />
      )}
      <SetupTable setup={r.setup} />
      <h2>Scores</h2>
      <div className="kpis">
        {modes.map((m) => (
          <Kpi
            key={m}
            n={pctMean(a[m]?.audited)}
            b={`${MODE_LABEL[m][0]}: ${MODE_LABEL[m][1]} (raw ${pctMean(a[m]?.raw)})`}
            tone={m === 'retrieved' ? 'ok' : undefined}
          />
        ))}
        {a.retrieval_cost?.audited != null && (
          <Kpi
            n={`${(a.retrieval_cost.audited * 100).toFixed(1)} pts`}
            b="retrieval cost: oracle minus retrieved"
            tone="warn"
          />
        )}
      </div>
      {ret?.full_given && (
        <p className="small muted">
          A full-answer page was in our top 5 for {pctMean(ret.full_given)} of questions, answered right{' '}
          {pctMean(ret.correct_when_full_given)} of the time
          {ret.correct_when_full_missed?.n ? `, against ${pctMean(ret.correct_when_full_missed)} without one` : ''}.
        </p>
      )}
      {cal && (
        <p className="small rule">
          <b>Judge check:</b> {cal.labelled} of {cal.target} answers labelled by a person
          {cal.labelled ? `, agreeing with the judge on ${cal.agree}` : ''} · {r.disputed ?? 0} gold answers disputed.
        </p>
      )}

      <h2>Verdicts per setting</h2>
      <div className="tw">
        <table className="metrics fit">
          <thead>
            <tr>
              <th>setting</th>
              <th className="num">● correct</th>
              <th className="num">◐ partial</th>
              <th className="num">✕ incorrect</th>
              <th className="num">errors</th>
              <th className="num">tokens per answer</th>
              <th className="num">time per answer</th>
            </tr>
          </thead>
          <tbody>
            {modes.map((m) => {
              const s = a[m];
              return (
                <tr key={m}>
                  <td className="sub">{MODE_LABEL[m][0]}</td>
                  <td className="num">{s?.verdicts.Correct ?? '—'}</td>
                  <td className="num">{s?.verdicts['Partially Correct'] ?? '—'}</td>
                  <td className="num">{s?.verdicts.Incorrect ?? '—'}</td>
                  <td className="num">{s?.errors ?? '—'}</td>
                  <td className="num">{s?.answer_tokens?.toLocaleString('en-AU') ?? '—'}</td>
                  <td className="num">{s?.ms == null ? '—' : `${(s.ms / 1000).toFixed(1)} s`}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {r.totals && (
        <p className="small muted">
          {r.totals.calls} Claude calls and {r.totals.input_tokens.toLocaleString('en-AU')} input tokens on the
          subscription (≈${r.totals.cost_usd} at per-token prices).
        </p>
      )}
      <AnswerTable r={r} rows={rows} modes={modes} />
    </>
  );
}

function AnswerTable({ r, rows, modes }: { r: AnswerRun; rows: AnswerRow[]; modes: Mode[] }) {
  const [sp, setSp] = useSearchParams();
  const nav = useNavigate();
  const lens = (sp.get('lens') as Lens | null) ?? 'all';
  const filter = sp.get('f') ?? '';
  const shown = useMemo(() => {
    const f = filter.trim().toLowerCase();
    return rows.filter(
      (q) =>
        inLens(q, lens) &&
        (!f ||
          String(q.query_id) === f ||
          q.query.toLowerCase().includes(f) ||
          q.query_types.some((t) => t.includes(f))),
    );
  }, [rows, lens, filter]);
  const set = (k: string, v: string) => {
    const next = new URLSearchParams(sp);
    if (v && v !== 'all') next.set(k, v);
    else next.delete(k);
    setSp(next, { replace: true });
  };
  return (
    <>
      <h2>Every question ({rows.length})</h2>
      <div className="seg" role="group" aria-label="lens">
        {LENSES.map(([k, label, what]) => (
          <button
            key={k}
            type="button"
            className={`tog ${lens === k ? 'on' : ''}`}
            aria-pressed={lens === k}
            title={what}
            onClick={() => set('lens', k)}
          >
            <span>
              {label} {rows.filter((q) => inLens(q, k)).length}
            </span>
          </button>
        ))}
      </div>
      <div className="row">
        <input
          type="text"
          className="grow"
          value={filter}
          placeholder="filter: words, a query id or a type"
          onChange={(e) => set('f', e.target.value)}
          aria-label="filter questions"
        />
        <span className="small muted">{shown.length} shown</span>
      </div>
      <div className="tw">
        <table className="qtable fit">
          <thead>
            <tr>
              <th>q</th>
              <th>question</th>
              {modes.map((m) => (
                <th key={m} className="num" title={MODE_LABEL[m][1]}>
                  {MODE_LABEL[m][0]}
                </th>
              ))}
              <th>gold answer</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((q) => (
              <tr key={q.query_id} className="click" onClick={() => nav(`/review/${r.run_id}/${q.query_id}`)}>
                <td className="mono">
                  <Link to={`/review/${r.run_id}/${q.query_id}`} onClick={(e) => e.stopPropagation()}>
                    q{q.query_id}
                  </Link>
                </td>
                <td className="q">
                  {q.query}
                  <div className="small muted">{q.query_types.join(' · ')}</div>
                </td>
                {modes.map((m) => (
                  <td key={m} className="num">
                    <Glyph m={q.modes[m]} />
                    {m === 'retrieved' && q.modes[m]?.full_given === false && q.n_full > 0 && (
                      <div className="small muted" title="no full-answer page in our top 5">
                        no full page
                      </div>
                    )}
                  </td>
                ))}
                <td>{q.reference === 'disputed' ? <span className="chip warn">◐ disputed</span> : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">● correct · ◐ partial · ✕ incorrect; only correct counts, as in the paper.</p>
    </>
  );
}

/** `/review/:runId/:qid` — one question: the reference, each setting's answer and verdict, and the audit. */
export function ReviewQuestion() {
  const { runId = '', qid = '' } = useParams();
  const nav = useNavigate();
  const [v, setV] = useState(0); // bumped after an audit, to read it back
  const { data: q, error } = useGet<AnswerQuestion>(`/api/runs/${runId}/answers/${qid}?v=${v}`);
  const [open, setOpen] = useState<number | null>(null);
  useEffect(() => setOpen(null), [qid]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!q || (e.target as HTMLElement).closest('input, textarea, select')) return;
      if (e.key === 'j' && q.next != null) nav(`/review/${runId}/${q.next}`);
      if (e.key === 'k' && q.prev != null) nav(`/review/${runId}/${q.prev}`);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [q, nav, runId]);
  const question = useGet<Question>(q ? `/api/datasets/${q.dataset}/queries/${q.query_id}` : null);
  if (!q) return <Loading error={error} what="the question" />;
  const modes = MODES.filter((m) => q.modes[m]);
  const boxes = (cid: number) => question.data?.pages.find((p) => p.corpus_id === cid)?.boxes ?? [];
  const grade = (cid: number) => q.gold.find((g) => g.corpus_id === cid)?.grade ?? 0;
  return (
    <>
      <p className="label">
        <Link to="/review">Review</Link> / <Link to={`/review/${runId}`}>{runId}</Link> / q{q.query_id}
      </p>
      <div className="row qnav">
        {q.prev != null && <Link to={`/review/${runId}/${q.prev}`}>← q{q.prev}</Link>}
        {q.next != null && <Link to={`/review/${runId}/${q.next}`}>q{q.next} →</Link>}
        <span className="small muted">
          <span className="mono">j</span> / <span className="mono">k</span> step through the run
        </span>
      </div>
      <h1 className="qh1">{q.query}</h1>
      <div className="row">
        <span className="small muted">{q.query_types.join(' · ')}</span>
        <Link className="btn" to={`/pipeline?dataset=${q.dataset}&qid=${q.query_id}`}>
          <span>Run it live in the RAG pipeline →</span>
        </Link>
      </div>

      <ReferenceCard q={q} onSaved={() => setV(v + 1)} />

      <div className="anscols" style={{ gridTemplateColumns: `repeat(${modes.length}, minmax(0, 1fr))` }}>
        {modes.map((m) => (
          <ModeCard
            key={m}
            runId={runId}
            q={q}
            mode={m}
            a={q.modes[m]!}
            onSaved={() => setV(v + 1)}
            onOpen={setOpen}
            open={open}
          />
        ))}
      </div>

      {open != null && (
        <section className="hitopen" aria-label="the open page">
          <div className="row">
            <span className="label">page · corpus {open}</span>
            <button type="button" className="linkish small" onClick={() => setOpen(null)}>
              close
            </button>
          </div>
          <PageView k={q.dataset} corpusId={open} boxes={boxes(open)} grade={grade(open)} />
        </section>
      )}
    </>
  );
}

function ReferenceCard({ q, onSaved }: { q: AnswerQuestion; onSaved: () => void }) {
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => setNote(''), [q.query_id]);
  const st = q.reference?.status;
  const save = (status: 'disputed' | 'holds') => {
    setBusy(true);
    setErr(null);
    post('/api/audits/reference', { dataset: q.dataset, query_id: q.query_id, status, note })
      .then(onSaved)
      .catch((e: Error) => setErr(e.message))
      .finally(() => setBusy(false));
  };
  return (
    <section className={`qcard refcard ${st === 'disputed' ? 'disputed' : ''}`} aria-label="the gold answer">
      <span className="label">
        gold answer{' '}
        {st === 'disputed' ? (
          <span className="chip warn">◐ disputed</span>
        ) : st === 'holds' ? (
          <span className="chip ok">● holds</span>
        ) : null}
      </span>
      <p className="qbig">{q.answer}</p>
      {q.reference?.note && (
        <p className="small muted">
          {q.reference.source}: {q.reference.note}
        </p>
      )}
      {q.raw_answers.length > 1 && (
        <details className="small">
          <summary>The {q.raw_answers.length} human answers it was merged from</summary>
          <ol>
            {q.raw_answers.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ol>
        </details>
      )}
      <div className="row">
        <input
          type="text"
          className="grow"
          value={note}
          placeholder="why, e.g. the table says 2,008, not 2,070"
          onChange={(e) => setNote(e.target.value)}
          aria-label="audit note"
        />
        <button type="button" className="tog" disabled={busy} onClick={() => save('holds')}>
          <span>● Gold answer holds</span>
        </button>
        <button type="button" className="tog" disabled={busy} onClick={() => save('disputed')}>
          <span>◐ Dispute it</span>
        </button>
      </div>
      {err && <p className="small warn-note">{err}</p>}
    </section>
  );
}

function ModeCard({
  runId,
  q,
  mode,
  a,
  onSaved,
  onOpen,
  open,
}: {
  runId: string;
  q: AnswerQuestion;
  mode: Mode;
  a: ModeAnswer;
  onSaved: () => void;
  onOpen: (cid: number) => void;
  open: number | null;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const label = (correct: boolean) => {
    setBusy(true);
    setErr(null);
    post('/api/audits/judge', { run_id: runId, query_id: q.query_id, mode, correct })
      .then(onSaved)
      .catch((e: Error) => setErr(e.message))
      .finally(() => setBusy(false));
  };
  const you = a.label?.correct;
  return (
    <section className="stagecol anscol" aria-label={MODE_LABEL[mode][0]}>
      <header>
        <b>{MODE_LABEL[mode][0]}</b>
        <span className="small muted">{MODE_LABEL[mode][1]}</span>
        <VerdictChip v={a.verdict} person={you} />
      </header>
      {a.error ? (
        <p className="small warn-note">{a.error}</p>
      ) : (
        <>
          <AnswerText text={a.answer} />
          {a.explanation && (
            <p className="small muted">
              <b>Judge:</b> {a.explanation}
            </p>
          )}
          <div className="row judgerow">
            <span className="small">Your call:</span>
            <button
              type="button"
              className={`tog ${you === true ? 'on' : ''}`}
              aria-pressed={you === true}
              disabled={busy}
              onClick={() => label(true)}
            >
              <span>● correct</span>
            </button>
            <button
              type="button"
              className={`tog ${you === false ? 'on' : ''}`}
              aria-pressed={you === false}
              disabled={busy}
              onClick={() => label(false)}
            >
              <span>✕ not correct</span>
            </button>
          </div>
          {a.label && (
            <p className="small muted">{a.label.agrees ? 'You agree with the judge.' : 'You overruled the judge.'}</p>
          )}
          {err && <p className="small warn-note">{err}</p>}
        </>
      )}
      {a.pages.length > 0 ? (
        <div className="givenpages">
          {a.pages.map((p: GivenPage, i: number) => (
            <button
              key={p.corpus_id}
              type="button"
              className={`given ${open === p.corpus_id ? 'on' : ''}`}
              onClick={() => onOpen(p.corpus_id)}
              title={`${p.doc_id} p.${p.page}`}
            >
              <img src={pageImage(q.dataset, p.corpus_id)} alt="" loading="lazy" />
              <span className={`mono small ${p.grade === 2 ? 'good' : p.grade === 1 ? 'poor' : 'muted'}`}>
                {mode === 'retrieved' ? `#${i + 1} ` : ''}
                {p.grade === 2 ? '● 2' : p.grade === 1 ? '◐ 1' : '○ 0'}
              </span>
            </button>
          ))}
        </div>
      ) : (
        <p className="small muted">No pages given.</p>
      )}
      {a.usage && (
        <p className="mono small muted">
          {a.usage.answer.input_tokens.toLocaleString('en-AU')} tokens · {(a.usage.answer.ms / 1000).toFixed(1)} s
        </p>
      )}
    </section>
  );
}
