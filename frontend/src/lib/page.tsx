/** One page, large: the image with the annotators' boxes drawn on it, and its OCR text beside it on demand. */
import { useState } from 'react';
import { type Box, type PageMeta, pageImage, useGet } from './api';
import { boxRects } from './filter';
import { Loading } from './ui';

export function PageView({
  k,
  corpusId,
  boxes,
  grade,
}: {
  k: string;
  corpusId: number;
  boxes: Box[];
  grade: number | null;
}) {
  const { data: meta, error } = useGet<PageMeta>(`/api/pages/${k}/${corpusId}`);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showText, setShowText] = useState(false);
  const rects = boxRects(boxes);
  const annotators = [...new Set(rects.map((r) => r.annotator))];
  if (!meta)
    return (
      <Loading
        error={error}
        what={`page ${corpusId} (the first view of a part of the corpus fetches ~230 pages from the Hub: about 15 s)`}
      />
    );
  return (
    <div className="pageview">
      <div className="row pv-bar">
        <span className="mono small">
          {meta.doc_id} · p.{meta.page} · corpus {meta.corpus_id} ·{' '}
          {grade === 2 ? 'grade 2: the full answer' : grade === 1 ? 'grade 1: a required part' : grade === 0 ? 'not a gold page' : 'no gold pages (free-text question)'}
        </span>
        <button type="button" className={`tog ${showBoxes ? 'on' : ''}`} aria-pressed={showBoxes} onClick={() => setShowBoxes((v) => !v)}>
          <span>
            {rects.length ? `boxes · ${rects.length} from ${annotators.length} ${annotators.length === 1 ? 'annotator' : 'annotators'}` : 'no human boxes on this page'}
          </span>
        </button>
        <button type="button" className={`tog ${showText ? 'on' : ''}`} aria-pressed={showText} onClick={() => setShowText((v) => !v)}>
          <span>OCR text</span>
        </button>
        <a href={pageImage(k, corpusId)} target="_blank" rel="noreferrer" className="small">
          open image
        </a>
      </div>
      <div className={`pv-body ${showText ? 'split' : ''}`}>
        <div className="pv-page">
          <img src={pageImage(k, corpusId)} alt={`${meta.doc_id} page ${meta.page}`} />
          {showBoxes && (
            <svg viewBox={`0 0 ${meta.width} ${meta.height}`} preserveAspectRatio="none" aria-hidden="true">
              {rects.map((r, i) => (
                <rect
                  key={i}
                  x={r.x}
                  y={r.y}
                  width={r.w}
                  height={r.h}
                  className={`hbox a${annotators.indexOf(r.annotator) % 3}`}
                  vectorEffect="non-scaling-stroke"
                />
              ))}
            </svg>
          )}
        </div>
        {showText && (
          <div className="pv-text">
            <span className="label">OCR markdown · what a text-only system sees</span>
            <pre>{meta.markdown || '(empty)'}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
