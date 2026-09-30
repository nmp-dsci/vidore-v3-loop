# DESIGN.md — the brief vidore-v3-loop is built to

Anything visual here follows the portfolio's "Field Guide" brief: the viewer
(`frontend/`), the Lavish plans in `.lavish/`, and README figures. The brief is
`nmp-dsci.github.io/DESIGN.md`; tau2-loop's `DESIGN.md` is the fullest port of it.
This file records only what is specific to this project. If it disagrees with the
site's `assets/css/tokens.css`, the site wins, apart from the divergences below.

## Divergences from the site palette (recorded 2026-09-30, plan s00 §7b)

| token | light | dark | means |
|---|---|---|---|
| `--accent` | `#A8440B` | `#F2945A` | **orange, the project colour**: act on this, links, the recommended option, the current tab. 5.55:1 on `--bg` (light), 7.96:1 (dark) |
| `--ok` | `#0A6552` | `#43C29A` | passed / holds: a reference that checks out, a human evidence box, grade 2 |
| `--amber` | `#6F5A00` | `#CDB04A` | partial / caveat: a reference that is off, a provisional number. Olive, not the site's amber, so it never reads as the orange accent |

The tokens themselves are in `frontend/src/tokens.css`, and the Lavish artifacts
paste the same block.

## Rules that bite here

- Headings state claims ("Finance: 309 English questions over 2,942 pages"), never topics.
- Every number carries its denominator ("1 of 309 references hand-checked").
- A verdict is a word plus a glyph (`● reference holds`, `◐ reference off`), never colour alone.
- Human evidence boxes are `--ok`. Each annotator gets its own stroke pattern (solid, dashed, dotted), not its own colour.
- `npm run lint:design` fails on a hex colour outside `tokens.css` or a font under 12.8px.
