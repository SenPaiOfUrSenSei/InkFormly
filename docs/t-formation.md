# Internal lowercase t refinement

## Historical stroke edits: withdrawn

The current implementation regenerates complete words at verified boundaries;
see [word clarity](word-clarity.md). The historical stroke surgery below remains
excluded from generation, including the current `--t-refinement` option.

User visual review found the higher-crossbar output worse, including its
surrounding writing. Crossbar raising and stem narrowing were removed from the
production generation path. The following describes
the rejected revisions and their historical checks, not validated improvements.

The crossing detector searched following motion without establishing exclusive
character ownership. Moving a connected crossing could bend adjoining motion.
Even unchanged raw points can move slightly because the renderer recalculates
line tilt, baseline and fitting. Comparison of the saved versions measured such
layout displacement above .01 mm on 22 of 37 changed lines, up to .038 mm;
this small displacement does not by itself explain the visual regression.
Unchanged readback scores did not detect the shape damage. The original
pre-adjustment essay trajectories have been restored with the same writer,
writing size, variation, ink and layout settings. No neural weights were changed.

The stem-only experiment attempts to correct an overly wide triangular loop in a sampled
lowercase `t` inside a word. Frozen model attention aligns the trajectory to the
text. A bounded correction moves only the rising stem's horizontal coordinates
toward its own descending stem, tapering to zero at the endpoints. It requires
an existing crossing stroke, sufficient alignment coverage and consistent stem
geometry. It preserves downstrokes, crossbars, joins, vertical positions, pen
lifts and other characters. Capital `T` and word-initial `t` are excluded.

The subsequent higher-crossbar update identifies existing near-horizontal
crossings after these stems and raises low ones by at most 18.7% of stem height
at normal strength. Each crossing follows its own stem's lean. A flat central
translation and smooth endpoint transitions preserve the original joins. Shared
crossings are moved once, and already high or ambiguous bars are skipped. No
stroke is inserted. This stage intentionally changes the crossbar's vertical
position; the stem-only invariants above describe the preceding stage.

After correction, blind TrOCR readback must rank at least as well as the original
checked output, prioritizing matching numbers and then edit distance. A gentler
correction is tried if needed; otherwise the original is retained. No fixed
glyph or missing crossbar is inserted. Ambiguous or more severely malformed
stems can remain unchanged. This is decoder geometry, not neural weight training.

## Measured checks

On six held-out sentences with two fixed seeds, the correction changed 11 of
12 lines. Exact readbacks rose from 9/12 to 10/12 and total character differences
fell from 3 to 2. No selected line's readback regressed. Two no-t controls and
an uppercase-only control were unchanged. These small selection-readback checks
are not independent human legibility or realism measurements. Paired images
and raw trajectories are in `artifacts/evaluation/t-stem-refinement/`.
These measurements describe the stem-only revision, before the higher-crossbar
update. The new essay's separate audit records its accepted crossing movements
and before/after wording checks.

The higher-crossbar essay accepts 44 raised crossings across 37 lines, with
92 exact line readbacks before and after. Every accepted line ranks no worse
than its preceding version. All seven revised pages and close before/after
samples were visually inspected. The non-API regression suite passes 58 tests;
crossbar tests cover bounds, unchanged stems/endpoints/pen lifts, duplicate
crossing protection, missing/high bars and zero strength.

The revised seven-page college essay retains its source text, paragraph breaks,
writer reference, larger writing, lean adjustment, ink and varied layout settings.
It contains 54 corrected t stems. Its 92 exact line readbacks are unchanged;
remaining review flags are retained. All seven final PDF pages were rendered
with Poppler and visually inspected. The audit is stored alongside the PDF in
`output/pdf/college-essay-clear-t/verification.json`.

## Rejected experiments

A small 8,473-parameter residual adapter was trained with the base model frozen
on existing real IAM writer references. Validation likelihood improved slightly,
but raw held-out readbacks regressed (4/12 to 2/12 exact; character differences
68 to 83). It is not loaded in production. Stronger local sampling precision
also produced mixed results. A model-contrast reranker rarely selected a new
line and did not reliably select a better-looking t. Those reproducible research
paths remain explicitly experimental; their results do not support claiming
successful pretrained-weight fine-tuning.

Historical regression checks cover isolated geometry, preserved non-target
strokes, missing-crossbar rejection, initial/capital exclusions, gentler fallback,
readback rejection and backward-compatible saved-document policy. Additional
rollback checks ensure default checked generation bypasses stroke surgery and
even explicit stem experiments cannot call crossbar raising. Old documents do
not acquire new corrections on rerendering.
