# Word-level t generation

The current `closed-word-resampling-v1` policy regenerates complete words with
the existing pretrained network. It retains writer reference 2 and the requested
sampling diversity. No neural weights, glyph templates or per-letter stroke
corrections are introduced.

## Boundaries and placement

Text attention estimates word positions, while actual blank vertical strips
determine possible separation points. A replaceable word must include every
trajectory point attributed to its characters, contain no points attributed to
other non-space characters, form a contiguous temporal block, and have pen lifts
at both ends. Ambiguous ownership skips regeneration. This is especially
important for detached final t strokes: a blank gap alone could leave the old t
behind and duplicate it when a replacement is inserted.

Each alternative is generated as a complete word, including its joins, dots and
crossbars. Placement uses one uniform scale and translation inside the old word's
box. Alternatives narrower than 80% or shorter than 85% of the original box are
rejected, as are incompatible baselines. Outside raw coordinates and pen-lift
states are preserved, to floating-point roundoff. Words are processed from right
to left so earlier point indices remain valid after replacements of different
lengths. Page normalization may still move rendered positions slightly; the
renderer retains the existing larger size, upright adjustment and varied layout.

## Evidence and rejection

TrOCR decodes word crops without receiving the expected caption. A separate
teacher-forced calculation compares content-token likelihood for the same word
across alternatives. It omits the tokenizer BOS and excludes EOS likelihood:
otherwise token misalignment or punctuation termination can dominate the score.
Case and punctuation are ignored only for isolated lexical-word matching;
internal spaces are retained, so `in to` is not accepted as `into`. The existing
full-line comparison still checks case, punctuation and numbers.

An alternative must read correctly in isolation, improve mean content-token
negative log likelihood by at least .10, and not worsen the current full-line
readback, prioritizing numbers and then edit distance. Other words' raw paths
remain unchanged. Failure to find a suitable take keeps the original. The UI
exposes unresolved word checks separately from a full-line OCR match, and retakes
cannot replace verified wording with weaker word or neighboring-context evidence.

## Verification and limits

Regression coverage includes preserved neighbors and pen lifts, uniform word
placement, rejection of connected or interleaved blocks, detached final-t
ownership, split-word rejection, blind-decoding isolation, correct likelihood
token alignment, content-only scoring, full-line fallback and retake protection.
Neural comparisons and paired images are saved under
`artifacts/evaluation/t-word-resampling/`. The essay audit is under
`output/pdf/college-essay-t-context-words/verification.json`.

These checks are conservative and recognizer-dependent. They do not establish
human-rated legibility or guarantee that every t is corrected. Uncertain words
are left intact for visual review. The earlier full-line-only reranker changed
one of 12 paired samples; it is retained for evaluation, while production uses
closed-word regeneration to protect surrounding pen paths. The revised essay
also retains one visually reviewed intact-line alternative from that comparison.

The final essay contains nine ownership-checked word replacements and that one
intact-line alternative, across ten changed lines. All ten before/after pairs
and all seven final PDF pages were visually inspected. Source text, paragraphs,
writer reference, ink and page settings are preserved. Full-line exact readbacks
remain 92 before and after; unresolved differences remain flagged. In the fixed
12-sample paired word-regeneration probe, four takes changed and full-line exact
readbacks remained 9/12. These are recognizer-based development probes rather
than a human-rated accuracy study. All 76 regression cases passed across the
non-API, API and targeted test runs.
