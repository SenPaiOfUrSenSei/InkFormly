# Local evaluation — 2026-11-03

## Coherent cursive export — 2026-11-16

Added the Readable cursive render preset and a bounded, optional minimum for
detected blank word gaps. The preset reduces added shape deformation, ink
density variation and line irregularity while keeping reference A and the
model's sampled variations. It preserves connected strokes. Gap detection
is geometric; connected words cannot be separated this way, and final fitting
can reduce the requested gap. Existing presets default to a zero minimum.

For the current essay, 32 uncertain lines were resampled with bias 1.05 and
requested form diversity .5. Candidates with worse readback were rejected.
Direct visual comparison rejected six additional alternatives where names or
letter shapes were less clear. The six-page export preserves the exact source
line text and writer reference. Exact base-trajectory readbacks increased from
74 to 91 of 106 written lines. This is selection readback, not an independent
accuracy or human-likeness measurement. Fifteen lines remain flagged.

All final PDF pages were rendered with Poppler and visually reviewed. The
geometry check found no nonfinite coordinates or writing outside page margins;
all pages are A4. The 41 regression tests pass, including spacing without
deforming letter geometry, legacy preset behavior, API rendering and downloads.
JavaScript syntax was checked. The browser UI also omits the new spacing field
when connected to an older running API until InkForm is restarted.

## Learned motion diversity — 2026-11-08

The new `form_diversity` control broadens learned motion choices independently
of Gaussian coordinate noise, while preserving reference 2. The
[investigation](human-variation.md) records the human trajectory dataset,
research, sampling equation, full raw and checked comparisons, and limitations.
The checked development comparison had equal total OCR edit distance for the
original and new settings; this is not evidence of improved overall accuracy.
All 40 regression tests pass, including sampler independence, old-document
compatibility, fallback settings, and real-model repeatability/variation.

## Connected cursive and variation controls

The user selected option A: connected cursive reference 2 (`House was crowded.`).
It is now the default reference with a 0.85 sampling bias. This changes style
conditioning, not model weights. The fixed reference remains shared across
lines; generation seeds still sample different trajectories.

Four presets (Light touch, Balanced cursive, Lively cursive, Loose cursive)
share definitions across the API, browser and CLI. Individual controls cover
local width, height, lean and rhythm; blank word gaps; line starts, spacing,
tilt and baseline; optical ink density; and an independent variation seed.
These are bounded geometric and ink simulations, not learned human motor or
pressure measurements. Blank-gap detection is geometric and can use fewer gaps
than the source's word count; joined strokes are never split.

All 37 tests pass, including independent width/height changes, positive gap
ordering, unchanged ink-group geometry, print bounds with every control at its
maximum, shared preset validation, and preset rerendering that preserves saved
source trajectories. The running localhost API serves writer 2, all four
presets and the new UI controls. JavaScript syntax and unique HTML IDs were
checked; interactive browser testing was unavailable.

## Historical upright reference update

The default was changed to reference 0, selected after visually comparing references
0, 3 and 10 on identical text. Reference 0 has taller, narrower forms and less
rounded joins than the earlier reference 3. This changes neural style priming;
the pretrained model weights and decoder are unchanged. Both reference 0 and 3
are included in the pinned checksum downloader. New page defaults are 7.5 mm
writing height and 11.5 mm spacing.

A full 646-word document was regenerated with a fixed reference 0, shorter
lines, 8 mm target height and 12 mm spacing. Of 106 written lines, 104 selected
candidates matched OCR; the remaining two differed in quote spacing. This is
selection readback, not an independent accuracy estimate. All six rendered
pages were visually reviewed. The 34 regression tests passed after changing
the reference and defaults. Earlier measurements below used reference 3 and
must not be attributed to the current reference.

## Letter formation update — 2026-11-04

The default is now a different pretrained neural network: Sean Vasquez's
standard-LSTM model, via the pinned PyTorch weight conversion. It was downloaded
and executed locally; no weights were fine-tuned. See the
[asset provenance](../models/vasquez/PROVENANCE.md) and checksum downloader.
The new inference architecture matches the inspected conversion's hidden states
exactly over 100 reference steps (maximum absolute difference 0; 3,637,231
parameters). It uses the real writer-3 reference across documents.

This is a general backend replacement. There are no `student`, `Eric`, `SAT`
or smiley exceptions in production code, and no manual glyphs. The decoder now
retains the final coordinate of a terminating stroke rather than discarding
it. Both text-attention completion and a pen lift are required; reaching the
step budget is not success. End-marker stopping for punctuation was calibrated
to reduce trailing strokes, while final letters/digits retain the strict
condition so late crossbars and dots can finish. This uses broad text classes,
not glyph templates or special words. Mixture, coordinate and pen-lift sampling remain
stochastic, with reproducible seeds.

Attempts to sharpen or slow the old model's attention helped isolated letters
but harmed other sentences, so those changes were not shipped. Epoch 46 was
also tested: it completed 11/12 new raw samples, with 5 exact OCR readbacks and
29 OCR differences among completed samples. Epoch 52 completed 12/12, with
7 exact readbacks and 17 differences. Epoch 50 was probed on reported phrases
without a convincing visual improvement.

The replacement and Epoch 52 were then compared on the same six new sentences,
seeds 42/817, bias 0.8, and their respective fixed references, with **no retries**:

| Backend | Completed | Exact OCR readbacks | Total OCR differences | Time including readback |
| --- | --- | --- | --- | --- |
| Epoch 52 | 12/12 | 7/12 | 17 | 39.4 s |
| Vasquez writer 3 | 12/12 | 10/12 | 4 | 38.1 s |

These small development results do not establish a universal accuracy
improvement. End-marker calibration used this set; it is not a blind test. The selection
also reflects inspected letter formation and real writer priming. Compare
[student/s](../artifacts/evaluation/formation/readable-heldout-0.png),
[Eric/E](../artifacts/evaluation/formation/readable-heldout-1.png), and
[SAT](../artifacts/evaluation/formation/readable-heldout-4.png).
The first two rows use Epoch 52 and the last two use the replacement.

The normal checked pipeline was then run on three reported phrases and three
new sentences, each with seeds 42 and 817. All 12 selected samples read back
as the source; two needed all six attempts. This is a **selection score**, not
an independent accuracy benchmark, because the same OCR selects and scores
the candidates. Visual inspection is still required: in particular, the smiley
can have extra arcs despite matching OCR. See the
[full sample sheet](../artifacts/evaluation/formation/checked-formation.png)
and [per-candidate diagnostics](../artifacts/evaluation/formation/checked-summary.json).
The two seeds produce different trajectories while retaining the same reference.

An additional 20 raw samples used ten short texts and new seeds 43/818,
including final `t`, `s`, `e`, digits, commas and exclamation marks. All completed;
12/20 read back exactly. The [diagnostics](../artifacts/evaluation/formation/terminal-stress.json)
retain the remaining differences, including case, extra punctuation and a
misread year. These are not silently corrected. A uniformly early ending
threshold cut off late `t` crossbars, so the shipped decoder retains strict
ending for letters and digits. Compare
[final crossbars](../artifacts/evaluation/formation/terminal-stress-3.png).
Reproduce with `scripts/evaluate_terminal_strokes.py`.

All 28 regression tests pass, including real-model reproducibility/variation,
end-of-stroke preservation, incomplete generation, explicit reference codecs,
and rejection of unavailable uppercase Q/X/Z. Old documents still render from
their trajectories; old reference JSON is converted explicitly when reused.
The real API workflow generated 217 characters, rewrote one line,
preserved other lines, recolored without changing trajectories, and exported
A4 PDF/PNG. Four of five written lines matched before rewriting; the remaining
line was flagged. The [receipt](../artifacts/evaluation/quality-live-smoke.json)
records the active backend and checks. The PDF was rendered with Poppler for
visual inspection. No print-on-paper trial has been performed.

The sections below document the earlier baseline and remain historical.

## What was actually run

Both `Epoch_52` and `Epoch_56` synthesis checkpoints from
[X-rayLaser's toolkit](https://github.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit)
at revision `899e9432dfe0998ba2ef31582d11c2d5f24a5185`.
Files are verified by `scripts/download_model.py`. No model was trained.

Environment: Ryzen AI 7 350 / Radeon 860M, Linux x86-64, Python 3.14.7,
PyTorch 2.15.0+cpu, one intra-op thread. No AMD GPU/NPU backend was installed.

Text: `Natural handwriting has rhythm and variation.`

The raw upstream sampler was run with bias/seed pairs `(0.5,42)`, `(0.8,42)`,
`(1.1,42)`, `(0.8,43)`, a 4,000-point ceiling and two trailing spaces.

| Pair | Epoch 52 | Epoch 56 |
| --- | --- | --- |
| 0.5 / 42 | 1.81 s, 1,195 points, readable | 7.04 s, 3,808 points, runaway geometry |
| 0.8 / 42 | 1.33 s, 1,185 points, readable | 2.30 s, 1,397 points, readable |
| 1.1 / 42 | 1.55 s, 1,384 points, readable | 6.59 s, 4,000 points, incomplete |
| 0.8 / 43 | 1.44 s, 1,258 points, readable | 2.17 s, 1,333 points, readable |

“Readable” is a visual judgement for these samples, not a CER/WER score.
Epoch 52 was selected because it behaved more consistently across this small
comparison. A broader blind evaluation could change this selection.

Raw measurements:
[Epoch 52](../artifacts/evaluation/epoch52-benchmark.json),
[Epoch 56](../artifacts/evaluation/epoch56-benchmark.json).

## Integration findings

- The upstream sampler constructs a multivariate normal distribution for each
  point. The app samples the same correlated Gaussian using two normal draws
  directly; it still samples the learned mixture and keeps the learned pen lifts.
  Different floating-point operations change individual takes.
- Reference priming needs to consume the final input point exactly once. Fixing
  a duplicated warm-up step removed repeated leading strokes in the inspected
  demo. A two-space boundary separates reference and target context.
- Model attention can reach the end despite ambiguous or erroneous glyphs.
  The UI exposes generation limits and preserves exact source text alongside
  output. This was the limit of the initial implementation; independent OCR
  readback was added in the reliability update described below.
- The model can drift diagonally on long lines. The layout stage removes most
  global linear tilt while retaining local variation, then uses a bounded
  baseline estimate. This does not improve transcription errors.
- Full paragraphs with a common reference and independent ink rendering run on
  CPU. The demo's five written lines plus one paragraph break took approximately
  22 seconds in one run while other checks were running. Timings vary.
- PNGs are 2480 × 3508 at 300 dpi; PDF media boxes are A4 (595.28 × 841.89 points).
  PDF pages were rendered back with Poppler and visually inspected.

The final live-server smoke test generated a 217-character paragraph, changed
the ink, checked byte-for-byte equality of trajectory JSON, and downloaded the
PDF and PNG ZIP. It passed; generation took 29.94 seconds during concurrent
checks. The [receipt](../artifacts/evaluation/live-smoke.json) records the job
IDs, output sizes and trajectory checksum. The initial automated suite passed
all 10 tests. Interactive browser QA was unavailable because no browser was
exposed to the browser-automation tool in this session.

## What remains unmeasured

No printed-paper trial, user handwriting adaptation, human-labeled OCR content accuracy study,
blind realism comparison, or ROCm benchmark. DiffInk, VATr and Eruku were reviewed
from their published source/configurations; their weights were not executed.

DiffInk's released entrypoint is dataset-driven and uses a large Transformer/VAE
configuration. Eruku uses T5-large and a VAE; both warrant separate memory and
latency experiments. VATr produces raster images and would require a different
rendering contract. None should be described as worse visually based on these
local Graves-only measurements.

## Reliability update — 2026-11-04

Added a separate [Microsoft TrOCR-small handwriting recognizer](https://huggingface.co/microsoft/trocr-small-handwritten),
downloaded at revision `b4648cfa171985a6745f37ddd637e98c0da958ac` and verified
with SHA-256. It receives only a clean handwriting image; it does not receive
the expected sentence. Readback is deterministic on CPU and takes roughly a
second per short line in the initial probe. The output is compared with the
source only after recognition.

Checked mode now samples up to six candidates, explores nearby sampling biases,
and ranks matching numbers before general edit distance. A fixed, model-generated
reference is used across documents, and checked lines are capped at 48 characters.
Case, numbers, words and punctuation are preserved in comparison; spaces before
punctuation and typographic quote variants are normalized. Unresolved differences
remain `review`, not `match`.

The studio displays input, readback, and the actual rendered line together.
Single-line rewriting preserves every other line and retains the previous take
if its readback is better. The API regression checks both behaviors. Old saved
documents load with `unchecked` status instead of implying retrospective checks.

Twenty tests passed, including numeric priority, missing/repeated words, case and
punctuation, bounded best-candidate selection, unchanged neighboring lines, and
the existing PDF/layout checks. The live workflow also exercised real OCR,
line-only rewriting, recoloring and print downloads; its receipt is
[`quality-live-smoke.json`](../artifacts/evaluation/quality-live-smoke.json).

The [project-text comparison](../artifacts/evaluation/quality-comparison.json)
uses eight sentences including dates, decimals and repeated letters. It measures
OCR disagreement, not human-labeled transcription accuracy. Candidate selection
uses the same recognizer, so a better score is not an independent accuracy
benchmark. Reused initial baselines also compare pipeline revisions, including
their reference styles.

In that eight-line before/after comparison, total OCR character differences
fell from 27 to 4, and exact readback matches rose from 2/8 to 4/8. All four
remaining differences were flagged: a final comma instead of a period, `#`
instead of `7`, `per cent` instead of `percent`, and lowercase `clean` instead
of `Clean`. These are recognizer readings of selected candidates, not proof
that only four visible errors remain. The checked run took about 157 seconds
including retries and readback under concurrent checks, versus about 47 seconds
for the initial baseline plus readback.

The numeric stress probe remains deliberately visible:
[`numeric-checked.json`](../artifacts/evaluation/numeric-checked.json).
The date's numbers read back correctly with the revised reference, but case and
punctuation still differed. Other samples included `25` reading as `2TS` and
`12.5` as `12.6`; these were flagged for review. The model is not reliable enough
to promise error-free numbers, technical text, or major projects without visual
proofreading. The checks and line repair make those limits actionable rather
than hiding them.
