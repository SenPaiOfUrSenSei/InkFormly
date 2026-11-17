# InkForm

A local handwriting studio using **real pretrained neural pen trajectories**,
an independent textured-ink renderer, and A4 page composition. It includes a
browser UI, CLI, reusable trajectory JSON, trajectory-reference priming,
PNG previews, and lossless 300 dpi PDF/PNG exports.

## Run

The project virtual environment and pretrained checkpoints are already
installed in this workspace. Start the studio:

```bash
.venv/bin/python -m inkform serve
```

Open **http://127.0.0.1:8000**. Click **Generate handwriting**, review the page,
then **Prepare print files**. Ink and paper edits use **Apply ink & paper to
this take**, which preserves the pen trajectories. Text and handwriting edits
require a new generation. Print at actual size / 100%.

**Relaxed layout** in the Paper tab adds restrained variations in line
starts, baseline, line spacing, slant and writing rhythm. It defaults to 45%;
zero restores tidy alignment. New pages use 7.5 mm writing height and 11.5 mm
line spacing by default. The smooth mapping preserves joins and pen lifts,
keeps strokes inside their line boxes, and reuses saved trajectories. Its seed
is independent of ink and DPI, so recoloring does not move the writing.
The CLI accepts `--naturalness 0.6` for generation or rendering.

**Letter variation** separately varies local width, height and lean over roughly
one or two letters at a time. Correlated smooth changes preserve coincident
joins and pen lifts while repeated forms differ in proportion. The underlying
learned writer, source text and saved trajectories are retained. This is bounded
geometric augmentation, not retraining or a measured human motor model. The
default is 55%; use `--letter-variation 0` to disable it. Both variation settings
can be adjusted by rerendering an existing take, without neural regeneration.

**Double-check the wording** is enabled by default. A separate local Microsoft
TrOCR handwriting recognizer reads each candidate without seeing the requested
text. InkForm tries up to six takes, prioritizes matching numbers, and prefers
a complete wording match. Retries explore nearby sampling biases while keeping
the reference style. Checked mode wraps at 48 characters to avoid long sequences.

Open **Wording readback** to compare the handwriting, input and recognized text.
Case, punctuation, missing words and numbers are checked; only harmless OCR
token spacing and typographic quotes are normalized. **Try another version of
this line** preserves all other lines. If the new readback ranks worse, the
previous line is retained. An OCR match is evidence, not a guarantee: proofread
the final project before printing.

**Clearer t and word flow** checks words containing internal lowercase `t` in
isolation. When the word can be separated safely, InkForm generates up to four
complete alternatives with the same writer reference and requested diversity.
It places a complete word using translation and uniform scaling only; individual
letters, crossbars and joins are never bent. Neighboring raw pen paths and pen
lifts stay in place. Ambiguous character ownership, interleaved strokes, missing
blank gaps, incompatible proportions and weaker full-line readback reject a take.

The earlier stem narrowing and crossbar raising are excluded from this option.
The new policy is enabled for checked requests; `--no-t-refinement` disables it.
Old saved documents retain their policy, and rerendering ink/paper never changes
their strokes. Unverified t words remain flagged even when full-line OCR matches.
This improves candidate selection without retraining the pretrained weights;
readback is evidence rather than a guarantee of every letter's shape.
See [word-level t generation](docs/word-clarity.md) for safeguards and evaluation,
and the [rejected stroke-edit experiments](docs/t-formation.md).

The reference-shaped cursive-r experiment has been withdrawn: user review found
its joins made words unreadable, even when OCR matched. The UI no longer offers
it, and generation/line retakes use native neural letter forms even for legacy
requests with `cursive_r=true`. Existing writer conditioning, natural variation
and whole-word t selection remain in place. Saved strokes are preserved during
ink-only rendering; old documents containing the rejected edits are flagged for
regeneration. See the [rejected r experiment](docs/cursive-r.md).

**Connected cursive r** fits an angled shoulder and rounded foot between two
points on the writer's continuous stroke. The entry and exit follow the local
stroke directions. InkForm rejects crossings and checks the word and line
wording. It can also try a new whole word from the same writer when the original
has too little room. Enable it in Handwriting or with `generate --prefer-cursive-r`.

## Clean installation

Python 3.11–3.14. This run used Python 3.14.7 and CPU-only PyTorch 2.15.0.
The default backend uses ordinary PyTorch LSTM cells. The older comparison
backend uses TorchScript and emits a Python 3.14 compatibility warning.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python scripts/download_readable_model.py
.venv/bin/python scripts/download_recognizer.py
.venv/bin/python -m inkform serve
```

The pinned requirements are for the tested Python 3.14 environment. On Python
3.11–3.13, install a compatible CPU PyTorch wheel first, then
`pip install -e '.[dev]'`. No GPU, training data, external service, or API key is required.

Checkpoints and the fixed writer reference are fetched from pinned upstream
revisions and verified against SHA-256 checksums. Downloaded weights and style
samples are excluded from Git. Their [provenance](models/vasquez/PROVENANCE.md)
is recorded separately; their redistribution license is not established.

## Architecture and model selection

```
Text and trajectory reference
  -> Vasquez attention LSTM + mixture-density sampler (CPU)
  -> validated coordinate deltas and pen-lift states
  -> independent TrOCR readback, candidate ranking and bounded retries
  -> trajectory normalization and A4 line/page layout
  -> variable-width optical ink deposition
  -> PNG / PDF / reusable JSON
```

**Default: the pretrained Sean Vasquez network**, using the pinned PyTorch
weight conversion and real writer reference 2 (connected cursive, option A). It has 3,637,231 parameters,
three standard 400-unit LSTM cells, Gaussian text attention and a 20-component
pen-motion distribution. InkForm implements its inference architecture locally.
The app previously defaulted to X-rayLaser Epoch 52; that backend remains
available for comparisons.

The replacement improves the inspected `s`, `E` and `SAT` formations while
retaining sampled trajectories and a fixed writer reference. It is a different
pretrained model, not a fine-tuned version of the old weights. The default
reference was subsequently changed from rounded writer 3 to more upright
writer 0 after visual comparison, then to connected-cursive writer 2 after
the user selected option A; this changes the learned style conditioning,
rather than trying to remove roundness with geometric deformation. The decoder also
retains the final pen movement when ending a line. No character templates,
per-word exceptions or manually drawn punctuation are used by the generator.

The earlier comparison used the rounded reference 3. On six new sentences
with two seeds each, both backends completed 12/12 raw
samples. OCR differences totaled 17 for Epoch 52 and 4 for the replacement;
exact readback matches were 7/12 and 10/12 respectively. This is a small
development comparison, not proof of error-free letter formation. Inspect the comparison
images and the normal checked-pipeline examples linked in the evaluation notes.
CPU generation depends on line length, priming and candidate retries.

See the [architecture decision](docs/architecture.md),
[evaluation notes](docs/evaluation.md), and comparison images:
[Epoch 52](artifacts/evaluation/epoch52-comparison.png),
[Epoch 56](artifacts/evaluation/epoch56-comparison.png).

## Variation controls

**Letter-form diversity** in Words changes neural sampling before rendering.
It broadens the choice of learned next pen movements without increasing the
Gaussian coordinate noise. The text attention, previous movements, and fixed
writer reference still condition every step. It defaults to 50%; zero restores
the original sampler. Use `generate --form-diversity 0.5` in the CLI. This
requires regenerating handwriting; rerendering an old document cannot add new
learned formations. Retakes record their actual diversity and restraint.

Checked retries reduce diversity on some attempts and include an original
sampler fallback. The existing numeric/readback ranking still decides which
take is retained. Stronger diversity can cause malformed writing; it does not
guarantee a distinct alternate shape for every occurrence of a character.
These are local learned motion components, not labeled letter variants.
Model weights have not been retrained. See the [human variation investigation](docs/human-variation.md)
for recorded samples, controlled comparisons, and limitations.

Option A (reference 2, `House was crowded.`) is the default learned cursive
style. The render presets are **Readable cursive**, **Light touch**, **Balanced cursive**,
**Lively cursive**, and **Loose cursive**. Presets keep this writer and the saved
neural strokes; they do not mix unrelated handwriting styles.

**Readable cursive** uses less added deformation and calmer ink variation,
while retaining the model's different letter formations. It widens detected
blank word gaps toward 1.2 mm before fitting the line. This is geometric gap
detection, not exact text alignment: it cannot separate words joined by an ink
stroke, and fitting a long line may reduce the requested gaps. The Paper tab
exposes this separately as **Minimum detected word gap**. Other presets retain
the previous gap behavior (zero minimum). The CLI accepts
`--variation-preset readable` and `--word-gap-min-mm 1.2`.

**Overall letter lean adjustment** in Paper changes the original letter lean
with a common horizontal shear. Negative values reduce a rightward lean;
positive values increase it. Zero preserves the original rendering. The angle
is a relative correction, not a measurement of the final slant. Joins, pen lifts
and the learned variations are retained; saved trajectories are unchanged.
Readable cursive now uses a -18 degree adjustment. Old saved settings default
to zero. The CLI accepts `--slant-adjustment-deg -18` when rendering or generating.

**Fit crowded lines individually** keeps a wide or tall line from shrinking
the whole document. Lines share a preferred writing size and only those that
need fitting are reduced. It is enabled by Readable cursive; old saved settings
retain the shared fitting behavior. Use `--fit-lines-individually` or
`--no-fit-lines-individually` to override the preset.

**Line endings** reserves a varying amount of space at the right edge for
crowded lines. Short lines keep their natural length. Together with **Line
starts** and **Relaxed layout**, this produces uneven page edges while keeping
whole strokes inside their row. The CLI accepts `--line-end-variation 0.7`.
The chosen line starts are preserved when fitting a crowded line in this mode.

In the Paper tab, expand **Individual variation controls** for letter width,
height, lean, rhythm, word spacing, line starts, line spacing irregularity,
line tilt and baseline drift. Ink has separate darkness variation, pressure,
flow and roughness controls. The letter/layout master sliders scale their
respective effects. Most individual controls default to 50%, matching the
previous renderer; detected word-gap variation defaults to off unless a preset
is selected. Gap detection uses blank ink intervals, not exact word alignment.

Use **Shuffle variations** and **Apply ink & paper to this take** for another
rendering of the same words and writer. The variation seed is independent of
ink and DPI. **New natural take** and sampling bias change the neural movements
and require generation; smaller sampling bias can also reduce legibility.
Readback describes the base neural strokes before render variation. Review the
final preview, especially with stronger variation settings.
All rendering changes are bounded inside printable line boxes, preserve joins
and pen lifts, and are geometric/optical simulation rather than model training.

```bash
.venv/bin/python -m inkform render output/project/document.json --out output/lively --variation-preset lively --variation-seed 101 --height 8 --spacing 12
# Override individual preset values:
.venv/bin/python -m inkform render output/project/document.json --out output/custom --variation-preset lively --slant-variation 0.3 --word-spacing-variation 0.6
```

The preset definitions are shared by the API, UI and CLI. Changing a preset or
seed never rewrites source text or replaces the selected reference.

## CLI

Generate from a text file:

```bash
.venv/bin/python -m inkform generate --input project.txt --out output/project --seed 42
```

Render the same handwriting with different ink, without loading the model:

```bash
.venv/bin/python -m inkform render output/project/document.json --out output/black --color '#22252b' --roughness 0.6
```

Reuse pen movements as a style reference:

```bash
.venv/bin/python -m inkform generate --text 'Another page in a related style.' --reference output/project/reference.json --out output/continued
```

`python -m inkform --help` lists the commands; subcommand help lists ink, size,
spacing, margin and resolution controls. To compare the raw upstream sampler:

```bash
.venv/bin/python scripts/download_model.py --epoch all
.venv/bin/python scripts/probe_model.py --checkpoint models/graves/Epoch_52 --label epoch52
.venv/bin/python scripts/probe_model.py --checkpoint models/graves/Epoch_56 --label epoch56
```

The probe writes SVG and JSON. Convert the SVG to PNG with `rsvg-convert` if
desired; it is not a runtime dependency of the app.

The recognizer is `microsoft/trocr-small-handwritten`, pinned to revision
`b4648cfa171985a6745f37ddd637e98c0da958ac`. Its weights and tokenizer are verified
by SHA-256 and load locally without remote code. It adds approximately 236 MiB
of downloads and additional CPU/RAM use. Readback uses a clean rendering of the
handwriting, independently of decorative ink and paper. To skip it for a quick
draft, uncheck the wording option or pass `generate --fast`.

## Tests

```bash
.venv/bin/python -m pytest -q
# With the local server running:
.venv/bin/python scripts/smoke_test.py
```

Tests cover pen lifts, print bounds, pagination, deterministic rendering and
model sampling, unsupported characters, failed synthesis, recoloring without
regeneration, numeric priority, unresolved readback differences, single-line
rewriting, and PDF/PNG downloads. API tests need normal local event-loop
permissions; the restrictive agent sandbox stalled Starlette's test client,
while the same tests passed outside it. Browser automation was unavailable in
the development session; UI assets and API flows were checked separately.

## Current limits

- This checkpoint sometimes merges, repeats, or misforms letters. Completion
  checks catch unfinished sampling. Readback and retries reduce likely mistakes,
  but recognition can also misread or infer text. Unresolved differences remain
  visible. Review the generated pages before printing.
- Style uses a common trajectory reference, not a learned writer-ID selector.
  The default is a fixed real writer sample across documents; an imported
  reference overrides it. A seed controls variation, not writer identity. Exact
  writer identity is not guaranteed.
- Reference input is normalized pen trajectories with matching text. Scanned
  handwriting and reference-image adaptation are not implemented.
- Supported characters are the checkpoint's Latin letters, digits, spaces and
  limited punctuation. **Uppercase Q, X and Z are absent from its vocabulary**
  and rejected explicitly, along with unsupported Unicode; they are never
  silently replaced or lowercased. Lowercase q, x and z are supported. Paragraph
  breaks are kept; repeated spaces are collapsed for word-aware wrapping.
- Pressure, local ink flow and grain are simulated. Ink changes preserve model
  geometry; layout removes most global line tilt and estimates baselines.
- Requests are limited to 4,000 characters / 100 lines and three queued or active
  tasks. The server keeps up to 40 session job records. Artifacts remain under
  `output/jobs`; restarting clears the in-memory job index, not saved files.
- DiffInk, VATr and Eruku were inspected but not run. They remain separate
  experiments, with the reasons documented in the architecture decision.

The app runs on localhost and makes no remote calls during generation or
rendering. Trained weights and writer samples originate from IAM-based upstream
projects; redistribution rights have not been established.
