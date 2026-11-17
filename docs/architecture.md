# InkForm architecture decision

## Decision

Use learned, text-conditioned online handwriting synthesis with separate ink
rendering and page composition. Start with a compact Graves-style attention LSTM
and mixture-density output, behind a Python generator interface. The default
is now the Sean Vasquez pretrained network with real writer reference 2 and a
locally implemented PyTorch inference architecture. The X-rayLaser Epoch 52
backend remains a comparison baseline. Neither was trained in this project.

The architecture is:

```
Text + sampling controls + optional trajectory reference
  -> input validation / word-aware line planning
  -> replaceable trajectory generator
  -> blind handwriting recognition and bounded candidate selection
  -> absolute strokes + generation diagnostics
  -> bounded A4 layout in millimetres
  -> ink deposition renderer
  -> PNG, PDF, and reusable trajectory JSON
```

Text remains available alongside each generated line. A generation that reaches
its step budget must be reported as incomplete, never silently exported as a
successful document. Unknown characters must be rejected instead of mapped to
the upstream unknown token. Pressure is simulated and identified as such.

Use one local model instance and serialize inference: the generator uses PyTorch
random state, and CPU autoregressive inference does not benefit from many
concurrent requests on this laptop. Seed generation independently of rendering.
Rendering changes must never re-run the model.

## Candidates inspected

| Candidate | Output | Practical consideration |
| --- | --- | --- |
| [Vasquez synthesis](https://github.com/sjvasquez/handwriting-synthesis), [converted weights](https://github.com/niccolodevries/svg-handwriting-generator) | Coordinate deltas and pen lifts | Selected for improved inspected letter shapes; 3.64M parameters, standard LSTM, real reference trajectories, CPU inference. Uppercase Q/X/Z unavailable. Asset licensing is not established. |
| [Graves PyTorch toolkit](https://github.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit) | Coordinate deltas and end-of-stroke flags | Small downloadable checkpoints, explicit CPU loader, context attention and trajectory priming. The checkpoint does not provide a selectable writer embedding. |
| [DiffInk](https://github.com/awei669/DiffInk) | Style-conditioned trajectories | Modern full-line model; weights are linked. Published inference loads dataset files and references; not a ready arbitrary-text endpoint. |
| [VATr](https://github.com/aimagelab/VATr) | Styled text image | Useful image-quality comparison, but no native pen trajectories. Documented setup pins PyTorch 1.13.1 and CUDA 11.7. |
| [Eruku](https://huggingface.co/blowing-up-groundhogs/eruku) | Styled text image | Reference-image conditioning, but T5-large-based architecture plus VAE, and no editable pen trajectories. |

DiffInk's published validation config specifies a 16-layer, 896-dimensional
diffusion Transformer, alongside a VAE and OCR/style modules:
https://github.com/awei669/DiffInk/blob/main/configs/dit_val_config.yaml
Its validation program selects CPU if CUDA is absent, so CUDA documentation alone
does **not** prove CPU inference is impossible. Its packaging, model size, and
dataset dependency make it a separate integration experiment.

Eruku's config explicitly names `google-t5/t5-large` and a separate VAE:
https://huggingface.co/blowing-up-groundhogs/eruku/blob/main/config.json

## Laptop constraints

Observed hardware: Ryzen AI 7 350, Radeon 860M, 16 logical CPUs, approximately
14 GiB usable physical memory. At the start of the experiment only about 1.5 GiB
was available, with swap already in use. Select CPU-only inference initially;
avoid adding an unverified ROCm stack to the system.

## Writer consistency

The Graves checkpoint has no writer-ID control. A seed guarantees reproducibility,
not writer identity. Its supported trajectory priming can provide a common style
reference across lines. It cannot infer the user's pen trajectory from a scanned
image. A future style-conditioned backend must preserve the same stroke contract.

The default is now the fixed real reference `House was crowded.` (writer 2, connected cursive / option A),
downloaded with verified checksums. Changing source text or seed keeps this
reference while mixture choice, Gaussian motion and learned pen lifts remain
stochastic. Imported references take precedence. References have an explicit
coordinate codec: old Graves-normalized references are converted to the new
model's native units and y-axis direction before priming. Saved documents can
still render directly without loading either neural model.

The new network uses softplus Gaussian attention and standard LSTM cells;
its weights are not compatible with the old peephole-LSTM checkpoint. The
implementation matches the inspected conversion's hidden states exactly over
100 reference steps. The sampling loop uses its actual output pen lift to end
the line once attention reaches the final character and the end marker is
sufficiently strong. Letters and digits require the end marker to dominate,
allowing late crossbars/dots; terminal punctuation accepts half the peak
attention to reduce trailing artifacts. The learned attention window is
unchanged. The final drawn endpoint is
included, avoiding truncated terminal strokes. A step-budget exit is failure.
There are no word-specific patches or character template substitutions.

## Content reliability

Checked mode is on by default: clean stroke rendering -> local TrOCR-small
handwriting recognition -> strict text comparison -> up to six candidates.
The recognizer receives only the image, never the requested text. Rank numerical
agreement ahead of total edit distance, preserve case and punctuation, and only
normalize OCR token spacing and typographic quote variants. Record candidate
seed/bias and readback diagnostics in trajectory JSON. Shorten checked lines to
48 characters. Incomplete geometry is rejected before readback.

Show unresolved differences, input, readback and a line image together. Allow
rewriting one line without changing the others, retaining the old take if the
replacement reads back worse. Recognition is imperfect and does not establish
ground-truth spelling. It runs on clean geometry, so changing decorative ink
does not re-run synthesis or silently change wording.

## Scope of the first implementation

Local web UI and command-line interface, real checkpoint inference, explicit
completion checks, reference-stroke priming, editable ink controls, A4 pagination,
300 dpi export and saved trajectories. No model training or pressure prediction
is needed. Visual quality and content fidelity remain subject to sample review;
attention completion is not an OCR accuracy guarantee.

## Cursive variation controls

Option A uses real reference 2. Neural synthesis retains text context and this
shared reference. The sampler seed changes neural movements; render variation
seeds do not regenerate text. Four presets share a single Python definition
served at `/api/variation-presets` and consumed by the UI and CLI.

Local shape, gap and layout variation use separate seeded streams. Positive
integrated writing pace preserves baseline order. Shared smooth maps preserve
coincident stroke joins. Word-gap variation translates complete ink groups
across blank projected intervals rather than cutting joined strokes. Fitting
keeps every stroke in its printable row. Controls remain geometric/optical
augmentation; they cannot infer motor habits or measured pressure absent data.
