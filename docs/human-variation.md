# Within-writer variation investigation — 2026-11-08

The selected connected-cursive reference 2 remains fixed. The change is in
neural sampling, not a new font, glyph replacement, or model fine-tuning.

## Evidence and design

Human handwriting varies in the movement used to construct a form and the
transition between forms. Size, position and slant also vary. Research on
[graphic coordination and coarticulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3781346/)
supports treating transitions as part of the task. This does not establish that
an arbitrary noise function or our decoder is a validated human motor model.

We downloaded [Character Trajectories](https://archive.ics.uci.edu/dataset/175/character%2Btrajectories),
Ben Williams (2006), DOI 10.24432/C58G7V, CC BY 4.0. It has 2,858 recordings of
20 letters from **one writer**, captured at 200 Hz. The archive contains
normalized, differentiated, smoothed and time-shifted recordings; only isolated
single-pen-down characters are represented. This is a different person from
the model's cursive reference and does not supply joined-word context.

`scripts/analyze_writer_variation.py` restores the relative x/y scales,
integrates velocities, resamples trajectories by arc length and normalizes
translation/overall size. Random repeated forms still differ in loop shape,
proportions and entry/exit movements. Its paired similarity-aligned residuals
retain both proportion and shape differences; they are not allograph labels.
The [recorded sample sheet](../artifacts/evaluation/human-variation/recorded-letter-variation.png)
shows eight unranked samples each of `a e g m n s`.
[Measurements and limitations](../artifacts/evaluation/human-variation/measurements.json)
are saved with the deterministic sample seed. These observations informed the
decision to vary learned movements rather than just deform finished letters.
They **did not calibrate a numeric target for the model**, provide pressure
training, or personalize this generator to that writer.

## Implemented change

The model already conditions its pen-motion mixture on text attention, previous
movements and the primed reference. Previously sampling bias sharpened mixture
selection and reduced coordinate standard deviations together. The additional
`form_diversity` control sets:

```
temperature = 1 + 0.65 * form_diversity
mixture probabilities = softmax(logits * (1 + bias) / temperature)
coordinate standard deviations = exp(log_sigma - bias)
```

Existing probability pruning, means, correlations, pen-lift probabilities and
termination rules remain unchanged. Mixture components represent local motions,
**not named letter shapes**. Broader mixture selection lets the autoregressive
network develop different trajectories, without directly increasing coordinate
noise. Higher diversity is not necessarily more human or more readable.

The API, Words panel and CLI expose the control, default 0.5. Zero restores the
original sampler. Old saved documents default to zero; their exact trajectories
still rerender unchanged. Document and line metadata record requested and actual
settings. Checked attempts use diversity multipliers `1, 1, .75, .5, 0, 1`
and retain the existing numeric/readback ranking. All candidates use the same
reference. Unresolved differences remain flagged. Render-only variation and
ink controls continue to be independent of neural generation.

## Controlled development comparison

Same writer, six texts, seeds 42/817, bias .85; no retries or rendering warps:

| Diversity | Completed / 12 | Exact OCR / 12 | Edit distance including incomplete penalty |
| --- | --- | --- | --- |
| 0 | 11 | 0 | 117 |
| .5 | 11 | 1 | 100 |
| 1 | 11 | 2 | 126 |

For an incomplete trajectory, the aggregate uses the full input character
count as its penalty. OCR preserves case, numbers and punctuation. This is
a small development probe; failures and readbacks are retained in
[raw samples](../artifacts/evaluation/form-diversity/samples.json).
All three settings produced problematic takes. Maximum diversity was not
chosen as the default.

The checked comparison used the same six texts, seed 817, up to six attempts:

| Requested diversity | Exact selected OCR / 6 | Total selected edit distance | CPU seconds |
| --- | --- | --- | --- |
| 0 | 3 | 6 | 85.5 |
| .5 | 4 | 6 | 93.0 |

Two of the six new-setting lines retained the original sampler fallback.
Three retained .5 with an exact readback; another retained .5 with a mismatch.
The [full checked records](../artifacts/evaluation/form-diversity/checked/samples.json)
include seeds, biases, diversity, attempts and all selected strokes.
This does **not demonstrate an overall accuracy improvement**, a measured match
to human variation, or exact writer identity. OCR can infer malformed writing.
The before/after proof uses three disclosed sentences, with rendering warps
disabled, to make visual comparison possible. It is not a regenerated essay.

## Reproduction

Install the optional `research` dependency (SciPy); runtime generation does not
need it. Place the official ZIP at
`data/character-trajectories/character-trajectories.zip`:

```
https://archive.ics.uci.edu/static/public/175/character%2Btrajectories.zip
SHA-256: 5d2db017ef0d8cf0e65ed060c9e90399f78eb9f1e3cb63e22ca8c3ef4ba67d52
```

```
.venv/bin/python -m scripts.analyze_writer_variation
.venv/bin/python -m scripts.evaluate_form_diversity
.venv/bin/python -m scripts.evaluate_checked_diversity
```

Evaluation scripts resume from saved records; use a new output directory for
fresh raw-sampler experiments. Delete/move checked records explicitly for a
fresh checked run after changing sampling policy. These measurements apply to
the code and checkpoint used on this date, not future model revisions.
