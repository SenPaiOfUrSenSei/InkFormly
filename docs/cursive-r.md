# Withdrawn cursive-r experiment

User review rejected the reference-shaped geometric alternate because words
containing it became unreadable. Matching OCR and the numeric placement guards
did not establish acceptable letter formation or joins. Earlier visual approval
of the preview was insufficient and is superseded by this feedback.

The original geometric alternate stays disabled. Legacy API and CLI requests
cannot enable it, and new documents record `cursive_r=false`. The current
**Connected cursive r** option uses a separate fit constrained by the two stroke
joins. Its setting is `prefer_cursive_r`, as described in the [README](../README.md).

Old saved documents still load without rewriting their coordinates. Those with
applied r edits receive a regeneration warning, even when wording checks are
off. Retakes cannot reapply the r modifier, and its OCR scores cannot force
retention of the rejected shapes. The full-line wording guard still applies.
An already exported PDF is unchanged; it must be regenerated to replace edits.

## Historical research

`inkform/r_style.py` and `scripts/evaluate_cursive_r.py` retain the rejected
experiment for reproducibility, outside the generation path. The curve changed
an aligned lowercase-r body and moved later letters inside its word before
uniformly fitting the word. Ownership, pen-lift, tangent and readback guards
failed to prevent visually broken joins.

The sixteen-word probe accepted two changes by OCR. Both those records and the
sentence examples in `artifacts/evaluation/cursive-r-context-final/` are rejected
as evidence of improved legibility. The previous comparison PDF is a historical
failed experiment, not an approved writing style. A reliable match to the
photographed r remains unresolved; another isolated glyph warp is not justified
by these results.
