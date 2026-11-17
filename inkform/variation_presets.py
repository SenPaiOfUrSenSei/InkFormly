"""Bounded render presets: each keeps the selected neural writer reference."""


def _preset(label, letters, layout, width, height, slant, rhythm, gaps, starts, spacing, tilt, baseline, density, pressure):
    return {"label": label, "page": {
        "letter_variation": letters, "naturalness": layout,
        "width_variation": width, "height_variation": height,
        "slant_variation": slant, "slant_adjustment_deg": 0, "rhythm_variation": rhythm,
        "word_spacing_variation": gaps, "word_gap_min_mm": 0, "line_start_variation": starts,
        "line_end_variation": 0, "fit_lines_individually": False,
        "line_spacing_variation": spacing, "line_tilt_variation": tilt,
        "baseline_variation": baseline},
        "ink": {"density_variation": density, "pressure_variation": pressure}}


VARIATION_PRESETS = {
    "readable": _preset("Readable cursive", .3,.4,.4,.35,.25,.4,.2,.45,.35,.3,.35,.4,.45),
    "light": _preset("Light touch", .3,.3,.35,.3,.25,.35,.2,.3,.25,.2,.25,.3,.35),
    "balanced": _preset("Balanced cursive", .5,.45,.5,.45,.35,.5,.35,.45,.4,.35,.4,.5,.5),
    "lively": _preset("Lively cursive", .65,.55,.65,.55,.5,.65,.55,.6,.5,.5,.55,.65,.6),
    "loose": _preset("Loose cursive", .75,.65,.7,.6,.65,.75,.65,.7,.65,.65,.65,.75,.7),
}

VARIATION_PRESETS["readable"]["page"]["word_gap_min_mm"] = 1.2
VARIATION_PRESETS["readable"]["page"]["slant_adjustment_deg"] = -18
VARIATION_PRESETS["readable"]["page"]["fit_lines_individually"] = True
VARIATION_PRESETS["readable"]["page"]["line_end_variation"] = .4
