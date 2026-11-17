from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class InkSettings(StrictModel):
    color: str = Field(default="#203f78", pattern=r"^#[0-9a-fA-F]{6}$")
    width_mm: float = Field(default=0.25, ge=0.15, le=0.9)
    opacity: float = Field(default=0.95, ge=0.15, le=1)
    roughness: float = Field(default=0.4, ge=0, le=1)
    pressure_variation: float = Field(default=0.45, ge=0, le=1)
    flow: float = Field(default=0.8, ge=0.25, le=1)
    density_variation: float = Field(default=0.5, ge=0, le=1)
    seed: int = Field(default=71, ge=0, le=2**31 - 1)


class PageSettings(StrictModel):
    margin_mm: float = Field(default=20, ge=10, le=40)
    writing_height_mm: float = Field(default=7.5, ge=3, le=9)
    line_spacing_mm: float = Field(default=11.5, ge=7, le=18)
    paper: Literal["plain", "ruled"] = "ruled"
    naturalness: float = Field(default=0.45, ge=0, le=1)
    letter_variation: float = Field(default=0.55, ge=0, le=1)
    width_variation: float = Field(default=0.5, ge=0, le=1)
    height_variation: float = Field(default=0.5, ge=0, le=1)
    slant_variation: float = Field(default=0.5, ge=0, le=1)
    slant_adjustment_deg: float = Field(default=0, ge=-50, le=50)
    rhythm_variation: float = Field(default=0.5, ge=0, le=1)
    word_spacing_variation: float = Field(default=0, ge=0, le=1)
    word_gap_min_mm: float = Field(default=0, ge=0, le=2.5)
    line_start_variation: float = Field(default=0.5, ge=0, le=1)
    line_end_variation: float = Field(default=0, ge=0, le=1)
    fit_lines_individually: bool = False
    line_spacing_variation: float = Field(default=0.5, ge=0, le=1)
    line_tilt_variation: float = Field(default=0.5, ge=0, le=1)
    baseline_variation: float = Field(default=0.5, ge=0, le=1)
    layout_seed: int = Field(default=29, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def fit_line_height(self):
        if self.line_spacing_mm < self.writing_height_mm + 1.5:
            raise ValueError("Line spacing must exceed writing height by at least 1.5 mm.")
        return self


class StyleReference(StrictModel):
    text: str = Field(min_length=1, max_length=160)
    # Normalized delta-x, delta-y, end-of-stroke; never an image or pressure trace.
    normalized_offsets: list[tuple[float, float, int]] = Field(min_length=10, max_length=8000)
    codec: Literal["graves-normalized-v1", "vasquez-native-v1"] = "graves-normalized-v1"

    @model_validator(mode="after")
    def validate_points(self):
        if any(c in self.text for c in "\r\n\t"):
            raise ValueError("A trajectory reference must describe one line of text.")
        if any(abs(x) > 50 or abs(y) > 50 or end not in (0, 1)
               for x, y, end in self.normalized_offsets):
            raise ValueError("Reference must contain normalized pen offsets and 0/1 stroke endings.")
        return self


class GenerationSettings(StrictModel):
    text: str = Field(min_length=1, max_length=4000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    bias: float = Field(default=0.85, ge=0.5, le=1.2)
    form_diversity: float = Field(default=0.5, ge=0, le=1)
    # Regenerate closed words at verified pen lifts; no per-letter deformation.
    t_refinement: bool = True
    # Legacy compatibility only; the rejected r alternate is never generated.
    cursive_r: bool = False
    prefer_cursive_r: bool = False
    consistent_style: bool = True
    quality_checks: bool = True
    reference: StyleReference | None = None
    page: PageSettings = Field(default_factory=PageSettings)
    ink: InkSettings = Field(default_factory=InkSettings)


class Readback(StrictModel):
    status: Literal["unchecked", "match", "review"] = "unchecked"
    recognized_text: str = ""
    edit_distance: int | None = Field(default=None, ge=0)
    character_error_rate: float | None = Field(default=None, ge=0)
    numbers_match: bool | None = None


class Line(StrictModel):
    text: str
    offsets: list[tuple[float, float, int]]
    seed: int
    seconds: float
    attempts: int
    primed: bool
    attention_complete: bool
    readback: Readback = Field(default_factory=Readback)
    sampling_bias: float | None = None
    sampling_form_diversity: float | None = None
    t_refinement: bool = False
    t_adapter_version: str | None = None
    t_formation: dict | None = None
    r_formation: dict | None = None
    r_selection: dict | None = None


class Document(StrictModel):
    version: Literal[1] = 1
    backend: str = "graves-epoch52"
    source_text: str
    seed: int
    bias: float
    # Old documents used the original sampler; rerender/rewrite must retain it.
    form_diversity: float = Field(default=0, ge=0, le=1)
    t_refinement: bool = False
    # Legacy compatibility only; the rejected r alternate is never generated.
    cursive_r: bool = False
    prefer_cursive_r: bool = False
    lines: list[Line]
    reference: StyleReference
    warnings: list[str] = Field(default_factory=list)
    generation_seconds: float
    pressure_source: Literal["simulated"] = "simulated"
    quality_checks: bool = False
    consistent_style: bool = True

    def withdrawn_r_warning(self) -> str | None:
        lines = [str(i + 1) for i, line in enumerate(self.lines)
                 if line.r_formation and line.r_formation.get('applied')]
        if lines:
            return f"Lines {', '.join(lines)} contain withdrawn r-style edits. Regenerate those lines to restore native letter forms."
        return None


class RenderSettings(StrictModel):
    ink: InkSettings = Field(default_factory=InkSettings)
    page: PageSettings = Field(default_factory=PageSettings)
