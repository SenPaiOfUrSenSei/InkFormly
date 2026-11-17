"""CPU trajectory backend. No page or ink decisions belong in this module."""
from functools import lru_cache
import json
from pathlib import Path
import threading
import time
from typing import Callable, Protocol

import torch

from .schema import Document, GenerationSettings, Line, StyleReference
from .recognition import get_recognizer, compare_readback

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/graves/Epoch_52"
INFERENCE_LOCK = threading.Lock()


class GenerationError(RuntimeError):
    pass


class TrajectoryGenerator(Protocol):
    def generate(self, settings: GenerationSettings,
                 progress: Callable[[int, int, str], None] | None = None) -> Document: ...


def plan_lines(text: str, limit: int) -> list[str]:
    """Wrap at word boundaries; retain explicit line and paragraph breaks."""
    lines = []
    for original in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        words = original.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            if len(word) > limit:
                raise ValueError(f"A word exceeds {limit} characters. Add a space or line break.")
            if current and len(current) + 1 + len(word) > limit:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        lines.append(current)
    return lines


class GravesGenerator:
    def __init__(self, model_dir: Path = MODEL_DIR, recognizer_factory=get_recognizer):
        from vendor.graves.models import SynthesisNetwork
        if not (model_dir / "model.pt").is_file():
            raise GenerationError("Pretrained checkpoint missing. Run scripts/download_model.py first.")
        meta = json.loads((model_dir / "meta.json").read_text())
        self.charset = meta["charset"]
        self.backend = f"graves-{model_dir.name.lower().replace('_', '')}"
        self.mu = torch.tensor(meta["mu"])
        self.std = torch.tensor(meta["std"])
        torch.set_num_threads(1)
        self.model = SynthesisNetwork.get_default_model(len(self.charset) + 1, torch.device("cpu"), bias=0.8)
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
        self.model.eval()
        self.recognizer_factory = recognizer_factory

    def validate_text(self, text: str):
        missing = sorted(set(text) - set(self.charset) - {"\r", "\n", "\t"})
        if missing:
            raise ValueError(f"This model does not support: {', '.join(repr(c) for c in missing)}. Use the supported Latin alphabet and punctuation.")
        if not text.strip():
            raise ValueError("Enter some text to generate.")

    def _context(self, text: str):
        ids = [self.charset.index(c) + 1 for c in text]
        return torch.eye(len(self.charset) + 1)[ids].unsqueeze(0)

    @staticmethod
    def _sample(mixture):
        """Correlated Gaussian sample, avoiding a Cholesky decomposition per point."""
        pi, mu, sd, rho, eos = (v[0, 0] for v in mixture)
        component = int(torch.multinomial(pi, 1))
        count = len(pi)
        a, b = torch.randn(2)
        r = rho[component].clamp(-0.999, 0.999)
        dx = mu[component] + sd[component] * a
        dy = mu[component + count] + sd[component + count] * (r * a + torch.sqrt(1 - r * r) * b)
        return torch.stack((dx, dy, (eos.reshape(()) > 0.5).float())).reshape(1, 1, 3)

    def _trajectory(self, text: str, bias: float, seed: int, reference: StyleReference | None, form_diversity=0.0):
        if form_diversity:
            raise ValueError("Form diversity requires the Vasquez backend.")
        model = self.model
        model.mixture.bias = bias
        prefix = reference.text + "  " if reference else ""
        context = self._context(prefix + text + "  ")
        x = model.get_initial_input().reshape(1, 1, 3)
        w = model.get_initial_window(1)
        k = torch.zeros(1, model.gaussian_components)
        h1s, h2s, h3s = model.get_all_initial_states(1)
        points = []
        max_steps = min(8000, max(1200, len(text) * 85))
        target_end = len(prefix) + len(text)
        complete = False

        def step(current):
            nonlocal h1s, h2s, h3s, w, k
            h1, h1s = model.lstm1(torch.cat((current, w), dim=-1), h1s)
            phi, k = model.window(h1, context, k)
            w = model.window.matmul_3d(phi, context)
            mixture, h2s, h3s = model.compute_mixture(current, h1, w, h2s, h3s)
            return mixture, phi

        # fork_rng restores caller state; the outer lock also guards the shared model.
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(seed)
            if reference:
                step(x)
                for p in reference.normalized_offsets[:-1]:
                    x = torch.tensor(p).reshape(1, 1, 3)
                    step(x)
                # The final reference point is consumed by the first generation
                # step, exactly once. Consuming it twice can repeat a leading glyph.
                x = torch.tensor(reference.normalized_offsets[-1]).reshape(1, 1, 3)
            for _ in range(max_steps):
                mixture, phi = step(x)
                x = self._sample(mixture)
                if not torch.isfinite(x).all() or x[0, 0, :2].abs().max() > 30:
                    break
                points.append(x[0, 0].clone())
                # End only after attention has advanced through the actual content.
                if int(phi[0, 0].argmax()) >= target_end and x[0, 0, 2] > 0.5:
                    complete = True
                    break
        if not points:
            return None
        normalized = torch.stack(points)
        normalized[-1, 2] = 1
        offsets = normalized * self.std + self.mu
        positions = offsets[:, :2].cumsum(0)
        span = positions.max(0).values - positions.min(0).values
        # Reject collapsed, runaway, or obviously implausible geometry.
        ratio = float(span[0] / span[1].clamp_min(1))
        plausible = float(span[1]) > 20 and 0.06 * len(text) < ratio < max(8, len(text) * 1.5)
        if not complete or not plausible:
            return None
        return offsets.tolist(), normalized.tolist()

    def select_line(self, text, bias, seed, reference, quality_checks, form_diversity=0.0,t_refinement=False,cursive_r=False):
        """Prefer exact independent readback; otherwise retain the closest take."""
        started = time.perf_counter()
        recognizer = self.recognizer_factory() if quality_checks else None
        best, best_rank = None, (True, float("inf"))
        for attempt in range(6 if quality_checks else 4):
            candidate_seed = (seed + attempt * 7919) % (2**31 - 1)
            # Explore restrained and more varied samples when the first take
            # reads incorrectly; retain the same reference movements throughout.
            adjustment = (0, 0.15, -0.15, 0.3, -0.3, 0.05)[attempt] if quality_checks else 0
            candidate_bias = min(1.2, max(0.5, bias + adjustment))
            # Keep the writer fixed. If wording is difficult, later attempts
            # reduce learned-motion diversity, including one original-sampler
            # fallback. The retained take records its actual settings.
            diversity = form_diversity * ((1, 1, .75, .5, 0, 1)[attempt] if quality_checks else 1)
            result = self._trajectory(text, candidate_bias, candidate_seed, reference, diversity)
            if result is None:
                continue
            offsets, normalized = result
            line = Line(text=text, offsets=offsets, seed=candidate_seed, seconds=0, attempts=0,
                        primed=reference is not None, attention_complete=True, sampling_bias=candidate_bias,
                        sampling_form_diversity=diversity)
            if recognizer:
                # The requested text is only used after blind image recognition.
                line.readback = compare_readback(text, recognizer.read(offsets))
                distance = line.readback.edit_distance
                rank = (not line.readback.numbers_match, distance)
            else:
                distance = 0
                rank = (False, 0)
            if rank < best_rank:
                best, best_rank = (line, normalized), rank
            if distance == 0:
                break
        if best is None:
            raise GenerationError(f"The model could not finish line: {text!r}. Try a new take or a shorter line.")
        best[0].attempts = attempt + 1
        best[0].seconds = time.perf_counter() - started
        return best

    @staticmethod
    def quality_warnings(document):
        warnings = ["Pressure is simulated. Review the writing before printing."]
        if document.quality_checks:
            from .word_clarity import line_word_review
            pending = [str(i + 1) for i, line in enumerate(document.lines) if line.text and line.readback.status != "match"]
            if pending:
                warnings.append(f"Readback differs from the input on lines {', '.join(pending)}. Review or rewrite those lines.")
            unclear=[str(i+1) for i,line in enumerate(document.lines)
                     if line.text and (line_word_review(line) or {}).get('status')=='review']
            if unclear:
                warnings.append(f"Separate t-word checks need visual review on lines {', '.join(unclear)}. No stroke correction was applied.")
            warnings.append("Readback is an independent OCR estimate. A matching word can still have malformed letters; inspect the shapes and punctuation before printing.")
        else:
            warnings.append("Word readback is off. Attention completion does not verify spelling.")
        rejected_r = document.withdrawn_r_warning()
        if rejected_r:
            warnings.append(rejected_r)
        if document.consistent_style:
            warnings.append("Style uses trajectory priming; exact writer identity is not guaranteed by this checkpoint.")
        else:
            warnings.append("Unprimed lines may have different handwriting styles.")
        return warnings

    def generate(self, settings: GenerationSettings, progress=None) -> Document:
        self.validate_text(settings.text)
        if settings.reference:
            self.validate_text(settings.reference.text)
        page = settings.page
        limit = min(48 if settings.quality_checks else 75,
                    int((210 - 2 * page.margin_mm) / (page.writing_height_mm * 0.42)))
        planned = plan_lines(settings.text, limit)
        if len(planned) > 100:
            raise ValueError("Too many lines. Generate at most 100 lines at a time.")
        reference = settings.reference
        prefer_cursive_r = (settings.prefer_cursive_r and settings.quality_checks
                            and settings.consistent_style and getattr(self, "supports_cursive_selection", False))
        cursive_options = {"prefer_cursive_r": True} if prefer_cursive_r else {}
        # One fixed reference keeps an ordinary text change
        # from selecting a completely different default writing style.
        if reference is None and settings.consistent_style:
            reference = getattr(self,"default_reference",None)
            if reference is None:
                reference = StyleReference.model_validate_json((Path(__file__).parent / "assets/default_style.json").read_text())
        lines = []
        start = time.perf_counter()
        with INFERENCE_LOCK:
            for index, text in enumerate(planned):
                if progress:
                    progress(index, len(planned), text)
                if not text:
                    lines.append(Line(text="", offsets=[], seed=settings.seed, seconds=0, attempts=0,
                                      primed=False, attention_complete=True))
                    continue
                used_reference = reference if settings.consistent_style or settings.reference else None
                seed = (settings.seed + index * 1009) % (2**31 - 1)
                try:
                    diversity = settings.form_diversity if self.backend.startswith("vasquez") else 0
                    line, normalized = self.select_line(text, settings.bias, seed, used_reference, settings.quality_checks, diversity,
                        settings.t_refinement and settings.quality_checks and self.backend.startswith('vasquez'),
                        False, **cursive_options)
                except GenerationError as exc:
                    raise GenerationError(f"The model could not finish line {index + 1}: {text!r}. Try a new take or a shorter line.") from exc
                lines.append(line)
                if reference is None:
                    reference = StyleReference(text=text, normalized_offsets=normalized,
                                               codec="vasquez-native-v1" if self.backend.startswith("vasquez") else "graves-normalized-v1")
        if reference is None:
            raise GenerationError("No handwriting was generated.")
        document = Document(backend=self.backend, source_text=settings.text, seed=settings.seed, bias=settings.bias, lines=lines,
                            form_diversity=settings.form_diversity if self.backend.startswith("vasquez") else 0,
                            t_refinement=settings.t_refinement and settings.quality_checks and self.backend.startswith('vasquez'),
                            cursive_r=False, prefer_cursive_r=prefer_cursive_r,
                            reference=reference, generation_seconds=time.perf_counter() - start,
                            quality_checks=settings.quality_checks, consistent_style=settings.consistent_style or settings.reference is not None)
        document.warnings = self.quality_warnings(document)
        if settings.cursive_r:
            document.warnings.append("The reference-r alternate was withdrawn because it broke word legibility. Native letter forms were used.")
        return document

    def rewrite_line(self, document: Document, index: int) -> Document:
        if not 0 <= index < len(document.lines) or not document.lines[index].text:
            raise ValueError("Choose a written line to rewrite.")
        original = document.lines[index]
        self.validate_text(original.text)
        if document.consistent_style:
            self.validate_text(document.reference.text)
        seed = (original.seed + 104729) % (2**31 - 1)
        reference = document.reference if document.consistent_style else None
        cursive_options = {"prefer_cursive_r": True} if (
            document.prefer_cursive_r and getattr(self, "supports_cursive_selection", False)) else {}
        with INFERENCE_LOCK:
            diversity = document.form_diversity if self.backend.startswith("vasquez") else 0
            line, _ = self.select_line(original.text, document.bias, seed, reference, document.quality_checks, diversity,
                document.t_refinement and document.quality_checks and self.backend.startswith('vasquez'),
                False, **cursive_options)
        updated = document.model_copy(deep=True)
        updated.cursive_r = False
        retained = (document.quality_checks and original.readback.edit_distance is not None and
                    line.readback.edit_distance is not None and
                    (line.readback.numbers_match is False, line.readback.edit_distance) >
                    (original.readback.numbers_match is False, original.readback.edit_distance))
        if document.quality_checks:
            from .word_clarity import rewrite_words_not_worse
            # The rejected r modifier overwrote word scores after changing joins;
            # those scores must not force a retake to keep its malformed curves.
            if not (original.r_formation and original.r_formation.get('applied')):
                retained=retained or not rewrite_words_not_worse(original,line)
        if not retained:
            updated.lines[index] = line
            if self.backend not in updated.backend.split("+"):
                updated.backend += "+" + self.backend
        updated.generation_seconds += line.seconds
        updated.warnings = self.quality_warnings(updated)
        if retained:
            updated.warnings.append(f"The rewrite had weaker wording or word-clarity evidence; the previous version of line {index + 1} was kept.")
        return updated


@lru_cache(maxsize=1)
def get_generator() -> GravesGenerator:
    # Lazy import keeps the shared generation/review pipeline independent of
    # the implementation of each neural backend.
    from .readable_generator import get_readable_generator
    return get_readable_generator()
