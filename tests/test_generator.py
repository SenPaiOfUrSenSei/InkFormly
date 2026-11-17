from pathlib import Path

import pytest

from inkform.generator import GenerationError, GravesGenerator, MODEL_DIR, plan_lines
from inkform.schema import GenerationSettings, StyleReference


def test_word_wrap_retains_paragraphs():
    assert plan_lines("one two three\n\nfour five", 8) == ["one two", "three", "", "four", "five"]
    with pytest.raises(ValueError, match="word exceeds"):
        plan_lines("unbrokenword", 5)


def test_reference_rejects_nonfinite_or_invalid_pen_states():
    with pytest.raises(ValueError):
        StyleReference(text="Hi", normalized_offsets=[(float("nan"), 0, 0)] * 10)
    with pytest.raises(ValueError):
        StyleReference(text="Hi", normalized_offsets=[(0, 0, 2)] * 10)


def test_exhausted_retries_raise_instead_of_exporting_bad_handwriting(monkeypatch):
    generator = object.__new__(GravesGenerator)
    generator.charset = "abcdefghijklmnopqrstuvwxyz "
    generator.backend = "graves-epoch52"
    monkeypatch.setattr(generator, "_trajectory", lambda *args: None)
    with pytest.raises(GenerationError, match="could not finish line 1"):
        generator.generate(GenerationSettings(text="some words", quality_checks=False))


@pytest.mark.model
@pytest.mark.skipif(not (MODEL_DIR / "model.pt").is_file(), reason="checkpoint not downloaded")
def test_real_model_is_repeatable_and_rejects_unknown_characters():
    generator = GravesGenerator()
    settings = GenerationSettings(text="A little variation makes handwriting natural.", seed=42, quality_checks=False)
    a = generator.generate(settings)
    b = generator.generate(settings)
    assert a.lines[0].offsets == b.lines[0].offsets
    assert a.lines[0].attention_complete
    assert len(a.lines[0].offsets) > 100
    with pytest.raises(ValueError, match="does not support"):
        generator.generate(GenerationSettings(text="Hello 🖊"))
