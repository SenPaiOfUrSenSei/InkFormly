import pytest

from inkform.generator import GravesGenerator
from inkform.recognition import canonical_text, compare_readback


@pytest.mark.parametrize("expected,readback", [
    ("The year was 1947.", "The year was 1948."),
    ("First: observe.", "First; observe."),
    ("Water is important.", "water is important."),
    ("a little different", "a different"),
    ("this is correct", "this this is correct"),
])
def test_readback_flags_numbers_punctuation_case_missing_and_repeated_words(expected, readback):
    result = compare_readback(expected, readback)
    assert result.status == "review"
    assert result.edit_distance > 0


def test_readback_ignores_only_spacing_and_typographic_quotes():
    assert canonical_text("A line , with words .") == "A line, with words."
    assert compare_readback("Writer's notes.", "Writer’s notes .").status == "match"
    assert compare_readback("The year was 2026.", "The year was 2021.").numbers_match is False


def candidate_generator(document, observations, monkeypatch):
    generator = object.__new__(GravesGenerator)
    generator.charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ."
    generator.backend = "graves-epoch52"
    reads = iter(observations)

    class Recognizer:
        calls = 0

        def read(self, offsets):
            self.calls += 1
            return next(reads)

    recognizer = Recognizer()
    generator.recognizer_factory = lambda: recognizer
    monkeypatch.setattr(generator, "_trajectory", lambda *args: (document.lines[0].offsets, document.reference.normalized_offsets))
    return generator, recognizer


def test_candidate_selection_retries_then_stops_at_matching_readback(document, monkeypatch):
    generator, recognizer = candidate_generator(document, ["A bad line", "A test line"], monkeypatch)
    line, _ = generator.select_line("A test line", 0.8, 42, None, True)
    assert line.readback.status == "match"
    assert line.attempts == recognizer.calls == 2
    assert line.seed == 42 + 7919


def test_candidate_selection_keeps_best_and_exposes_unresolved_mismatch(document, monkeypatch):
    generator, recognizer = candidate_generator(document, ["bad", "A test lines", "wrong", "wrong", "wrong", "wrong"], monkeypatch)
    line, _ = generator.select_line("A test line", 0.8, 42, None, True)
    assert line.readback.status == "review"
    assert line.readback.recognized_text == "A test lines"
    assert line.attempts == recognizer.calls == 6
    assert line.seed == 42 + 7919


def test_diversity_retries_include_original_sampler_and_record_retained_settings(document,monkeypatch):
    generator, _ = candidate_generator(document,['wrong']*4+['A test line'],monkeypatch)
    seen = []
    def trajectory(text,bias,seed,reference,diversity):
        seen.append((reference,diversity))
        return document.lines[0].offsets,document.reference.normalized_offsets
    monkeypatch.setattr(generator,'_trajectory',trajectory)
    line,_ = generator.select_line('A test line',.85,42,document.reference,True,.6)
    assert [d for _,d in seen] == pytest.approx([.6,.6,.45,.3,0])
    assert all(reference is document.reference for reference,_ in seen)
    assert line.sampling_form_diversity == 0
    assert line.readback.status == 'match'


def test_correct_numbers_take_priority_over_fewer_other_ocr_differences(document, monkeypatch):
    observations = ["We counted 24 samples.", "We recorded 25 samples."] + ["We counted 24 samples."] * 4
    generator, _ = candidate_generator(document, observations, monkeypatch)
    line, _ = generator.select_line("We counted 25 samples.", 0.8, 42, None, True)
    assert line.readback.recognized_text == "We recorded 25 samples."
    assert line.readback.numbers_match
    assert line.readback.status == "review"


def test_single_line_rewrite_preserves_other_lines_and_never_replaces_a_better_readback(document, monkeypatch):
    document.lines.append(document.lines[0].model_copy(deep=True))
    document.quality_checks = True
    document.lines[0].readback = compare_readback("A test line", "A test line")
    before = document.model_dump_json()
    generator, _ = candidate_generator(document, ["incorrect"] * 6, monkeypatch)
    result = generator.rewrite_line(document, 0)
    assert result.lines == document.lines
    assert document.model_dump_json() == before
    assert any("previous version" in warning for warning in result.warnings)
