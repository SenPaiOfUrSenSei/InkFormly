import numpy as np

from inkform.cursive_selection import cursive_r_scores
from inkform.generator import GravesGenerator
from inkform.readable_generator import ReadableGenerator
from inkform.recognition import compare_readback
from inkform.schema import GenerationSettings


def test_cursive_preference_scores_a_continuous_leg_and_distinct_exit():
    points = np.array([
        [0, 0], [.2, -.1], [.45, -.38], [.72, -.70], [.83, -1.],
        [.88, -1.03], [.94, -.95], [.91, -.78], [.78, -.42], [.65, -.08],
        [.68, 0], [.83, -.07], [1.02, -.27], [1.25, -.5],
    ]) * 10
    moves = np.diff(points, axis=0, prepend=[[0, 0]])
    offsets = np.column_stack((moves, np.zeros(len(points))))
    offsets[-1, 2] = 1
    labels = [0] * 11 + [1] * 3
    score = cursive_r_scores(offsets, labels, "ra")
    assert score[0] >= .68

    narrower_exit = points.copy()
    narrower_exit[-3:, 0] = [6.8, 7.1, 7.5]
    offsets[:, :2] = np.diff(narrower_exit, axis=0, prepend=[[0, 0]])
    assert cursive_r_scores(offsets, labels, "ra")[0] < score[0]
    offsets[5, 2] = 1
    assert cursive_r_scores(offsets, labels, "ra")[0] == 0


def test_cursive_selection_uses_current_writer_for_generation_and_retake(document, monkeypatch):
    generator = object.__new__(ReadableGenerator)
    generator.backend = "vasquez-test"
    generator.default_reference = document.reference
    monkeypatch.setattr(generator, "validate_text", lambda _: None)
    line = document.lines[0].model_copy(update={"text": "river", "readback": compare_readback("river", "river")})
    monkeypatch.setattr(GravesGenerator, "select_line", lambda *args: (line, document.reference.normalized_offsets))
    calls = []

    def select_words(generator, baseline, reference, bias, diversity, seed):
        calls.append((baseline[0], reference, seed))
        return baseline

    monkeypatch.setattr("inkform.cursive_selection.select_cursive_words", select_words)
    settings = GenerationSettings(text="river", prefer_cursive_r=True)
    result = generator.generate(settings)
    assert result.prefer_cursive_r
    assert calls == [(line, document.reference, settings.seed)]
    generator.rewrite_line(result, 0)
    assert len(calls) == 2 and calls[1][1] == result.reference
    assert calls[1][2] != calls[0][2]
    for flags in ({"quality_checks": False}, {"consistent_style": False}):
        unchecked = generator.generate(settings.model_copy(update=flags))
        assert not unchecked.prefer_cursive_r
    assert len(calls) == 2


def test_reference_r_preserves_attachment_points_and_rejects_separate_letters():
    from inkform.cursive_selection import RContour
    from inkform.joined_r import fit_joined_r

    points = np.array([
        [-5, -5], [-3, -1], [-1, .1], [0, 0], [1, -1], [2, -3], [3, -5.5],
        [4, -8], [5, -10], [4.9, -8.5], [4.7, -7], [4.5, -5], [4.3, -3],
        [4.2, -1.5], [4.25, -.5], [4.5, 0], [6, -.35], [8, -1.5],
        [10, -3], [12, -5], [14, -7], [16, -9], [18, -8], [19, -7],
    ], dtype=float)
    labels = [0] * 4 + [1] * 12 + [2] * 8
    offsets = np.column_stack((np.diff(points, axis=0, prepend=[[0, 0]]), np.zeros(len(points))))
    offsets[-1, 2] = 1
    contour = RContour(character=1, entry=5, crest=8, foot=15, exit=18, score=.7)
    result, changes = fit_joined_r(offsets, labels, "ara", [contour])
    assert len(changes) == 1
    start, stop = changes[0]["start"], changes[0]["stop"]
    drawn = np.asarray(result)
    np.testing.assert_array_equal(drawn[:, 2], offsets[:, 2])
    np.testing.assert_array_equal(drawn[:start + 1], offsets[:start + 1])
    np.testing.assert_array_equal(drawn[stop + 1:], offsets[stop + 1:])
    positions = drawn[:, :2].cumsum(axis=0)
    np.testing.assert_allclose(positions[[start, stop]], points[[start, stop]], atol=1e-10)
    assert not np.allclose(positions[start + 1:stop], points[start + 1:stop])

    # A lift immediately before the r means there is no incoming connection.
    offsets[3, 2] = 1
    _, changes = fit_joined_r(offsets, labels, "ara", [contour])
    assert not changes

    # A different stroke across the new body makes this fit unsafe.
    offsets[3, 2] = 0
    crossed = np.vstack((offsets, [[-19, 1, 0], [18, 0, 1]]))
    _, changes = fit_joined_r(crossed, labels + [2, 2], "ara", [contour])
    assert not changes
