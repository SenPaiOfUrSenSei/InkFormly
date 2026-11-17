from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from inkform.generator import get_generator
from inkform.readable_generator import ReadableGenerator, READABLE_MODEL_DIR
from inkform.schema import GenerationSettings, StyleReference
from inkform.stroke_network import StrokeNetwork


def test_form_diversity_broadens_learned_modes_without_adding_coordinate_noise():
    network = StrokeNetwork(hidden=4, windows=1, mixtures=2)
    with torch.no_grad():
        network.density.weight.zero_()
        network.density.bias.zero_()
        network.density.bias[:2] = torch.tensor([1., -1.])
    state = SimpleNamespace(hidden=(None, None, torch.zeros(1,4)))
    original = network.distribution(state,.85,0)
    varied = network.distribution(state,.85,1)
    assert torch.allclose(original[0],torch.softmax(torch.tensor([[1.,-1.]])*1.85,-1))
    assert varied[0][0,1] > original[0][0,1]
    # Means, covariance scales, correlation and lift probability are unchanged.
    for before,after in zip(original[1:],varied[1:]):
        assert torch.equal(before,after)
    for invalid in (-.1,1.1,float('nan')):
        with pytest.raises(ValueError,match='Form diversity'):
            network.sample('Hi',None,42,.85,10,invalid)


def test_old_saved_documents_default_to_original_sampling(document):
    from inkform.schema import Document
    data = document.model_dump()
    data.pop('form_diversity')
    assert Document.model_validate(data).form_diversity == 0
    assert GenerationSettings(text='Hi').form_diversity == .5


def test_end_of_line_keeps_final_segment_and_does_not_stop_at_an_earlier_lift(monkeypatch):
    network = StrokeNetwork(hidden=4, windows=1, mixtures=2)
    attention = iter([0, 2, 2])
    moves = iter([(1., .2, 1.), (2., .3, 0.), (3., .4, 1.)])
    monkeypatch.setattr(network, "step", lambda *args: SimpleNamespace(
        attention=torch.nn.functional.one_hot(torch.tensor(next(attention)), 3)))
    monkeypatch.setattr(network, "draw", lambda *args: torch.tensor([next(moves)]))
    values, complete = network.sample("Hi", None, 42, .8, 10)
    assert complete
    assert values.tolist() == torch.tensor([[1., .2, 1.], [2., .3, 0.], [3., .4, 1.]]).tolist()


def test_step_budget_is_not_reported_as_completion(monkeypatch):
    network = StrokeNetwork(hidden=4, windows=1, mixtures=2)
    monkeypatch.setattr(network, "step", lambda *args: SimpleNamespace(attention=torch.tensor([1., 0., 0.])))
    monkeypatch.setattr(network, "draw", lambda *args: torch.tensor([[1., .2, 1.]]))
    values, complete = network.sample("Hi", None, 42, .8, 4)
    assert len(values) == 4
    assert not complete


def test_terminal_marker_requires_last_character_and_finishes_current_stroke(monkeypatch):
    network = StrokeNetwork(hidden=4, windows=1, mixtures=2)
    # A broad early end-marker signal must not end text before its last
    # character; a credible final marker must still wait for the pen lift.
    weights = iter([[1., .1, .1, .6], [.1, .1, .8, .3], [.1, .1, .8, .41], [.1, .1, .8, .41]])
    lifts = iter([1., 1., 0., 1.])
    monkeypatch.setattr(network, "step", lambda *args: SimpleNamespace(attention=torch.tensor(next(weights))))
    monkeypatch.setattr(network, "draw", lambda *args: torch.tensor([[1., .2, next(lifts)]]))
    values, complete = network.sample("Hi.", None, 42, .8, 8)
    assert complete
    assert len(values) == 4
    assert values[-1, 2] == 1


def test_final_letter_allows_a_late_crossbar_before_dominant_end_marker(monkeypatch):
    network = StrokeNetwork(hidden=4, windows=1, mixtures=2)
    weights = iter([[.1, .8, .41], [.1, .8, .41], [.1, .8, .9]])
    moves = iter([[1., 2., 1.], [1., 0., 0.], [-1., 0., 1.]])
    monkeypatch.setattr(network, "step", lambda *args: SimpleNamespace(attention=torch.tensor(next(weights))))
    monkeypatch.setattr(network, "draw", lambda *args: torch.tensor([next(moves)]))
    values, complete = network.sample("it", None, 42, .8, 8)
    assert complete and len(values) == 3
    assert values[-1].tolist() == [-1., 0., 1.]


def test_reference_codecs_are_explicit_and_old_pen_lifts_survive_conversion():
    native = StyleReference(text="Hi", normalized_offsets=[(.2, -.3, 1)] * 10, codec="vasquez-native-v1")
    assert torch.allclose(ReadableGenerator._primer(native), torch.tensor(native.normalized_offsets))
    old = native.model_copy(update={"codec": "graves-normalized-v1"})
    converted = ReadableGenerator._primer(old)
    assert torch.isfinite(converted).all()
    assert torch.equal(converted[:, 2], torch.ones(10))
    assert not torch.equal(converted[:, :2], torch.tensor(old.normalized_offsets)[:, :2])
    with pytest.raises(ValueError):
        StyleReference(text="Hi", normalized_offsets=native.normalized_offsets, codec="guessed")


def test_default_factory_selects_new_network(monkeypatch):
    import inkform.readable_generator as backend
    sentinel = object()
    monkeypatch.setattr(backend, "get_readable_generator", lambda: sentinel)
    get_generator.cache_clear()
    try:
        assert get_generator() is sentinel
    finally:
        get_generator.cache_clear()


def test_api_exposes_actual_alphabet_and_rejects_unsupported_caps_before_sampling(tmp_path, monkeypatch):
    import inkform.server as server
    for name in ("model.pt", "styles/style-2-chars.npy", "styles/style-2-strokes.npy"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.touch()
    monkeypatch.setattr(server, "READABLE_MODEL_DIR", tmp_path)
    with TestClient(server.create_app(tmp_path / "jobs")) as client:
        info = client.get("/api/model").json()
        assert info["backend"] == "Vasquez · writer 2"
        assert info["parameters"] == 3637231
        assert info["t_refinement_available"] is True
        assert info["cursive_r_available"] is False
        assert info["cursive_r_selection_available"] is True
        assert info["t_refinement_default"] is True
        assert info["t_refinement_mode"] == "closed-word-resampling-v1"
        assert all(c not in info["charset"] for c in "QXZ")
        assert all(c in info["charset"] for c in "sESAT:)")
        for text in ("Q", "X", "Z", "🖊"):
            response = client.post("/api/generate", json={"text": text, "quality_checks": False})
            assert response.status_code == 422
            assert "Unsupported characters" in response.json()["detail"]


@pytest.mark.model
@pytest.mark.skipif(not (READABLE_MODEL_DIR / "model.pt").is_file(), reason="checkpoint not downloaded")
def test_new_model_preserves_variation_and_repeatability_with_fixed_writer():
    generator = ReadableGenerator()
    settings = GenerationSettings(text="Every student studies science.", seed=42, quality_checks=False)
    first = generator.generate(settings)
    repeat = generator.generate(settings)
    alternate = generator.generate(settings.model_copy(update={"seed": 817}))
    original_sampler = generator.generate(settings.model_copy(update={"form_diversity": 0}))
    legacy_r = generator.generate(settings.model_copy(update={"cursive_r": True}))
    assert first.lines[0].offsets == legacy_r.lines[0].offsets
    assert legacy_r.cursive_r is False and legacy_r.lines[0].r_formation is None
    assert first.lines[0].offsets == repeat.lines[0].offsets
    assert first.lines[0].offsets != alternate.lines[0].offsets
    assert first.lines[0].offsets != original_sampler.lines[0].offsets
    assert first.form_diversity == first.lines[0].sampling_form_diversity == .5
    assert original_sampler.form_diversity == original_sampler.lines[0].sampling_form_diversity == 0
    assert first.reference == alternate.reference == generator.default_reference
    assert first.reference.codec == "vasquez-native-v1"
    assert first.backend == "vasquez-17900-style2"
    assert first.lines[0].attention_complete and alternate.lines[0].attention_complete
    assert len(first.lines[0].offsets) > 100
    for unsupported in "QXZ":
        with pytest.raises(ValueError, match="does not support"):
            generator.generate(settings.model_copy(update={"text": unsupported}))
