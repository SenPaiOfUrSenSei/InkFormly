from types import SimpleNamespace

import numpy as np
import pytest

from inkform.r_style import cursive_r_alternate,rejected_reference_r_probe
from inkform.schema import GenerationSettings,Document,Line
from inkform.readable_generator import ReadableGenerator
from inkform.recognition import compare_readback
from inkform.generator import GravesGenerator


def sample():
    prefix=np.array([[0.,0.],[1.,-2.],[2.,0.],[3.,-1.]])
    t=np.linspace(0,1,45);r=np.column_stack((4+10*t,-8*np.sin(np.pi*t)-4*t))
    suffix=np.array([[15.,0.],[16.,-2.],[17.,0.],[18.,-1.],[19.,0.],[20.,-2.],[21.,0.]])
    xy=np.vstack((prefix,r,suffix));ends=np.zeros(len(xy));ends[-1]=1
    offsets=np.column_stack((np.diff(np.vstack((np.zeros((1,2)),xy)),axis=0),ends)).tolist()
    labels=np.r_[np.array([0,0,1,1]),np.full(45,2),np.array([3,3,4,4,5,5,5])].tolist()
    return offsets,labels


def test_reference_r_preserves_other_letters_pen_lifts_and_endpoint_tangents():
    source,labels=sample();before=np.array(source)
    styled,audit=cursive_r_alternate(source,labels,'carpet',42)
    assert len(audit)==1
    a=audit[0];start,stop=a['start'],a['stop'];count=a['new_points']
    xy=np.cumsum(before[:,:2],axis=0);after=np.cumsum(np.array(styled)[:,:2],axis=0)
    np.testing.assert_allclose(after[:start],xy[:start],atol=1e-12)
    shift=np.array([a['following_translation'],0.])
    np.testing.assert_allclose(after[start+count:],xy[stop+1:]+shift,atol=1e-12)
    np.testing.assert_allclose(after[start:start+3],xy[start:start+3],atol=1e-12)
    np.testing.assert_allclose(after[start+count-3:start+count],xy[stop-2:stop+1]+shift,atol=1e-12)
    np.testing.assert_array_equal(np.array(styled)[:start,2],before[:start,2])
    np.testing.assert_array_equal(np.array(styled)[start+count:,2],before[stop+1:,2])
    np.testing.assert_array_equal(source,before)
    assert np.isfinite(styled).all() and count>=64
    assert cursive_r_alternate(source,labels,'carpet',42)==(styled,audit)
    wider=np.array(source);wider[:,0]*=1.06
    assert cursive_r_alternate(wider,labels,'carpet',42)[0]!=styled


def test_unrelated_capital_and_lifted_r_are_not_changed():
    source,labels=sample()
    for text in ('caRpet','catpet'):
        styled,audit=cursive_r_alternate(source,labels,text)
        assert not audit and styled==source
    source[20][2]=1
    styled,audit=cursive_r_alternate(source,labels,'carpet')
    assert not audit and styled==source


def test_r_option_defaults_off_for_new_and_old_documents(document):
    assert GenerationSettings(text='carpet').cursive_r is False
    assert GenerationSettings(text='carpet',cursive_r=True).cursive_r is True
    saved=document.model_dump();saved.pop('cursive_r')
    assert Document.model_validate(saved).cursive_r is False


def test_legacy_r_retake_restores_native_forms_even_when_old_word_scores_prefer_warp(document,monkeypatch):
    g=object.__new__(GravesGenerator);g.backend='vasquez-test'
    monkeypatch.setattr(g,'validate_text',lambda text:None)
    document.cursive_r=True;document.quality_checks=True
    line=document.lines[0];line.readback=compare_readback(line.text,line.text)
    line.r_formation=dict(words=[dict(word='test',word_index=1,applied=True,after=dict(nll=.3))],applied=True)
    candidate=line.model_copy(deep=True);candidate.r_formation=None
    monkeypatch.setattr('inkform.word_clarity.rewrite_words_not_worse',lambda *args:False)
    seen=[]
    def select(*args):seen.append(args[-1]);return candidate,None
    monkeypatch.setattr(g,'select_line',select)
    result=g.rewrite_line(document,0)
    assert seen==[False] and result.lines[0]==candidate
    assert result.cursive_r is False and document.cursive_r is True
    assert not any('previous version' in warning for warning in result.warnings)


def test_generation_never_enables_withdrawn_r_even_for_legacy_requests(document,monkeypatch):
    g=object.__new__(GravesGenerator);g.backend='vasquez-test';g.default_reference=document.reference
    monkeypatch.setattr(g,'validate_text',lambda text:None)
    seen=[]
    def select(text,*args):
        seen.append(args[-1]);line=document.lines[0].model_copy(deep=True);line.text=text
        return line,document.reference.normalized_offsets
    monkeypatch.setattr(g,'select_line',select)
    result=g.generate(GenerationSettings(text='carpet',t_refinement=False,cursive_r=True))
    assert seen[-1] is False and result.cursive_r is False
    assert any('withdrawn' in warning for warning in result.warnings)
    result=g.generate(GenerationSettings(text='carpet',t_refinement=False,cursive_r=True,quality_checks=False))
    assert seen[-1] is False and result.cursive_r is False


def test_historical_probe_word_gate_does_not_establish_visual_legibility(monkeypatch):
    source,labels=sample()
    original=Line(text='carpet',offsets=source,seed=42,seconds=0,attempts=1,primed=True,attention_complete=True,readback=compare_readback('carpet','carpet'))
    snapshot=original.model_dump_json();g=object.__new__(ReadableGenerator)
    g.t_evaluator=SimpleNamespace(character_trace=lambda *args:(None,labels))
    for observed_word,observed_line in [('cape','carpet'),('carpet','wrong word')]:
        g.recognizer_factory=lambda:SimpleNamespace(evidence=lambda *args:dict(recognized_text=observed_word,nll=.2),read=lambda *args:observed_line)
        line,_=rejected_reference_r_probe(g,'carpet',42,None,(original,None))
        assert line.offsets==original.offsets and not line.r_formation['applied']
        assert original.model_dump_json()==snapshot
    g.recognizer_factory=lambda:SimpleNamespace(evidence=lambda *args:dict(recognized_text='carpet',nll=.2),read=lambda *args:'carpet')
    line,_=rejected_reference_r_probe(g,'carpet',42,None,(original,None))
    assert line.r_formation['applied'] and line.readback.status=='match'
    assert original.model_dump_json()==snapshot


@pytest.mark.parametrize('quality_checks',[False,True])
@pytest.mark.parametrize('t_refinement',[False,True])
def test_legacy_r_request_cannot_modify_native_strokes_or_bypass_t_selection(document,monkeypatch,quality_checks,t_refinement):
    g=object.__new__(ReadableGenerator)
    baseline=document.lines[0].model_copy(deep=True)
    baseline.text='A little river.'
    selected=baseline.model_copy(deep=True)
    selected.t_refinement=True
    native=object()
    calls=[]
    monkeypatch.setattr(GravesGenerator,'select_line',lambda *args:(baseline,native))
    def choose_t(*args):
        calls.append(args)
        return selected,native
    def forbidden(*args,**kwargs):
        raise AssertionError('Rejected r geometry reached production')
    monkeypatch.setattr(g,'_select_t_words',choose_t)
    monkeypatch.setattr('inkform.r_style.cursive_r_alternate',forbidden)
    monkeypatch.setattr('inkform.r_style.rejected_reference_r_probe',forbidden)
    snapshot=baseline.model_dump_json()
    line,normalized=g.select_line(baseline.text,.85,42,None,quality_checks,.5,t_refinement,True)
    assert line is (selected if quality_checks and t_refinement else baseline)
    assert normalized is native and bool(calls)==(quality_checks and t_refinement)
    assert baseline.model_dump_json()==snapshot
    assert line.r_formation is None


def test_loading_withdrawn_r_strokes_warns_without_silently_rewriting_document(document):
    saved=document.model_dump_json()
    document.lines[0].r_formation=dict(applied=True)
    document.cursive_r=False
    for checked in (True,False):
        document.quality_checks=checked
        snapshot=document.model_dump_json()
        assert any('withdrawn r-style edits' in w for w in GravesGenerator.quality_warnings(document))
        assert document.model_dump_json()==snapshot
    restored=Document.model_validate_json(saved)
    assert not any('withdrawn r-style edits' in w for w in GravesGenerator.quality_warnings(restored))


def test_legacy_r_retake_still_rejects_worse_line_wording(document,monkeypatch):
    g=object.__new__(GravesGenerator);g.backend='vasquez-test'
    monkeypatch.setattr(g,'validate_text',lambda text:None)
    document.cursive_r=True;document.quality_checks=True
    old=document.lines[0];old.readback=compare_readback(old.text,old.text)
    old.r_formation=dict(applied=True)
    candidate=old.model_copy(deep=True);candidate.r_formation=None
    candidate.readback=compare_readback(old.text,'wrong words')
    monkeypatch.setattr(g,'select_line',lambda *args:(candidate,None))
    result=g.rewrite_line(document,0)
    assert result.lines[0]==old and result.cursive_r is False
    assert any('withdrawn r-style edits' in w for w in result.warnings)
    assert any('previous version' in w for w in result.warnings)
