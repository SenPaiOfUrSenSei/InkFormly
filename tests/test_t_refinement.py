import math
from types import SimpleNamespace

import pytest
import torch

from inkform.recognition import compare_readback
from inkform.schema import Document,GenerationSettings
from inkform.stroke_network import StrokeNetwork,ALPHABET
from inkform.t_adapter import TStrokeAdapter,motion_nll
from inkform.t_formation import TFormationEvaluator,prefer_t_candidate


def test_lowercase_focus_excludes_primer_and_emphasizes_internal_t():
    prefix='test '
    text=prefix+'top letter'
    profile=StrokeNetwork.t_focus_profile(text,len(prefix),include_uppercase=False)
    assert torch.count_nonzero(profile[:len(prefix)])==0
    assert profile[len(prefix)]==pytest.approx(.45)
    assert profile[text.index('tt')]==1
    assert profile[text.index('tt')+1]==1
    assert torch.count_nonzero(StrokeNetwork.t_focus_profile('SAT STATE',include_uppercase=False))==0


def test_zero_initialized_residual_is_identity_and_likelihood_matches_standard_gaussian():
    adapter=TStrokeAdapter(hidden=4,rank=2,outputs=7)
    assert torch.count_nonzero(adapter(torch.ones(3,4)))==0
    raw=torch.zeros(2,7);target=torch.tensor([[0.,0.,0.],[1.,0.,1.]])
    nll=motion_nll(raw,target,mixtures=1)
    assert nll[0].item()==pytest.approx(math.log(2*math.pi)+math.log(2),abs=1e-6)
    assert (nll[1]-nll[0]).item()==pytest.approx(.5,abs=1e-6)


@pytest.mark.parametrize('text',['room','SAT STATE TITLE'])
def test_loaded_nonzero_adapter_does_not_change_inputs_without_lowercase_t(text):
    network=StrokeNetwork(hidden=4,windows=1,mixtures=2)
    network.t_adapter=TStrokeAdapter(hidden=4,rank=2,outputs=13)
    with torch.no_grad():network.t_adapter.up.bias.fill_(.4)
    plain,done=network.sample(text,None,123,.85,20,.5)
    refined,refined_done=network.sample(text,None,123,.85,20,.5,t_clarity=1)
    assert torch.equal(plain,refined) and done==refined_done


def test_model_contrast_uses_internal_lowercase_t_and_preserves_text_positions():
    class Model:
        def initial_state(self):return None
        def step(self,pen,state,text):
            hidden=text[:,1,:]
            return SimpleNamespace(hidden=(None,None,hidden),attention=torch.tensor([[0.,1.,0.]]))
        def density(self,hidden):
            raw=torch.zeros(1,7)
            raw[:,4]=4*(1-hidden[:,ALPHABET.index('t')])
            return raw
    # One-component density for this test stub.
    import inkform.t_formation as module
    original=module.motion_nll
    try:
        module.motion_nll=lambda raw,target:motion_nll(raw,target,mixtures=1)
        evaluator=TFormationEvaluator(Model(),lambda reference:None)
        score=evaluator.score('at',[(0.,0.,1)]*12,None)
        assert score['targets']==1 and score['margin']==pytest.approx(8,abs=1e-5)
        assert evaluator.score('to',[(0.,0.,1)]*12,None) is None
        assert evaluator.score('AT',[(0.,0.,1)]*12,None) is None
    finally:module.motion_nll=original


def test_t_replacement_requires_nonworse_wording_and_stronger_t_evidence(document):
    old=document.lines[0].model_copy(deep=True);old.text='A letter.'
    old.readback=compare_readback(old.text,old.text)
    candidate=old.model_copy(deep=True)
    shape=lambda margin:dict(targets=2,effective_steps=40.,margin=margin)
    assert prefer_t_candidate(old,candidate,shape(.2),shape(.3))
    assert not prefer_t_candidate(old,candidate,shape(.2),shape(.21))
    candidate.readback=compare_readback(old.text,'A latter.')
    assert not prefer_t_candidate(old,candidate,shape(.2),shape(.5))
    candidate.readback=old.readback
    assert not prefer_t_candidate(old,candidate,shape(.2),shape(float('nan')))
    candidate.text='Changed text'
    assert not prefer_t_candidate(old,candidate,shape(.2),shape(.5))


def test_old_documents_keep_policy_and_new_requests_enable_whole_line_word_checks(document):
    saved=document.model_dump();saved.pop('t_refinement')
    assert Document.model_validate(saved).t_refinement is False
    assert GenerationSettings(text='A letter.').t_refinement is True
    assert GenerationSettings(text='A letter.',t_refinement=True).t_refinement is True
