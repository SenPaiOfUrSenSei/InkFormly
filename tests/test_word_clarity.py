from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from inkform.word_clarity import word_regions,internal_t_words,prefer_complete_take,line_word_review,place_complete_word,lexical_word
from inkform.readable_generator import ReadableGenerator
from inkform.recognition import HandwritingRecognizer,compare_readback


def paths(joined=False):
    xy=np.array([[0.,0.],[2.,10.],[4.,0.],[6.,8.],[8.,0.],
                 [18.,0.],[20.,10.],[22.,0.],[24.,8.],[26.,0.]])
    ends=np.zeros(len(xy));ends[-1]=1;ends[4]=0 if joined else 1
    return np.column_stack((np.diff(np.vstack((np.zeros((1,2)),xy)),axis=0),ends)).tolist()


def evidence(target=.8,neighbor=.8,exact=True):
    return dict(words=[dict(word='little',recognized_text='little' if exact else 'lille',target=True,nll=target,exact=exact),
                       dict(word='calm',recognized_text='calm',target=False,nll=neighbor,exact=True)],
                target_words=1,exact_words=int(exact),mean_nll=target,context_nll=neighbor)


def test_word_regions_preserve_every_point_and_pen_lift_and_refuse_connected_cuts():
    source=paths();labels=[0,0,1,1,2,3,3,4,4,4]
    regions=word_regions(source,labels,'at no')
    assert [r['text'] for r in regions]==['at','no']
    assert sorted(i for r in regions for i in r['source_indices'])==list(range(len(source)))
    xy=np.cumsum(np.array(source)[:,:2],axis=0)
    for region in regions:
        np.testing.assert_allclose(np.cumsum(np.array(region['offsets'])[:,:2],axis=0),xy[region['source_indices']])
        np.testing.assert_array_equal(np.array(region['offsets'])[:,2],np.array(source)[region['source_indices'],2])
    assert word_regions(paths(joined=True),labels,'at no') is None
    assert word_regions(source,labels[:-1],'at no') is None


def test_word_clarity_requires_better_t_evidence_without_weaker_neighbors():
    baseline=evidence(target=1.2)
    assert prefer_complete_take(baseline,evidence(target=.7))
    assert not prefer_complete_take(baseline,evidence(target=1.15))
    assert not prefer_complete_take(baseline,evidence(target=.7,neighbor=1.1))
    weaker_neighbor=evidence(target=.7)
    weaker_neighbor['words'][1]['exact']=False
    assert not prefer_complete_take(baseline,weaker_neighbor)
    assert not prefer_complete_take(baseline,evidence(target=.7,exact=False))
    assert not prefer_complete_take(baseline,None)
    assert not prefer_complete_take(baseline,evidence(target=float('nan')))
    assert internal_t_words("top letter SAT little at don't")==[1,3,4,5]
    assert lexical_word('in to')!=lexical_word('into')
    assert lexical_word('stiff .')==lexical_word('stiff')


def test_recognizer_likelihood_aligns_content_after_eos_without_tokenizer_bos(monkeypatch):
    import inkform.recognition as module
    monkeypatch.setattr(module,'recognition_image',lambda offsets:Image.new('RGB',(10,10),'white'))
    class Processor:
        def __call__(self,**kwargs):return SimpleNamespace(pixel_values=torch.zeros(1,3,10,10))
        def tokenizer(self,text,add_special_tokens):
            assert text=='at' and add_special_tokens is False
            return SimpleNamespace(input_ids=[7,8])
        def batch_decode(self,ids,**kwargs):return ['at']
    class Model:
        generation_config=SimpleNamespace(decoder_start_token_id=2,eos_token_id=2)
        def encoder(self,**kwargs):return SimpleNamespace(last_hidden_state=torch.zeros(1,1,1))
        def generate(self,**kwargs):
            assert 'labels' not in kwargs and 'decoder_input_ids' not in kwargs
            return torch.tensor([[2,7,8,2]])
        def __call__(self,**kwargs):
            assert kwargs['decoder_input_ids'].tolist()==[[2,7,8]]
            logits=torch.full((1,3,10),-10.)
            # A punctuation/EOS prediction must not swamp letter confidence.
            for i,token in enumerate((7,8,0)):logits[0,i,token]=10
            return SimpleNamespace(logits=logits)
    reader=object.__new__(HandwritingRecognizer);reader.processor=Processor();reader.model=Model()
    result=reader.evidence(paths(),'at')
    assert result['recognized_text']=='at' and result['nll']<.001


def test_selection_keeps_one_complete_neural_take_and_rejects_worse_context(document,monkeypatch):
    g=object.__new__(ReadableGenerator)
    old=document.lines[0].model_copy(deep=True);old.text='little calm';old.readback=compare_readback(old.text,old.text)
    old.seed=42;before=old.model_dump_json();native=[(1.,2.,1)]
    variants=[[(10.+i,20.,1)]*3 for i in range(4)]
    lookup={id(old.offsets):evidence(1.2)}
    scores=[evidence(.5,1.5),evidence(.7),evidence(.6,exact=False),None]
    calls=[]
    def trajectory(text,bias,seed,reference,diversity):
        assert reference is document.reference and diversity==.5
        index=len(calls);calls.append(seed);lookup[id(variants[index])]=scores[index]
        return variants[index],native
    monkeypatch.setattr(g,'_trajectory',trajectory)
    monkeypatch.setattr(g,'_word_evidence',lambda text,off,ref,reader:lookup[id(off)])
    g.recognizer_factory=lambda:SimpleNamespace(read=lambda offsets:old.text)
    selected,returned=g._select_t_context(old.text,.85,42,document.reference,.5,(old,None))
    assert selected.offsets==[tuple(v) for v in variants[1]]
    assert returned is native and len(calls)==4
    assert selected.t_formation['strokes_modified'] is False
    assert selected.sampling_form_diversity==.5 and selected.t_formation['applied']
    assert old.model_dump_json()==before


def test_uncertain_word_boundaries_retain_original_without_sampling(document,monkeypatch):
    g=object.__new__(ReadableGenerator);old=document.lines[0]
    monkeypatch.setattr(g,'_word_evidence',lambda *args:None)
    def unexpected(*args):raise AssertionError('No sampling without usable baseline evidence.')
    monkeypatch.setattr(g,'_trajectory',unexpected)
    g.recognizer_factory=lambda:object()
    line,native=g._select_t_context('little calm',.85,42,None,.5,(old,'native'))
    assert line.offsets==old.offsets and native=='native'
    assert not line.t_formation['applied']


def test_word_correct_candidate_with_wrong_sentence_is_rejected(document,monkeypatch):
    g=object.__new__(ReadableGenerator);old=document.lines[0].model_copy(deep=True)
    old.text='little calm';old.readback=compare_readback(old.text,old.text)
    monkeypatch.setattr(g,'_word_evidence',lambda *args:evidence(1.2))
    monkeypatch.setattr(g,'_trajectory',lambda *args:(paths(),None))
    g.recognizer_factory=lambda:SimpleNamespace(read=lambda offsets:'little wrong')
    line,_=g._select_t_context(old.text,.85,42,None,.5,(old,None))
    assert line.offsets==old.offsets and not line.t_formation['applied']
    assert len(line.t_formation['proposals'])==4


def test_inconclusive_t_words_are_flagged_even_when_full_line_matches(document):
    from inkform.generator import GravesGenerator
    line=document.lines[0];line.readback=compare_readback(line.text,line.text)
    line.t_formation=dict(method='whole-line-word-clarity-v1',selected_words=None)
    document.quality_checks=True
    assert line_word_review(line)['status']=='review'
    assert any('t-word checks' in warning for warning in GravesGenerator.quality_warnings(document))
    line.t_formation['selected_words']=evidence(target=.6)
    assert line_word_review(line)['status']=='match'
    line.t_formation['selected_words']['words'][0]['boundary_verified']=False
    assert line_word_review(line)['status']=='review'


def test_retake_cannot_replace_verified_words_with_weaker_neighbors(document,monkeypatch):
    from inkform.generator import GravesGenerator
    g=object.__new__(GravesGenerator);g.backend='vasquez-test'
    monkeypatch.setattr(g,'validate_text',lambda text:None)
    document.quality_checks=True;document.t_refinement=True
    old=document.lines[0];old.readback=compare_readback(old.text,old.text)
    old.t_formation=dict(method='whole-line-word-clarity-v1',selected_words=evidence())
    candidate=old.model_copy(deep=True);candidate.seed+=1
    candidate.t_formation['selected_words']=evidence(target=.6,neighbor=1.5)
    monkeypatch.setattr(g,'select_line',lambda *args:(candidate,None))
    before=document.model_dump_json();result=g.rewrite_line(document,0)
    assert result.lines[0]==old and document.model_dump_json()==before
    assert any('previous version' in warning for warning in result.warnings)


def sample_word():
    xy=np.array([[0.,0.],[.5,2.5],[1.,5.],[1.5,2.5],[2.,0.],[3.,4.],[4.,0.]])
    ends=np.zeros(len(xy));ends[-1]=1
    return np.column_stack((np.diff(np.vstack((np.zeros((1,2)),xy)),axis=0),ends)).tolist()


def test_complete_word_placement_preserves_neighbors_and_does_not_deform_letters():
    original=paths();region=word_regions(original,[0,0,1,1,2,3,3,4,4,4],'at no')[0]
    placed,transform=place_complete_word(original,region,sample_word())
    xy=np.cumsum(np.array(placed)[:,:2],axis=0);source=np.cumsum(np.array(original)[:,:2],axis=0)
    sampled=np.cumsum(np.array(sample_word())[:,:2],axis=0)
    np.testing.assert_allclose(xy[:7],sampled*transform['scale']+transform['translation'])
    np.testing.assert_allclose(xy[7:],source[5:],atol=1e-9,rtol=0)
    np.testing.assert_array_equal(np.array(placed)[7:,2],np.array(original)[5:,2])
    assert transform['uniform_word_placement'] and transform['outside_points_preserved'] and transform['ownership_verified']
    assert place_complete_word(paths(joined=True),region,sample_word()) is None
    interleaved={**region,'source_indices':[0,2,4]}
    assert place_complete_word(original,interleaved,sample_word()) is None
    # A detached final t outside the apparent crop must not be left behind.
    detached=word_regions(original,[0,0,1,1,2,1,3,4,4,4],'at no')[0]
    assert not detached['ownership_verified']
    assert place_complete_word(original,detached,sample_word()) is None


def test_closed_word_selection_keeps_neighbors_and_records_actual_blind_readback(document,monkeypatch):
    g=object.__new__(ReadableGenerator);old=document.lines[0].model_copy(deep=True)
    old.text='at no';old.offsets=[(float(x),float(y),int(end)) for x,y,end in paths()];old.readback=compare_readback(old.text,old.text)
    before=old.model_dump_json()
    g.t_evaluator=SimpleNamespace(character_trace=lambda *args:(None,[0,0,1,1,2,3,3,4,4,4]))
    calls=[]
    def trajectory(text,bias,seed,reference,diversity):
        calls.append((text,reference,diversity));return sample_word(),None
    monkeypatch.setattr(g,'_trajectory',trajectory)
    g.recognizer_factory=lambda:SimpleNamespace(evidence=lambda offsets,expected:dict(recognized_text='at .',nll=.2),read=lambda offsets:'at no')
    proof=dict(words=[dict(word_index=0,word='at',target=True,nll=2.,exact=False,recognized_text='af'),dict(word_index=1,word='no',target=False,nll=.3,exact=True,recognized_text='no')],exact_words=0,target_words=1,mean_nll=2.,context_nll=.3)
    line,_=g._select_t_words(old.text,.85,42,document.reference,.5,(old,None),cached_evidence=proof)
    assert line.t_formation['applied'] and line.t_formation['selected_words']['words'][0]['recognized_text']=='at .'
    assert all(text=='at' and reference is document.reference and d==.5 for text,reference,d in calls)
    after=np.cumsum(np.array(line.offsets)[:,:2],axis=0)
    np.testing.assert_allclose(after[7:],np.cumsum(np.array(old.offsets)[:,:2],axis=0)[5:],atol=1e-9)
    assert old.model_dump_json()==before
    g.recognizer_factory=lambda:SimpleNamespace(evidence=lambda offsets,expected:dict(recognized_text='at',nll=.1),read=lambda offsets:'wrong words')
    rejected,_=g._select_t_words(old.text,.85,42,document.reference,.5,(old,None),cached_evidence=proof)
    assert rejected.offsets==old.offsets and not rejected.t_formation['applied']
