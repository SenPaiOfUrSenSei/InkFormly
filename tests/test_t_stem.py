import numpy as np
from types import SimpleNamespace

from inkform.t_stem import refine_internal_t_stems
from inkform.readable_generator import ReadableGenerator
from inkform.recognition import compare_readback
from inkform.generator import GravesGenerator


def trajectory(crossbar=True):
    lead=np.array([[-5.,0.],[-4.,1.],[-3.,2.],[-2.,1.],[-1.,0.],[0.,0.]])
    up=np.column_stack((np.linspace(0,10,20),np.linspace(0,14,20)))
    down=np.column_stack((np.linspace(10,5,20),np.linspace(14,0,20)))
    stem=np.vstack((lead,up,down))
    labels=np.r_[np.zeros(len(lead),int),np.ones(len(up)+len(down),int)]
    ends=np.zeros(len(stem));ends[-1]=1
    if crossbar:
        bar=np.column_stack((np.linspace(2,13,21),np.full(21,8.)))
        stem=np.vstack((stem,bar));labels=np.r_[labels,np.full(len(bar),2)]
        ends=np.r_[ends,np.zeros(len(bar))];ends[-1]=1
    offsets=np.column_stack((np.diff(np.vstack((np.zeros((1,2)),stem)),axis=0),ends))
    return offsets,labels


def test_refinement_uses_own_stem_and_preserves_crossbar_joins_and_other_letters():
    source,labels=trajectory();before=source.copy()
    refined,audit=refine_internal_t_stems(source,labels,'at')
    refined=np.array(refined);assert len(audit)==1
    changed=np.array(audit[0]['changed_points']);assert len(changed)>5
    xy=np.cumsum(source[:,:2],0);after=np.cumsum(refined[:,:2],0)
    np.testing.assert_array_equal(source,before)
    np.testing.assert_array_equal(refined[:,2],source[:,2])
    np.testing.assert_allclose(after[:,1],xy[:,1],atol=1e-12)
    untouched=np.setdiff1d(np.arange(len(source)),changed)
    np.testing.assert_allclose(after[untouched],xy[untouched],atol=1e-12)
    assert np.all(labels[changed]==1)
    assert np.all(after[changed,0]>=xy[changed,0])
    # The correction remains inside the generated character's envelope.
    assert after[changed,0].max()<=xy[labels==1,0].max()+1e-9


def test_missing_crossbar_initial_t_and_capital_T_are_left_intact():
    source,labels=trajectory(crossbar=False)
    refined,audit=refine_internal_t_stems(source,labels,'at')
    assert not audit;np.testing.assert_array_equal(source,refined)
    source,labels=trajectory()
    for text in ('AT','ta'):
        refined,audit=refine_internal_t_stems(source,labels,text)
        assert not audit;np.testing.assert_array_equal(source,refined)


def test_decoder_falls_back_to_gentler_correction_and_never_accepts_worse_readback(document):
    native,labels=trajectory();rendered=native.copy();rendered[:,:2]*=20;rendered[:,1]*=-1
    old=document.lines[0].model_copy(deep=True);old.text='at';old.offsets=[(float(x),float(y),int(end)) for x,y,end in rendered];old.readback=compare_readback('at','at')
    before=old.model_dump_json()
    g=object.__new__(ReadableGenerator)
    g.t_evaluator=SimpleNamespace(character_trace=lambda *args:(native.tolist(),labels))
    reads=iter(['al','at'])
    g.recognizer_factory=lambda:SimpleNamespace(read=lambda *args:next(reads))
    line,_=g._refine_t_line('at',.85,42,None,.5,(old,native.tolist()))
    assert line.t_formation['applied'] and line.t_formation['strength']==.55
    assert line.readback.status=='match'
    assert old.model_dump_json()==before
    g.recognizer_factory=lambda:SimpleNamespace(read=lambda *args:'al')
    retained,_=g._refine_t_line('at',.85,42,None,.5,(old,native.tolist()))
    assert retained.offsets==old.offsets and not retained.t_formation['applied']


def test_normal_checked_generation_preserves_baseline_strokes_without_geometry(monkeypatch,document):
    g=object.__new__(ReadableGenerator)
    baseline=(document.lines[0],[(0.,0.,1)])
    monkeypatch.setattr(GravesGenerator,'select_line',lambda *args,**kwargs:baseline)
    def unexpected(*args,**kwargs):
        raise AssertionError('Default generation must not alter neural trajectories.')
    monkeypatch.setattr(g,'_refine_t_line',unexpected)
    monkeypatch.setattr(g,'_select_t_context',lambda *args:baseline)
    monkeypatch.setattr(g,'_select_t_words',lambda *args:baseline)
    assert g.select_line('A little thought.',.85,42,None,True,.5) is baseline
    assert g.select_line('A little thought.',.85,42,None,True,.5,True) is baseline
    assert g.select_line('A little thought.',.85,42,None,False,.5,True) is baseline


def test_even_explicit_stem_experiment_does_not_raise_crossbars(monkeypatch,document):
    import inkform.t_crossbar
    def unexpected(*args,**kwargs):
        raise AssertionError('Crossbar raising was withdrawn after visual regression.')
    monkeypatch.setattr(inkform.t_crossbar,'raise_t_crossbars',unexpected)
    native,labels=trajectory();rendered=native.copy();rendered[:,:2]*=20;rendered[:,1]*=-1
    old=document.lines[0].model_copy(deep=True);old.text='at'
    old.offsets=[(float(x),float(y),int(end)) for x,y,end in rendered]
    old.readback=compare_readback('at','at')
    g=object.__new__(ReadableGenerator)
    g.t_evaluator=SimpleNamespace(character_trace=lambda *args:(native.tolist(),labels))
    g.recognizer_factory=lambda:SimpleNamespace(read=lambda *args:'at')
    result,_=g._refine_t_line('at',.85,42,None,.5,(old,native.tolist()))
    assert result.t_formation['applied']
    assert 'crossbar_adjustments' not in result.t_formation
