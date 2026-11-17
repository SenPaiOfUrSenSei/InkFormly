import numpy as np
import pytest

from inkform.t_crossbar import raise_t_crossbars
from test_t_stem import trajectory


def test_crossbar_moves_up_preserving_stem_other_letters_endpoints_and_pen_lifts():
    native,_=trajectory()
    stems=[dict(character_index=1,stem_peak=25,downstroke_end=45)]
    before=native.copy()
    result,audit=raise_t_crossbars(native,stems)
    assert len(audit)==1
    xy=np.cumsum(native[:,:2],axis=0);after=np.cumsum(np.asarray(result)[:,:2],axis=0)
    np.testing.assert_array_equal(native,before)
    np.testing.assert_array_equal(np.asarray(result)[:,2],native[:,2])
    changed=audit[0]['changed_points']
    assert set(changed).issubset(range(46,67))
    assert np.all(after[changed,1]>xy[changed,1])
    assert audit[0]['height_before']<audit[0]['height_after']<=.68
    assert audit[0]['maximum_rise']<=.22*.85*14+1e-9
    untouched=np.setdiff1d(np.arange(len(native)),changed)
    np.testing.assert_allclose(after[untouched],xy[untouched],atol=1e-12)
    # Repeated candidates cannot move a shared crossing twice.
    duplicate,repeated=raise_t_crossbars(native,stems*2)
    assert repeated==audit
    np.testing.assert_array_equal(result,duplicate)


def test_no_crossbar_and_already_high_crossbars_are_unchanged():
    for missing in (False,True):
        native,_=trajectory(crossbar=not missing)
        if not missing:
            xy=np.cumsum(native[:,:2],axis=0);xy[46:,1]=12
            native[:,:2]=np.diff(np.vstack((np.zeros((1,2)),xy)),axis=0)
        result,audit=raise_t_crossbars(native,[dict(character_index=1,stem_peak=25,downstroke_end=45)])
        assert not audit
        np.testing.assert_array_equal(result,native)


def test_zero_strength_is_identity_and_invalid_strength_is_rejected():
    native,_=trajectory();stems=[dict(character_index=1,stem_peak=25,downstroke_end=45)]
    result,audit=raise_t_crossbars(native,stems,strength=0)
    assert not audit
    np.testing.assert_array_equal(result,native)
    with pytest.raises(ValueError):raise_t_crossbars(native,stems,strength=2)
