"""Non-destructive, word-local evidence for selecting complete neural takes.

No glyph boundary is used for editing. Word crops are permitted only at blank
vertical strips, so no visible stroke or join can be cut by the evaluator.
"""
import math
import re

import numpy as np

CLARITY_MODE='closed-word-resampling-v1'
CLARITY_METHODS={CLARITY_MODE,'whole-line-word-clarity-v1'}


def lexical_word(value):
    # Ignore OCR case/punctuation for an isolated word, but never merge a
    # split word ("in to" must not count as a verified "into").
    return ' '.join(''.join(c for c in value.casefold() if c.isalnum() or c.isspace()).split())


def internal_t_words(text):
    return [i for i,m in enumerate(re.finditer(r'\S+', text))
            if any(c=='t' and any(p.isalpha() for p in m.group()[:j]) for j,c in enumerate(m.group()))]


def word_regions(offsets, character_indices, text):
    """Return complete pen paths separated by verified blank word boundaries.

    Attention estimates where a space belongs; actual ink-free projection gaps
    determine the cut. Missing/ambiguous gaps return None, never a guessed crop.
    """
    values = np.asarray(offsets, dtype=float)
    labels = np.asarray(character_indices, dtype=int)
    words = list(re.finditer(r'\S+', text))
    if values.ndim!=2 or values.shape[1]!=3 or len(values)!=len(labels) or not len(values):
        return None
    if not np.isfinite(values).all() or not words:
        return None
    xy = np.cumsum(values[:, :2], axis=0)
    height = float(np.ptp(xy[:, 1]))
    if height<=0:
        return None
    low,high = float(xy[:,0].min()),float(xy[:,0].max())
    step = height/200
    if (high-low)/step>200000:
        return None
    size = int(np.ceil((high-low)/step))+1
    occupied = np.zeros(size, dtype=bool)
    for point in xy:
        occupied[min(size-1,max(0,int((point[0]-low)/step)))]=True
    for i in range(len(xy)-1):
        if values[i,2]>.5:
            continue
        a,b=sorted((xy[i,0],xy[i+1,0]))
        occupied[max(0,int(np.floor((a-low)/step))):min(size,int(np.ceil((b-low)/step))+1)]=True
    blank=np.flatnonzero(~occupied)
    runs=np.split(blank,np.flatnonzero(np.diff(blank)>1)+1) if len(blank) else []
    gaps=[(low+r[0]*step,low+r[-1]*step) for r in runs
          if len(r)>5 and r[0]>0 and r[-1]<size-1]
    cuts=[low-height*.05]
    for left,right in zip(words,words[1:]):
        indices=np.flatnonzero((labels>=left.end())&(labels<right.start()))
        if not len(indices):
            return None
        middle=float(np.median(xy[indices,0]))
        options=sorted((abs((a+b)/2-middle),(a+b)/2) for a,b in gaps
                       if a>cuts[-1] and abs((a+b)/2-middle)<.65*height)
        if not options:
            return None
        # Competing equally close gaps do not establish a word boundary.
        if len(options)>1 and options[1][0]-options[0][0]<.035*height:
            return None
        cuts.append(options[0][1])
    cuts.append(high+height*.05)
    if any(a>=b for a,b in zip(cuts,cuts[1:])):
        return None
    regions=[];assigned=0
    for match,a,b in zip(words,cuts,cuts[1:]):
        selected=np.flatnonzero((xy[:,0]>a)&(xy[:,0]<b))
        if len(selected)<2:
            return None
        # An actual line segment crossing either boundary invalidates all crops.
        for index in selected:
            if index and values[index-1,2]<=.5 and not a<xy[index-1,0]<b:
                return None
            if index+1<len(xy) and values[index,2]<=.5 and not a<xy[index+1,0]<b:
                return None
        points=xy[selected]
        result=np.column_stack((np.diff(np.vstack((np.zeros((1,2)),points)),axis=0),values[selected,2]))
        result[-1,2]=1
        owned=np.flatnonzero((labels>=match.start())&(labels<match.end()))
        covered=len(owned)>0 and np.all(np.isin(owned,selected))
        foreign=[i for i in selected if 0<=labels[i]<len(text) and not text[labels[i]].isspace()
                 and not match.start()<=labels[i]<match.end()]
        ownership=bool(covered and not foreign)
        regions.append(dict(text=match.group(),start=match.start(),stop=match.end(),
                            offsets=result.tolist(),source_indices=selected.tolist(),ownership_verified=ownership))
        assigned+=len(selected)
    return regions if assigned==len(values) else None


def assess_words(offsets, labels, text, reader):
    targets=internal_t_words(text)
    regions=word_regions(offsets,labels,text)
    if regions is None:
        return None
    rows=[]
    for index in range(len(regions)):
        region=regions[index]
        caption=region['text'].strip('.,;:!?"()[]')
        evidence=reader.evidence(region['offsets'],caption) if lexical_word(caption) else dict(recognized_text=caption,nll=0.)
        if not math.isfinite(evidence['nll']):
            return None
        rows.append(dict(word_index=index,word=region['text'],target=index in targets,**evidence,
            boundary_verified=region['ownership_verified'],
            exact=region['ownership_verified'] and lexical_word(evidence['recognized_text'])==lexical_word(caption)))
    focus=[w for w in rows if w['target']]
    context=[w for w in rows if not w['target']]
    return dict(words=rows,exact_words=sum(w['exact'] for w in focus),target_words=len(focus),
                mean_nll=sum(w['nll'] for w in focus)/max(1,len(focus)),
                context_nll=sum(w['nll'] for w in context)/max(1,len(context)))


def prefer_complete_take(baseline, candidate):
    """Require every target word to read correctly, with stronger visual evidence.

    A whole-line exact readback and unchanged reference are separate mandatory
    gates at the caller. These likelihood comparisons concern identical words;
    they are not a human-rated character formation guarantee.
    """
    if not baseline or not candidate or not baseline['words']:
        return False
    a,b=baseline['words'],candidate['words']
    if [(w['word'],w['target']) for w in a]!=[(w['word'],w['target']) for w in b]:
        return False
    if not all(w['exact'] for w in b if w['target']):
        return False
    if not all(math.isfinite(w['nll']) for w in a+b):
        return False
    if any(x['exact'] and not y['exact'] for x,y in zip(a,b)):
        return False
    if any(y['nll']>x['nll']+.25 for x,y in zip(a,b)):
        return False
    if candidate['context_nll']>baseline['context_nll']+.05:
        return False
    return candidate['mean_nll']<baseline['mean_nll']-.10


def line_word_review(line):
    """A compact UI result; do not confuse a full-line match with word proof."""
    audit=line.t_formation or {}
    if audit.get('method') not in CLARITY_METHODS:
        return None
    evidence=audit.get('selected_words')
    if evidence is None:
        return dict(status='review',words=[],reason='Word boundaries could not be checked safely.')
    targets=[dict(word=w['word'],recognized_text=w['recognized_text'],boundary_verified=w.get('boundary_verified',True),
                  exact=w['exact'] and w.get('boundary_verified',True))
             for w in evidence['words'] if w['target']]
    return dict(status='match' if all(w['exact'] for w in targets) else 'review',words=targets,reason='')


def rewrite_words_not_worse(original, candidate):
    """Preserve verified word clarity when the user requests a line retake."""
    old=original.t_formation or {};new=candidate.t_formation or {}
    if old.get('method') not in CLARITY_METHODS or not old.get('selected_words'):
        return True
    if new.get('method') not in CLARITY_METHODS or not new.get('selected_words'):
        return False
    a,b=old['selected_words'],new['selected_words']
    if [(w['word'],w['target']) for w in a['words']]!=[(w['word'],w['target']) for w in b['words']]:
        return False
    if not all(math.isfinite(w['nll']) for w in a['words']+b['words']):
        return False
    if any((x['exact'] and not y['exact']) or y['nll']>x['nll']+.25 for x,y in zip(a['words'],b['words'])):
        return False
    return b['context_nll']<=a['context_nll']+.05 and b['mean_nll']<=a['mean_nll']+.10


def place_complete_word(line_offsets, region, word_offsets):
    """Replace a closed word block; preserve every outside point and pen state.

    A newly sampled complete word receives only translation and uniform scale.
    Reject interleaved pen events, joined boundaries, or a mismatched aspect
    ratio. No letter, crossbar, connecting stroke or neighbor is deformed.
    """
    original=np.asarray(line_offsets,dtype=float)
    generated=np.asarray(word_offsets,dtype=float)
    indices=np.asarray(region['source_indices'],dtype=int)
    if not region.get('ownership_verified') or len(indices)<2 or not np.all(np.diff(indices)==1) or generated.ndim!=2 or generated.shape[1]!=3 or len(generated)<2:
        return None
    start,stop=int(indices[0]),int(indices[-1])
    if (start and original[start-1,2]<.5) or original[stop,2]<.5:
        return None
    if generated[-1,2]<.5 or not np.isfinite(generated).all():
        return None
    xy=np.cumsum(original[:,:2],axis=0);sample=np.cumsum(generated[:,:2],axis=0)
    low,high=xy[indices].min(0),xy[indices].max(0)
    a,b=sample.min(0),sample.max(0)
    old_span,new_span=high-low,b-a
    if min(*old_span,*new_span)<=0:
        return None
    scale=float(min(old_span/new_span))
    fitted_span=new_span*scale
    if fitted_span[0]<.80*old_span[0] or fitted_span[1]<.85*old_span[1]:
        return None
    # Fit inside the old word's box, keeping its baseline as close as possible.
    shift=low-a*scale
    desired=float(np.quantile(xy[indices,1],.8)-np.quantile(sample[:,1],.8)*scale)
    shift[1]=np.clip(desired,low[1]-a[1]*scale,high[1]-b[1]*scale)
    fitted=sample*scale+shift
    if abs(float(np.quantile(fitted[:,1],.8)-np.quantile(xy[indices,1],.8)))>.12*old_span[1]:
        return None
    combined=np.vstack((xy[:start],fitted,xy[stop+1:]))
    ends=np.r_[original[:start,2],generated[:,2],original[stop+1:,2]]
    result=np.column_stack((np.diff(np.vstack((np.zeros((1,2)),combined)),axis=0),ends))
    # Outside paths retain exactly their original spatial geometry, including
    # all neighboring joins. Re-differencing may introduce roundoff only.
    rebuilt=np.cumsum(result[:,:2],axis=0)
    assert np.allclose(rebuilt[:start],xy[:start],atol=1e-9,rtol=0)
    assert np.allclose(rebuilt[start+len(fitted):],xy[stop+1:],atol=1e-9,rtol=0)
    return result.tolist(),dict(start=start,stop=stop,scale=scale,translation=shift.tolist(),
        new_points=len(fitted),outside_points_preserved=True,uniform_word_placement=True,ownership_verified=True)
