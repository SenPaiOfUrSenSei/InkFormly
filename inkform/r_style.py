"""Rejected reference-r experiment, retained only for research reproducibility.

User review found broken, unreadable joins despite matching OCR. Production
handwriting generation must never apply these geometric modifications.
"""
import numpy as np


def _reference_curve(samples):
    # Normalized interpretation of the user's reference: rise, small shoulder
    # loop, descending leg and connected exit. No pressure is inferred.
    p=np.array([[0.,0.],[.30,.35],[.57,.88],[.61,1.],[.69,.96],
        [.70,.85],[.63,.78],[.83,.74],[.70,.49],[.51,.20],
        [.48,.07],[.65,.05],[.94,.32],[1.06,.48]])
    t=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];t/=t[-1]
    slopes=np.empty_like(p);slopes[0]=(p[1]-p[0])/(t[1]-t[0]);slopes[-1]=(p[-1]-p[-2])/(t[-1]-t[-2])
    slopes[1:-1]=(p[2:]-p[:-2])/(t[2:]-t[:-2])[:,None]
    s=np.linspace(0,1,samples);i=np.clip(np.searchsorted(t,s,side='right')-1,0,len(p)-2)
    dt=t[i+1]-t[i];u=(s-t[i])/dt
    curve=((2*u**3-3*u**2+1)[:,None]*p[i]+(u**3-2*u**2+u)[:,None]*dt[:,None]*slopes[i]
        +(-2*u**3+3*u**2)[:,None]*p[i+1]+(u**3-u**2)[:,None]*dt[:,None]*slopes[i+1])
    return s,curve


def cursive_r_alternate(offsets,character_indices,text,seed=42):
    data=np.asarray(offsets,dtype=float);labels=np.asarray(character_indices,dtype=int)
    if data.ndim!=2 or data.shape[1]!=3 or len(labels)!=len(data) or not np.isfinite(data).all():
        raise ValueError('r styling needs aligned, finite pen trajectories.')
    xy=np.cumsum(data[:,:2],axis=0);styled=xy.copy();ends=data[:,2].copy();audit=[]
    word_low,word_high=float(xy[:,1].min()),float(xy[:,1].max());word_height=word_high-word_low
    for position in reversed(range(len(text))):
        character=text[position]
        if character!='r':continue
        ids=np.flatnonzero(labels==position)
        if len(ids)<20 or np.any(np.diff(ids)!=1):continue
        # A leading single-point pen-up mark is not part of the joined body.
        if data[ids[0],2]>.5:ids=ids[1:]
        if len(ids)<20 or np.any(data[ids[:-1],2]>.5):continue
        original=xy[ids];a,b=original[0],original[-1]
        width=float(b[0]-a[0]);height=float(np.ptp(original[:,1]))
        if height<2 or not .3*height<width<3*height:continue
        rng=np.random.default_rng(np.random.SeedSequence([seed,position,203]))
        height*=rng.uniform(.97,1.03)
        target_width=float(np.clip(width,.95*height,1.30*height))
        following_shift=target_width-width
        # Extra samples resolve the small shoulder loop without angular corners.
        count=max(64,len(ids));source_s=np.linspace(0,1,len(ids))
        s=np.r_[source_s[:3],np.linspace(source_s[2],source_s[-3],count-4)[1:-1],source_s[-3:]]
        dense_s,p=_reference_curve(384)
        p=np.column_stack((np.interp(s,dense_s,p[:,0]),np.interp(s,dense_s,p[:,1])))
        previous=np.column_stack((np.interp(s,source_s,original[:,0]),np.interp(s,source_s,original[:,1])))
        line_y=a[1]+s*(b[1]-a[1]);delta=-height*(p[:,1]-s*.48)
        # Keep the r close to the existing x-height rather than adding a tall
        # ascender or a new descender to the writer's style.
        caps=[1.]
        up=delta<-1e-9;down=delta>1e-9
        if up.any():caps.append(float(np.min((line_y[up]-(word_low-.06*word_height))/-delta[up])))
        if down.any():caps.append(float(np.min(((word_high+.03*word_height)-line_y[down])/delta[down])))
        amplitude=max(0.,min(caps))
        if amplitude<.35:continue
        target=np.column_stack((a[0]+p[:,0]/1.06*width,
            line_y+amplitude*delta))
        target[:,0]=a[0]+p[:,0]/1.06*target_width
        # A wider shoulder needs actual room. Move the remaining word rigidly
        # rather than squeezing or bending its following letter shapes.
        previous[:,0]+=following_shift*(s*s*(3-2*s))
        def smooth(z):
            z=np.clip(z,0,1);return z**3*(10-15*z+6*z*z)
        # Keep three source points at either end. Entry positions and both
        # tangents are preserved; a wider exit translates the following word
        # paths rigidly, without bending their shapes.
        edge=source_s[2]
        weight=smooth((s-edge)/.16)*smooth((1-s-edge)/.16)
        replacement=previous*(1-weight[:,None])+target*weight[:,None]
        replacement[:3]=original[:3];replacement[-3:]=original[-3:]+np.array([following_shift,0.])
        new_ends=np.zeros(len(replacement));new_ends[:3]=data[ids[:3],2];new_ends[-3:]=data[ids[-3:],2]
        following=styled[ids[-1]+1:]+np.array([following_shift,0.])
        styled=np.vstack((styled[:ids[0]],replacement,following))
        ends=np.r_[ends[:ids[0]],new_ends,ends[ids[-1]+1:]]
        audit.append(dict(character_index=position,start=int(ids[0]),stop=int(ids[-1]),
            endpoint_neighborhoods_preserved=True,following_translation=following_shift,
            new_points=len(replacement),amplitude=amplitude))
    result=np.column_stack((np.diff(np.vstack((np.zeros((1,2)),styled)),axis=0),ends)) if audit else data.copy()
    return result.tolist(),audit


def rejected_reference_r_probe(generator,text,seed,reference,baseline):
    # Historical experiment only; user review rejected its word legibility.
    import math
    import time
    from .recognition import compare_readback
    from .word_clarity import word_regions,place_complete_word,lexical_word
    started=time.perf_counter();original=baseline[0];line=original.model_copy(deep=True)
    _,labels=generator.t_evaluator.character_trace(text,original.offsets,reference)
    regions=word_regions(original.offsets,labels,text);reader=generator.recognizer_factory()
    audit=dict(method='reference-cursive-r-v1',style_seed=seed,words=[],applied=False,source='reference-shaped geometric alternate; weights unchanged')
    if regions:
        for index in reversed(range(len(regions))):
            region=regions[index]
            if 'r' not in region['text']:continue
            row=dict(word=region['text'],word_index=index,applied=False);audit['words'].append(row)
            if not region['ownership_verified']:
                row['reason']='Uncertain word ownership.';continue
            ids=region['source_indices'];local_labels=np.asarray(labels)[ids]-region['start']
            styled,changes=cursive_r_alternate(region['offsets'],local_labels,region['text'],(seed+index*1009)%(2**31-1))
            if not changes:row['reason']='No unambiguous continuous r body.';continue
            placed=place_complete_word(line.offsets,region,styled)
            if placed is None:row['reason']='Incompatible complete-word placement.';continue
            offsets,transform=placed
            xy=np.asarray(offsets)[:,:2].cumsum(0);ends=np.asarray(offsets)[:,2]
            start=transform['start'];stop=start+transform['new_points']
            crop=np.column_stack((np.diff(np.vstack((np.zeros((1,2)),xy[start:stop])),axis=0),ends[start:stop])).tolist()
            caption=region['text'].strip('.,;:!?"()[]')
            before=reader.evidence(region['offsets'],caption);after=reader.evidence(crop,caption)
            row.update(before=before,after=after)
            if lexical_word(after['recognized_text'])!=lexical_word(caption) or not math.isfinite(before['nll']) or not math.isfinite(after['nll']) or after['nll']>before['nll']+.25:
                row['reason']='Word readability did not pass.';continue
            readback=compare_readback(text,reader.read(offsets))
            if (not readback.numbers_match,readback.edit_distance)>(not line.readback.numbers_match,line.readback.edit_distance):
                row['reason']='Full-line wording worsened.';continue
            line.offsets=[(float(x),float(y),int(end)) for x,y,end in offsets];line.readback=readback
            row.update(applied=True,characters=changes,placement=transform)
            if line.t_formation and line.t_formation.get('selected_words'):
                evidence=line.t_formation['selected_words'];word=evidence['words'][index]
                word.update(recognized_text=after['recognized_text'],nll=after['nll'],exact=True,boundary_verified=True)
                focus=[w for w in evidence['words'] if w['target']];context=[w for w in evidence['words'] if not w['target']]
                evidence.update(exact_words=sum(w['exact'] for w in focus),mean_nll=sum(w['nll'] for w in focus)/max(1,len(focus)),context_nll=sum(w['nll'] for w in context)/max(1,len(context)))
    else:audit['reason']='No safe word boundaries.'
    audit['applied']=any(w['applied'] for w in audit['words']);line.r_formation=audit
    line.seconds=original.seconds+time.perf_counter()-started
    if not audit['applied']:return line,baseline[1]
    native=np.asarray(line.offsets,dtype=float);native[:,:2]/=20;native[:,1]*=-1
    return line,native.tolist()

