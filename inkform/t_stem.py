"""Bounded refinement of an attention-aligned, overly open lowercase-t stem.

No glyph is inserted: the generated downstroke is the reference for its own
upstroke. Existing crossbars, joins, endpoints, y coordinates and pen lifts
remain in place. Ambiguous alignment or missing crossing evidence is skipped.
"""
import numpy as np


def _crossing_exists(xy,ends,start,stop,stem_y,stem_x,low,high):
    height=high-low
    for width in (3,5,9):
        for i in range(max(0,start-20),min(len(xy)-width,stop+200)):
            segment=xy[i:i+width+1]
            if np.any(ends[i:i+width]>.5):continue
            dx,dy=segment[-1]-segment[0]
            if abs(dx)<.3*height or abs(dy)>.4*abs(dx):continue
            y=float(segment[:,1].mean())
            if not low+.2*height<y<high-.05*height:continue
            if np.ptp(segment[:,1])>.25*height:continue
            x=float(np.interp(y,stem_y,stem_x))
            if segment[:,0].min()<x-.05*height and segment[:,0].max()>x+.05*height:return True
    return False


def refine_internal_t_stems(native,character_indices,text,strength=.85):
    values=np.asarray(native,dtype=np.float64)
    labels=np.asarray(character_indices,dtype=int)
    if values.ndim!=2 or values.shape[1]!=3 or len(labels)!=len(values):
        raise ValueError('Character trace must align with the pen trajectory.')
    if not np.isfinite(values).all() or not 0<=strength<=1:
        raise ValueError('Invalid t stem refinement input.')
    original=np.cumsum(values[:,:2],axis=0);points=original.copy();audit=[]
    for position,character in enumerate(text):
        if character!='t' or position==0 or not text[position-1].isalpha():continue
        indices=np.flatnonzero(labels==position)
        if len(indices)<12:continue
        blocks=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
        block=max(blocks,key=len)
        if len(block)<12 or len(block)<.7*len(indices):continue
        y=original[block,1]
        smooth=np.convolve(np.pad(y,(1,1),mode='edge'),[.25,.5,.25],mode='valid')
        peak=int(smooth.argmax());height=float(np.ptp(smooth))
        if height<=.5 or peak<5 or peak>len(block)-5:continue
        tolerance=.018*height;up=peak;down=peak
        while up>0 and smooth[up]-smooth[up-1]>=-tolerance:up-=1
        while down+1<len(block) and smooth[down+1]-smooth[down]<=tolerance:down+=1
        if peak-up<5 or down-peak<4:continue
        stem=block[up:down+1]
        if np.any(values[stem[:-1],2]>.5):continue
        descending=original[block[peak:down+1]]
        order=np.argsort(descending[:,1]);sy=descending[order,1];sx=descending[order,0]
        keep=np.r_[True,np.diff(sy)>1e-8];sy,sx=sy[keep],sx[keep]
        if len(sy)<4:continue
        low=max(float(original[block[up],1]),float(sy[0]));high=float(sy[-1])
        if high-low<.4*height:continue
        if not _crossing_exists(original,values[:,2],int(stem[0]),int(stem[-1]),sy,sx,low,high):continue
        rising=block[up:peak+1];u=(original[rising,1]-low)/max(high-low,1e-8)
        target=np.interp(original[rising,1],sy,sx)
        gap=target-original[rising,0]
        core=(u>.15)&(u<.85)
        if core.sum()<3:continue
        median_gap=float(np.median(gap[core]))
        if not .08*height<median_gap<.85*height:continue
        if np.mean(gap[core]>0)<.8:continue
        envelope=np.sin(np.pi*np.clip(u,0,1))
        shift=strength*envelope*np.maximum(gap,0)
        shift[(u<=0)|(u>=1)]=0
        shift=np.minimum(shift,.55*height)
        if float(shift.max())<.04*height:continue
        points[rising,0]+=shift
        audit.append(dict(character_index=position,upstroke_start=int(rising[0]),
            stem_peak=int(block[peak]),downstroke_end=int(block[down]),
            changed_points=[int(i) for i,delta in zip(rising,shift) if delta>1e-9],
            original_median_loop_width=median_gap,maximum_shift=float(shift.max()),
            existing_crossbar=True))
    result=values.copy()
    if audit:result[:,:2]=np.diff(np.vstack((np.zeros((1,2)),points)),axis=0)
    return result.tolist(),audit
