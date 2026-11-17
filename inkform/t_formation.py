"""Model-based contrast for internal lowercase t, without fixed glyph templates."""
from collections import OrderedDict

import torch
from torch.nn import functional as F

from .stroke_network import ALPHABET
from .t_adapter import motion_nll


class TFormationEvaluator:
    def __init__(self,model,primer_factory):
        self.model=model
        self.primer_factory=primer_factory
        self.cache=OrderedDict()

    @torch.inference_mode()
    def _primed(self,text,reference):
        # A reference object is shared by a generation; include its serialized
        # values so a later imported reference cannot reuse an unrelated state.
        key=(text,reference.model_dump_json() if reference else None)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        context=F.one_hot(torch.tensor([[ALPHABET.index(c) for c in text]+[0]]),len(ALPHABET)).float()
        state=self.model.initial_state()
        primer=self.primer_factory(reference)
        if primer is not None:
            for point in primer:state=self.model.step(point.reshape(1,3),state,context)
        self.cache[key]=(context,state)
        if len(self.cache)>20:self.cache.popitem(last=False)
        return context,state

    @torch.inference_mode()
    def character_trace(self,text,offsets,reference):
        import numpy as np
        prefix=reference.text+' ' if reference else ''
        context,state=self._primed(prefix+text,reference)
        values=np.asarray(offsets,dtype=np.float64).copy()
        values[:,:2]/=20;values[:,1]*=-1
        native=torch.tensor(values,dtype=torch.float32)
        pen=torch.tensor([[0.,0.,1.]]);labels=[]
        for point in native:
            state=self.model.step(pen,state,context)
            labels.append(int(state.attention.argmax())-len(prefix))
            pen=point.reshape(1,3)
        return values.tolist(),labels

    @torch.inference_mode()
    def score(self,text,offsets,reference):
        positions=[i for i,c in enumerate(text) if c=='t' and i and text[i-1].isalpha()]
        if not positions:return None
        prefix=reference.text+' ' if reference else ''
        native=torch.tensor(offsets,dtype=torch.float32)
        native[:,:2]/=20;native[:,1]*=-1
        mask=torch.zeros(len(prefix+text)+1)
        for i in positions:
            p=len(prefix)+i;mask[p]=1
            if p+1<len(mask)-1:mask[p+1]=max(float(mask[p+1]),.25)
        records=[];focus=None
        for replacement in (None,'l','f'):
            variant=text if replacement is None else ''.join(replacement if i in positions else c for i,c in enumerate(text))
            context,state=self._primed(prefix+variant,reference)
            pen=torch.tensor([[0.,0.,1.]])
            raw=[];weights=[]
            for target in native:
                state=self.model.step(pen,state,context)
                raw.append(self.model.density(state.hidden[2])[0])
                if replacement is None:
                    attention=state.attention.reshape(-1)
                    weights.append((attention*mask).sum()/attention.sum().clamp_min(1e-8))
                pen=target.reshape(1,3)
            if replacement is None:focus=torch.stack(weights)
            nll=motion_nll(torch.stack(raw),native)
            records.append(float((nll*focus).sum()/focus.sum().clamp_min(1e-8)))
        return dict(targets=len(positions),effective_steps=float(focus.sum()),
            intended_nll=records[0],confusion_nll=dict(zip(('l','f'),records[1:])),
            margin=min(value-records[0] for value in records[1:]))


def prefer_t_candidate(original,candidate,original_shape,candidate_shape):
    """Both word readback and t evidence must pass; baseline is always retained."""
    import math
    if not original_shape or not candidate_shape:return False
    if candidate.text!=original.text:return False
    if any(line.readback.edit_distance is None or line.readback.numbers_match is None for line in (original,candidate)):return False
    if not all(math.isfinite(s['margin']) for s in (original_shape,candidate_shape)):return False
    if candidate_shape['effective_steps']<3*candidate_shape['targets']:return False
    rank=lambda line:(not line.readback.numbers_match,line.readback.edit_distance)
    if rank(candidate)>rank(original):return False
    return candidate_shape['margin']>max(0.,original_shape['margin']+.025)
