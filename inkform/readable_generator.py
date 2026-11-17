"""A general neural backend; no per-word patches, fixed glyphs, or drawn letters."""
from functools import lru_cache
from pathlib import Path
import hashlib
import math
import time

import numpy as np
import torch

from .generator import GravesGenerator, GenerationError, ROOT
from .recognition import get_recognizer,compare_readback
from .schema import StyleReference,Line
from .stroke_network import StrokeNetwork, ALPHABET

READABLE_MODEL_DIR = ROOT / "models/vasquez"
DEFAULT_STYLE = 2
DISPLAY_NAME = f"Vasquez · writer {DEFAULT_STYLE}"
T_ADAPTER_PATH = READABLE_MODEL_DIR/'t_candidate_adapter.pt'
T_ADAPTER_SHA256 = '10b6eb094a033c5b4ff5f544d0184bcbcfe86a3d95423f5b5e1175a147ba12b5'


class ReadableGenerator(GravesGenerator):
    supports_cursive_selection = True

    def __init__(self, model_dir: Path = READABLE_MODEL_DIR, recognizer_factory=get_recognizer,
                 adapter_path: Path | None = None):
        required = ("model.pt", f"styles/style-{DEFAULT_STYLE}-strokes.npy", f"styles/style-{DEFAULT_STYLE}-chars.npy")
        if not all((model_dir/name).is_file() for name in required):
            raise GenerationError("Readable stroke checkpoint or writer sample missing. Run scripts/download_readable_model.py.")
        torch.set_num_threads(1)
        self.model = StrokeNetwork()
        self.model.import_weights(torch.load(model_dir/"model.pt",map_location="cpu",weights_only=True))
        if adapter_path is not None and adapter_path.is_file():
            if adapter_path==T_ADAPTER_PATH and hashlib.sha256(adapter_path.read_bytes()).hexdigest()!=T_ADAPTER_SHA256:
                raise GenerationError('The t candidate adapter checksum differs from the validated local checkpoint.')
            from .t_adapter import TStrokeAdapter
            self.model.t_adapter=TStrokeAdapter()
            self.model.t_adapter.load_state_dict(torch.load(adapter_path,map_location='cpu',weights_only=True),strict=True)
        self.t_adapter_version=hashlib.sha256(adapter_path.read_bytes()).hexdigest()[:16] if self.model.t_adapter is not None else None
        self.model.eval()
        self.charset = ALPHABET[1:]
        self.backend = f"vasquez-17900-style{DEFAULT_STYLE}"
        self.recognizer_factory = recognizer_factory
        primer = np.load(model_dir/f"styles/style-{DEFAULT_STYLE}-strokes.npy",allow_pickle=False)
        text = np.load(model_dir/f"styles/style-{DEFAULT_STYLE}-chars.npy",allow_pickle=False).tobytes().decode("utf-8")
        self.default_reference = StyleReference(text=text,normalized_offsets=primer.tolist(),codec="vasquez-native-v1")
        from .t_formation import TFormationEvaluator
        self.t_evaluator=TFormationEvaluator(self.model,self._primer)

    @staticmethod
    def _primer(reference):
        if reference is None:
            return None
        values = torch.tensor(reference.normalized_offsets,dtype=torch.float32)
        if reference.codec == "graves-normalized-v1":
            # Import old references explicitly into the new coordinate system.
            mean = torch.tensor([8.217162132263184,.1212363988161087,0.])
            std = torch.tensor([42.34926223754883,37.07347869873047,1.])
            values = values*std+mean
            values[:,1] *= -1
            unit = values[:,:2].norm(dim=1).median().clamp_min(1e-4)
            values[:,:2] /= unit
        return values

    def _trajectory(self,text,bias,seed,reference,form_diversity=0.0,t_clarity=0.0,t_strategy='adapter'):
        prefix = reference.text+" " if reference else ""
        native,complete = self.model.sample(prefix+text,self._primer(reference),seed,bias,
                                          min(8000,max(600,len(text)*65)),form_diversity,
                                          t_clarity=t_clarity,content_start=len(prefix),t_strategy=t_strategy)
        if native is None or not complete:
            return None
        positions = native[:,:2].cumsum(0)
        span = positions.max(0).values-positions.min(0).values
        ratio = float(span[0]/span[1].clamp_min(.1))
        if float(span[1])<.5 or not .04*len(text)<ratio<max(10,len(text)*2):
            return None
        rendered = native.clone()
        rendered[:,:2] *= 20
        rendered[:,1] *= -1
        return rendered.tolist(),native.tolist()

    def select_line(self,text,bias,seed,reference,quality_checks,form_diversity=0.0,t_refinement=False,cursive_r=False,
                    *, prefer_cursive_r=False):
        from .word_clarity import internal_t_words
        baseline=super().select_line(text,bias,seed,reference,quality_checks,form_diversity)
        if t_refinement and quality_checks and internal_t_words(text):
            baseline=self._select_t_words(text,bias,seed,reference,form_diversity,baseline)
        if prefer_cursive_r and quality_checks and reference is not None and 'r' in text:
            from .cursive_selection import select_cursive_words

            started = time.perf_counter()
            line, native = select_cursive_words(self, baseline, reference, bias, form_diversity, seed)
            return line.model_copy(update={"seconds": line.seconds + time.perf_counter() - started}), native
        return baseline

    def _select_t_words(self,text,bias,seed,reference,form_diversity,baseline,cached_evidence=None):
        from .word_clarity import assess_words,word_regions,internal_t_words,place_complete_word,lexical_word,CLARITY_MODE
        started=time.perf_counter();reader=self.recognizer_factory()
        original=baseline[0];current=original.model_copy(deep=True)
        _,labels=self.t_evaluator.character_trace(text,original.offsets,reference)
        regions=word_regions(original.offsets,labels,text)
        evidence=cached_evidence if cached_evidence is not None else assess_words(original.offsets,labels,text,reader)
        audit=dict(method=CLARITY_MODE,generation='closed-word-resampling',
            applied=False,baseline_seed=original.seed,baseline_words=evidence,selected_words=evidence,
            individual_strokes_deformed=False,word_placements=[],proposals=[])
        current.t_refinement=True
        if evidence is None or regions is None:
            audit['retained_reason']='No unambiguous blank word boundaries.'
            current.t_formation=audit;current.seconds+=time.perf_counter()-started
            return current,baseline[1]
        # Work from right to left so replacing a word cannot invalidate the
        # source point indices of a preceding word. Each boundary is a pen lift.
        weak=[w for w in evidence['words'] if w['target'] and (not w['exact'] or w['nll']>.75)]
        selected={w['word_index']:dict(w) for w in evidence['words']};attempts=0
        for word in sorted(weak,key=lambda w:w['word_index'],reverse=True):
            region=regions[word['word_index']];best=None;best_nll=word['nll'];placement=None;best_proof=None
            if not region['ownership_verified']:
                audit['proposals'].append(dict(word=region['text'],accepted=False,reason='Incomplete or foreign character ownership.'))
                selected[word['word_index']]['exact']=False
                continue
            for attempt,adjustment in enumerate((.15,.25,.05,.30)):
                word_seed=(seed+104729*(attempt+1)+15485863*(word['word_index']+1))%(2**31-1)
                candidate_bias=min(1.2,max(.5,bias+adjustment));attempts+=1
                sample=self._trajectory(region['text'],candidate_bias,word_seed,reference,form_diversity)
                entry=dict(word=region['text'],seed=word_seed,accepted=False);audit['proposals'].append(entry)
                if sample is None:entry['reason']='incomplete';continue
                placed=place_complete_word(current.offsets,region,sample[0])
                if placed is None:entry['reason']='unsafe boundary or incompatible whole-word size';continue
                offsets,transform=placed
                # Assess the exact placed pen path, not the smaller raw sample.
                data=np.asarray(offsets);xy=data[:,:2].cumsum(0)
                start=transform['start'];stop=start+transform['new_points']
                crop=np.column_stack((np.diff(np.vstack((np.zeros((1,2)),xy[start:stop])),axis=0),data[start:stop,2])).tolist()
                caption=region['text'].strip('.,;:!?"()[]')
                proof=reader.evidence(crop,caption);entry['word_evidence']=proof
                if lexical_word(proof['recognized_text'])!=lexical_word(caption) or not math.isfinite(proof['nll']) or proof['nll']>=best_nll-.10:
                    entry['reason']='no verified word improvement';continue
                readback=compare_readback(text,reader.read(offsets))
                rank=lambda proof:(not proof.numbers_match,proof.edit_distance)
                if rank(readback)>rank(current.readback):entry['reason']='whole-line wording worsened';continue
                candidate=current.model_copy(deep=True)
                candidate.offsets=[(float(x),float(y),int(end)) for x,y,end in offsets]
                candidate.readback=readback
                best=candidate;best_nll=proof['nll'];best_proof=proof;placement=dict(word=region['text'],word_index=word['word_index'],seed=word_seed,bias=candidate_bias,**transform)
                entry['accepted']=True
            if best is not None:
                current=best
                selected[word['word_index']].update(recognized_text=best_proof['recognized_text'],exact=True,nll=best_nll,boundary_verified=True)
                audit['word_placements'].append(placement)
        rows=[selected[i] for i in sorted(selected)];targets=[w for w in rows if w['target']]
        neighbors=[w for w in rows if not w['target']]
        audit['selected_words']=dict(words=rows,exact_words=sum(w['exact'] for w in targets),target_words=len(targets),
            mean_nll=sum(w['nll'] for w in targets)/max(1,len(targets)),context_nll=sum(w['nll'] for w in neighbors)/max(1,len(neighbors)))
        audit['applied']=bool(audit['word_placements']);current.t_formation=audit
        current.t_adapter_version=None;current.attempts=original.attempts+attempts
        current.seconds=original.seconds+time.perf_counter()-started
        if not audit['applied']:return current,baseline[1]
        native=np.asarray(current.offsets,dtype=float);native[:,:2]/=20;native[:,1]*=-1
        return current,native.tolist()

    def _word_evidence(self,text,offsets,reference,reader):
        from .word_clarity import assess_words
        _,labels=self.t_evaluator.character_trace(text,offsets,reference)
        return assess_words(offsets,labels,text,reader)

    def _select_t_context(self,text,bias,seed,reference,form_diversity,baseline):
        """Select an intact neural take; never deform or splice t or its joins."""
        from .word_clarity import internal_t_words,prefer_complete_take
        if not internal_t_words(text):return baseline
        started=time.perf_counter();reader=self.recognizer_factory()
        original=baseline[0];best=(original.model_copy(deep=True),baseline[1])
        evidence=self._word_evidence(text,original.offsets,reference,reader)
        audit=dict(method='whole-line-word-clarity-v1',applied=False,baseline_seed=original.seed,
            baseline_words=evidence,selected_words=evidence,proposals=[],strokes_modified=False)
        best[0].t_refinement=True;best[0].t_formation=audit
        # Uncertain segmentation never supplies evidence for a replacement.
        if evidence is None:
            audit['retained_reason']='No unambiguous blank word boundaries.'
            best[0].seconds+=time.perf_counter()-started
            return best
        if evidence['exact_words']==evidence['target_words'] and evidence['mean_nll']<.75:
            audit['retained_reason']='Target words already read clearly in isolation.'
            best[0].seconds+=time.perf_counter()-started
            return best
        # Fixed bounded proposals keep the same requested writer and diversity.
        # All coordinates and pen lifts of a winner are taken verbatim from one
        # complete autoregressive generation, including surrounding words.
        attempts=0;winner=evidence
        for offset,adjustment in ((104729,.15),(628374,.20),(314187,.10),(209458,.30)):
            candidate_seed=(seed+offset)%(2**31-1)
            candidate_bias=min(1.2,max(.5,bias+adjustment))
            result=self._trajectory(text,candidate_bias,candidate_seed,reference,form_diversity)
            attempts+=1
            entry=dict(seed=candidate_seed,bias=candidate_bias,accepted=False)
            audit['proposals'].append(entry)
            if result is None:
                entry['reason']='incomplete';continue
            offsets,native=result
            readback=compare_readback(text,reader.read(offsets))
            if readback.status!='match':
                entry['reason']='whole-line readback mismatch';continue
            candidate_words=self._word_evidence(text,offsets,reference,reader)
            entry['words']=candidate_words
            if not prefer_complete_take(evidence,candidate_words):
                entry['reason']='word or surrounding-context evidence did not pass';continue
            if candidate_words['mean_nll']>=winner['mean_nll']:
                entry['reason']='weaker than retained take';continue
            candidate=Line(text=text,offsets=offsets,seed=candidate_seed,seconds=0,attempts=0,
                primed=reference is not None,attention_complete=True,sampling_bias=candidate_bias,
                sampling_form_diversity=form_diversity,t_refinement=True,readback=readback)
            entry['accepted']=True;winner=candidate_words;best=(candidate,native)
        audit['applied']=best[0].seed!=original.seed
        audit['selected_words']=winner
        audit['selected_seed']=best[0].seed
        if not audit['applied']:audit['retained_reason']='No complete take passed all word and context gates.'
        best[0].t_formation=audit
        best[0].attempts=original.attempts+attempts
        best[0].seconds=original.seconds+time.perf_counter()-started
        return best

    def _refine_t_line(self,text,bias,seed,reference,form_diversity,baseline):
        from .t_stem import refine_internal_t_stems
        import numpy as np
        started=time.perf_counter();original=baseline[0]
        if not any(c=='t' and i and text[i-1].isalpha() for i,c in enumerate(text)):return baseline
        native,labels=self.t_evaluator.character_trace(text,original.offsets,reference)
        best=(original.model_copy(deep=True),baseline[1]);trials=0
        best[0].t_refinement=True
        best[0].t_formation=dict(method='attention-guided-stem-refinement',adjustments=[],applied=False)
        rank=lambda line:(not line.readback.numbers_match,line.readback.edit_distance)
        reader=self.recognizer_factory()
        for strength in (.85,.55):
            corrected,audit=refine_internal_t_stems(native,labels,text,strength)
            if not audit:break
            trials+=1
            rendered=np.asarray(corrected);rendered[:,:2]*=20;rendered[:,1]*=-1
            candidate=original.model_copy(deep=True)
            candidate.offsets=[(float(x),float(y),int(end)) for x,y,end in rendered];candidate.t_refinement=True
            candidate.t_adapter_version=None
            candidate.readback=compare_readback(text,reader.read(candidate.offsets))
            if rank(candidate)<=rank(original):
                candidate.t_formation=dict(method='attention-guided-stem-refinement',strength=strength,adjustments=audit,applied=True)
                best=(candidate,corrected)
                break
        best[0].attempts=original.attempts+trials
        best[0].seconds=original.seconds+time.perf_counter()-started
        return best

    def _experimental_refine_t_line(self,text,bias,seed,reference,form_diversity,baseline):
        from .t_formation import prefer_t_candidate
        started=time.perf_counter();best=(baseline[0].model_copy(deep=True),baseline[1])
        shape=self.t_evaluator.score(text,best[0].offsets,reference)
        if shape is None or not math.isfinite(shape['margin']):return baseline
        best[0].t_refinement=True;best[0].t_formation=shape
        reader=self.recognizer_factory();attempts=0
        modes=('base','adapter') if self.model.t_adapter is not None else ('base','precision')
        for index,mode in enumerate(modes):
            candidate_seed=(seed+104729+index*7919)%(2**31-1)
            candidate_bias=max(bias,1.0)
            result=self._trajectory(text,candidate_bias,candidate_seed,reference,form_diversity,
                t_clarity=1. if mode=='adapter' else .4 if mode=='precision' else 0.,
                t_strategy='adapter' if mode=='adapter' else 'precision')
            attempts+=1
            if result is None:continue
            offsets,native=result
            candidate=Line(text=text,offsets=offsets,seed=candidate_seed,seconds=0,attempts=0,
                primed=reference is not None,attention_complete=True,sampling_bias=candidate_bias,
                sampling_form_diversity=form_diversity,t_refinement=True,
                t_adapter_version=self.t_adapter_version if mode=='adapter' else None,
                readback=compare_readback(text,reader.read(offsets)))
            rank=lambda line:(not line.readback.numbers_match,line.readback.edit_distance)
            if rank(candidate)>rank(best[0]):continue
            candidate.t_formation=self.t_evaluator.score(text,offsets,reference)
            if prefer_t_candidate(best[0],candidate,shape,candidate.t_formation):
                best=(candidate,native);shape=candidate.t_formation
        best[0].attempts=baseline[0].attempts+attempts
        best[0].seconds=baseline[0].seconds+time.perf_counter()-started
        return best


@lru_cache(maxsize=1)
def get_readable_generator():
    return ReadableGenerator()
