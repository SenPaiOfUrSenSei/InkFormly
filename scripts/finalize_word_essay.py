"""Replay accepted word samples with the final semantic-boundary safeguards."""
import json
from pathlib import Path

import numpy as np

from inkform.readable_generator import ReadableGenerator
from inkform.recognition import compare_readback
from inkform.schema import Document,Line
from inkform.word_clarity import word_regions,place_complete_word,lexical_word
from scripts.refine_essay_word_blocks import SOURCE,OUT,CACHE,compose,save


def main():
    doc=Document.model_validate_json((SOURCE/'document.json').read_text())
    g=ReadableGenerator();reader=g.recognizer_factory()
    for path in sorted((OUT/'lines').glob('*.json')):
        row=json.loads(path.read_text());old=Line.model_validate(row['line']);audit=old.t_formation or {}
        if audit.get('generation')!='closed-word-resampling':continue
        index=row['index'];prior=json.loads((CACHE/path.name).read_text());base=Line.model_validate(prior['line'])
        current=base.model_copy(deep=True)
        _,labels=g.t_evaluator.character_trace(base.text,base.offsets,doc.reference)
        regions=word_regions(base.offsets,labels,base.text)
        proof=json.loads(json.dumps(audit['baseline_words']))
        kept=[];rejected=[]
        if proof and regions:
            for word in proof['words']:
                region=regions[word['word_index']]
                word['boundary_verified']=region['ownership_verified']
                word['exact']=region['ownership_verified'] and lexical_word(word['recognized_text'])==lexical_word(word['word'])
            for proposal in audit.get('word_placements',[]):
                region=regions[proposal['word_index']]
                if not region['ownership_verified']:
                    rejected.append(dict(word=region['text'],reason='Incomplete or foreign character ownership.'));continue
                sample=g._trajectory(region['text'],proposal['bias'],proposal['seed'],doc.reference,doc.form_diversity)
                placed=place_complete_word(current.offsets,region,sample[0]) if sample else None
                if placed is None:
                    rejected.append(dict(word=region['text'],reason='Unsafe complete-word fit.'));continue
                offsets,transform=placed
                data=np.asarray(offsets);xy=data[:,:2].cumsum(0);start=transform['start'];stop=start+transform['new_points']
                crop=np.column_stack((np.diff(np.vstack((np.zeros((1,2)),xy[start:stop])),axis=0),data[start:stop,2])).tolist()
                observed=reader.evidence(crop,region['text'].strip('.,;:!?"()[]'))
                word=proof['words'][proposal['word_index']]
                if lexical_word(observed['recognized_text'])!=lexical_word(region['text']) or observed['nll']>=word['nll']-.10:
                    rejected.append(dict(word=region['text'],reason='Final isolated-word check failed.'));continue
                readback=compare_readback(base.text,reader.read(offsets))
                rank=lambda r:(not r.numbers_match,r.edit_distance)
                if rank(readback)>rank(current.readback):
                    rejected.append(dict(word=region['text'],reason='Final whole-line check worsened.'));continue
                current.offsets=[(float(x),float(y),int(end)) for x,y,end in offsets]
                current.readback=readback
                word.update(recognized_text=observed['recognized_text'],nll=observed['nll'],exact=True)
                kept.append({**proposal,**transform,'ownership_verified':True})
            targets=[w for w in proof['words'] if w['target']]
            proof.update(exact_words=sum(w['exact'] for w in targets),mean_nll=sum(w['nll'] for w in targets)/max(1,len(targets)))
        audit.update(applied=bool(kept),word_placements=kept,selected_words=proof,final_boundary_rejections=rejected)
        # Keep the already verified intact-line alternative if no word was
        # replaced; otherwise record the replayed, ownership-safe word edits.
        if kept or not prior['changed']:
            current.t_formation=audit;current.t_refinement=True
        current.seconds=old.seconds;current.attempts=old.attempts
        row.update(line=current.model_dump(),changed=current.offsets!=doc.lines[index].offsets)
        save(path,row)
        if kept or rejected:print(index+1,'kept',len(kept),'rejected',len(rejected),flush=True)
    compose()


if __name__=='__main__':main()
