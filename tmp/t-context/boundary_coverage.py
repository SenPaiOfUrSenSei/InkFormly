import json,numpy as np
from pathlib import Path
from inkform.readable_generator import ReadableGenerator
from inkform.word_clarity import word_regions
from inkform.schema import Document
D=Document.model_validate_json(Path('output/pdf/college-essay-restored-context/document.json').read_text());g=ReadableGenerator()
for i in [1,8,11,30,34,43,56,67,79,92,93,98,106,115]:
 l=D.lines[i-1];_,labels=g.t_evaluator.character_trace(l.text,l.offsets,D.reference);labels=np.array(labels);rs=word_regions(l.offsets,labels,l.text)
 data=json.loads(Path(f'output/pdf/college-essay-t-context-words/lines/line-{i:03d}.json').read_text());placements=data['line']['t_formation'].get('word_placements',[])
 for p in placements:
  r=rs[p['word_index']];ids=np.array(r['source_indices']);last=r['stop']-1;all_ids=np.flatnonzero((labels>=r['start'])&(labels<r['stop']));last_ids=np.flatnonzero(labels==last)
  print(i,r['text'],'wordcov',round(len(np.intersect1d(ids,all_ids))/max(1,len(all_ids)),3),'lastcov',round(len(np.intersect1d(ids,last_ids))/max(1,len(last_ids)),3),'lastn',len(last_ids),flush=True)
