from pathlib import Path
import json,numpy as np
from inkform.readable_generator import ReadableGenerator
from inkform.schema import Document
from inkform.word_clarity import word_regions
D=Document.model_validate_json(Path('output/pdf/college-essay-restored-context/document.json').read_text());g=ReadableGenerator();records=[]
for p in Path('output/pdf/college-essay-t-context-words/lines').glob('*.json'):
 row=json.loads(p.read_text());l=D.lines[row['index']];a=row['line']['t_formation'];ps=a.get('word_placements',[])
 if not ps:continue
 cache=json.loads((Path('output/pdf/college-essay-t-word-clarity/lines')/p.name).read_text());base=cache['line'] if cache['changed'] else l.model_dump()
 _,labels=g.t_evaluator.character_trace(l.text,base['offsets'],D.reference);regions=word_regions(base['offsets'],labels,l.text)
 checks=[(w['word_index'],regions[w['word_index']]['ownership_verified']) for w in ps]
 records.append(dict(index=row['index'],checks=checks));print(row['index']+1,[(regions[j]['text'],ok) for j,ok in checks],flush=True)
Path('tmp/t-context/ownership-audit.json').write_text(json.dumps(records,indent=2))
