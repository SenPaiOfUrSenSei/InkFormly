from pathlib import Path
import json,time
from inkform.readable_generator import ReadableGenerator
from inkform.word_clarity import assess_words,prefer_complete_take
from PIL import Image,ImageDraw
from inkform.recognition import recognition_image
out=Path('tmp/t-context/evidence');out.mkdir(exist_ok=True);g=ReadableGenerator();r=g.recognizer_factory();rows=[]
for p in sorted(Path('tmp/t-context/probe-509-My').glob('[0-9].json')):
 d=json.loads(p.read_text());start=time.monotonic();_,labels=g.t_evaluator.character_trace(d['text'],d['offsets'],g.default_reference)
 e=assess_words(d['offsets'],labels,d['text'],r);row=dict(index=d['index'],evidence=e,line_readback=d['readback'],seconds=time.monotonic()-start)
 rows.append(row);(out/'notes.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
print('prefer',[(x['index'],prefer_complete_take(rows[0]['evidence'],x['evidence'])) for x in rows],flush=True)
