import json
from pathlib import Path
import numpy as np
from inkform.readable_generator import ReadableGenerator,READABLE_MODEL_DIR
from inkform.schema import GenerationSettings,StyleReference,PageSettings,InkSettings
from inkform.render import layout_document,render_page
from PIL import Image,ImageDraw

OUT=Path('output/pdf/cursive-samples');OUT.mkdir(parents=True,exist_ok=True)
TEXT='My first hero was missing half his body.\nI still want to be a hero. Only now, I stay for\nthe parts of the story I used to skip.'
g=ReadableGenerator();records=[]
page=PageSettings(margin_mm=20,writing_height_mm=7.5,line_spacing_mm=11,naturalness=.25,letter_variation=0)
ink=InkSettings(color='#203f78',width_mm=.3,opacity=.98,roughness=.25)
for label,style in zip('ABC',(6,8,10)):
 ref=StyleReference(text=np.load(READABLE_MODEL_DIR/f'styles/style-{style}-chars.npy',allow_pickle=False).tobytes().decode(),normalized_offsets=np.load(READABLE_MODEL_DIR/f'styles/style-{style}-strokes.npy',allow_pickle=False).tolist(),codec='vasquez-native-v1')
 cached=OUT/f'style-{style}.json'
 if cached.exists():
  from inkform.schema import Document
  d=Document.model_validate_json(cached.read_text())
 else:
  d=g.generate(GenerationSettings(text=TEXT,seed=42,bias=.8,reference=ref,page=page,ink=ink),lambda i,n,t:print(f'{label} / reference {style} / {i+1}/{n}: {t}',flush=True))
  d.backend=f'vasquez-17900-reference{style}'
  cached.write_text(d.model_dump_json())
  (OUT/f'reference-{label}.json').write_text(ref.model_dump_json())
 layout=layout_document(d,page)
 image=render_page(layout.pages[0],page,ink,150)
 px=150/25.4
 crop=image.crop((int(18*px),int(18*px),int(193*px),int((20+11*len(d.lines)+2)*px)))
 crop.save(OUT/f'style-{label}.png')
 records.append({'label':label,'reference':style,'readback':[l.readback.model_dump() for l in d.lines]})
 print(json.dumps(records[-1]),flush=True)
(OUT/'readback.json').write_text(json.dumps(records,indent=2))
