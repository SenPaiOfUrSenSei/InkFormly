import json
from pathlib import Path
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.recognition import recognition_image
out=Path('tmp/r-style');g=ReadableGenerator();reader=g.recognizer_factory();rows=[]
for word in ['river','around','friend','hero']:
 for seed in [42,137,509,817]:
  p=out/f'{word}-{seed}.json'
  if p.exists():r=json.loads(p.read_text())
  else:
   sample=g._trajectory(word,.95,seed,g.default_reference,.5)
   if sample is None:continue
   offsets,_=sample;_,labels=g.t_evaluator.character_trace(word,offsets,g.default_reference)
   r=dict(text=word,seed=seed,offsets=offsets,labels=labels,readback=reader.read(offsets));p.write_text(json.dumps(r));print(word,seed,r['readback'],flush=True)
  rows.append(r)
sheet=Image.new('RGB',(1600,160*len(rows)),'white');draw=ImageDraw.Draw(sheet)
for i,r in enumerate(rows):
 draw.text((10,i*160+4),f"{r['text']} / {r['seed']} / {r['readback']}",fill='black')
 im=recognition_image(r['offsets']);im.thumbnail((1500,110));sheet.paste(im,(10,i*160+35))
sheet.save(out/'word-gallery.png')
