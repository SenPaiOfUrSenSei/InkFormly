import json,torch
from pathlib import Path
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.recognition import recognition_image
import importlib
reference=importlib.import_module('tmp.r-style.conditioning').reference
out=Path('tmp/r-style/aligned');out.mkdir(exist_ok=True);g=ReadableGenerator();ref=reference(g,5,50);real=len(g.default_reference.normalized_offsets);last=len(ref.normalized_offsets);slot=len(ref.text)-1;step=g.model.step;counter=[0]
def aligned(pen,state,text):
 s=step(pen,state,text);i=counter[0];counter[0]+=1
 if real<=i<last:
  center=slot-.25+(i-real)/max(1,last-real-1)
  s.position=s.position+(center-float(s.attention.argmax()))
  a=torch.zeros_like(s.attention);lo=int(center);fraction=center-lo
  a[:,lo]=1-fraction;a[:,lo+1]=fraction;a*=s.attention.sum()
  s.attention=a;s.window=(a.unsqueeze(-1)*text).sum(1)
 return s
g.model.step=aligned;rows=[]
for word in ['around','hero','river','friend']:
 for seed in [42,509]:
  counter[0]=0;sample=g._trajectory(word,.95,seed,ref,.5)
  if sample is None:print('failed',word,seed,flush=True);continue
  r=dict(text=word,seed=seed,offsets=sample[0],readback=g.recognizer_factory().read(sample[0]));rows.append(r);(out/f'{word}-{seed}.json').write_text(json.dumps(r));print(word,seed,r['readback'],flush=True)
s=Image.new('RGB',(1400,150*len(rows)),'white');d=ImageDraw.Draw(s)
for i,r in enumerate(rows):
 d.text((10,i*150+3),f"{r['text']} / {r['seed']} / {r['readback']}",fill='black');im=recognition_image(r['offsets']);im.thumbnail((1300,100));s.paste(im,(10,i*150+30))
s.save(out/'gallery.png')
