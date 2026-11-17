"""Reproduce a rejected reference-r experiment; not a production quality test."""
import json
from pathlib import Path
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.r_style import rejected_reference_r_probe
from inkform.recognition import recognition_image,compare_readback
from inkform.schema import Line

def main():
 out=Path('artifacts/evaluation/cursive-r');out.mkdir(parents=True,exist_ok=True)
 g=ReadableGenerator();records=[]
 for text,seed in [(word,seed) for word in ('around','friend','hero','river') for seed in (42,137,509,817)]:
  target=out/f'{text}-{seed}.json'
  if target.exists():record=json.loads(target.read_text())
  else:
   sample=g._trajectory(text,.95,seed,g.default_reference,.5)
   if sample is None:
    print(text,seed,'incomplete baseline',flush=True);continue
   source=dict(offsets=sample[0],readback=g.recognizer_factory().read(sample[0]))
   before=Line(text=text,offsets=source['offsets'],seed=seed,seconds=0,attempts=1,primed=True,attention_complete=True,readback=compare_readback(text,source['readback']))
   line,_=rejected_reference_r_probe(g,text,seed,g.default_reference,(before,None))
   record=dict(text=text,seed=seed,before=before.model_dump(),after=line.model_dump(),changed=line.offsets!=before.offsets);target.write_text(json.dumps(record))
  records.append(record);print(text,seed,record['changed'],record['after']['readback']['recognized_text'],flush=True)
  s=Image.new('RGB',(1500,300),'white');d=ImageDraw.Draw(s)
  for i,key in enumerate(('before','after')):
   r=record[key];d.text((10,i*150+3),f"{key}: {text} / {r['readback']['recognized_text']}",fill='black');im=recognition_image(r['offsets']);im.thumbnail((1450,100));s.paste(im,(10,i*150+32))
  s.save(out/f'{text}-{seed}.png')
 summary=dict(status='rejected by user: words with cursive r not understandable',production_enabled=False,total=len(records),changed=sum(r['changed'] for r in records),limitations='A reference-shaped geometric alternate, not trained weights. Inconclusive ownership, word reading or full-line reading retains original pen paths.')
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)

if __name__=='__main__':main()
