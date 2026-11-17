import json,time
from pathlib import Path
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.recognition import recognition_image
out=Path('artifacts/evaluation/cursive-r-context');out.mkdir(exist_ok=True);g=ReadableGenerator();ref=g.default_reference
for index,text in enumerate(['A friend walked around the river.','The hero returned with courage.']):
 p=out/f'line-{index+1}.json'
 if p.exists():r=json.loads(p.read_text())
 else:
  base=g.select_line(text,.85,517+index*1009,ref,True,.5,False,False)
  line,_=g._select_r_words(text,517+index*1009,ref,base)
  assert (not line.readback.numbers_match,line.readback.edit_distance)<=(not base[0].readback.numbers_match,base[0].readback.edit_distance)
  r=dict(text=text,before=base[0].model_dump(),after=line.model_dump(),changed=line.offsets!=base[0].offsets);p.write_text(json.dumps(r))
 print(text,r['changed'],r['after']['readback']['recognized_text'],flush=True)
 s=Image.new('RGB',(1700,340),'white');d=ImageDraw.Draw(s)
 for n,key in enumerate(('before','after')):
  l=r[key];d.text((10,n*170+4),f'{key}: {text}',fill='black');im=recognition_image(l['offsets']);im.thumbnail((1650,125));s.paste(im,(10,n*170+30))
 s.save(out/f'comparison-{index+1}.png')
