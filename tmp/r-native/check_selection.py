from pathlib import Path
import json,time
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.recognition import recognition_image
from inkform.schema import GenerationSettings

g=ReadableGenerator()
text='A friend walked around the river.\nThe hero returned with courage.\nI rarely carry a paper map.'
base=GenerationSettings(text=text,seed=517)
before=g.generate(base, lambda i,n,s:print('before',s,flush=True))
after=g.generate(base.model_copy(update={'prefer_cursive_r':True}), lambda i,n,s:print('after',s,flush=True))
output=Path('tmp/r-native/selection')
output.mkdir(exist_ok=True)
image=Image.new('RGB',(1220,224*len(before.lines)),'white')
draw=ImageDraw.Draw(image)
for i,(a,b) in enumerate(zip(before.lines,after.lines)):
   print(a.text, b.readback.status, [(x['word'],x['before'],x['after']) for x in (b.r_selection or {}).get('words',[])], round(b.seconds,2),flush=True)
   for j,(label,line) in enumerate((('Original',a),('Cursive r preference',b))):
      y=224*i+112*j
      draw.text((14,y+6),label,fill='black')
      thumb=recognition_image(line.offsets);thumb.thumbnail((1170,86));image.paste(thumb,(14,y+23))
(image).save(output/'comparison.png')
(output/'before.json').write_text(before.model_dump_json())
(output/'after.json').write_text(after.model_dump_json())
