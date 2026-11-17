import json
from pathlib import Path
from PIL import Image,ImageDraw
from inkform.generator import get_generator,INFERENCE_LOCK
from inkform.recognition import recognition_image
from inkform.schema import Document
out=Path('tmp/pdfs/cursive-A-review');out.mkdir(parents=True,exist_ok=True)
p=Document.model_validate_json(Path('output/pdf/college-essay-cursive-A/paragraph-02.json').read_text());g=get_generator();records=[];rows=[]
for text in ('man beneath it,','I turned back to my','asked what I wanted to be, I shouted,'):
 for take in range(2):
  with INFERENCE_LOCK:line,_=g.select_line(text,.95,817+300007+take*104729,p.reference,True)
  records.append({'text':text,'take':take,'line':line.model_dump()})
  row=Image.new('RGB',(1700,160),'white');ImageDraw.Draw(row).text((10,4),f'{text} | take {take} | {line.readback.status} | {line.readback.recognized_text}',fill='black');im=recognition_image(line.offsets);im.thumbnail((1650,115));row.paste(im,(10,35));rows.append(row)
  (out/'phrase-takes.json').write_text(json.dumps(records));print(text,take,line.readback.model_dump(),flush=True)
 sheet=Image.new('RGB',(1700,320),'white')
 for i,row in enumerate(rows[-2:]):sheet.paste(row,(0,i*160))
 sheet.save(out/f'phrase-{len(records)//2}.png')
