import json
import sys
from pathlib import Path
from PIL import Image,ImageDraw
from inkform.generator import get_generator,INFERENCE_LOCK
from inkform.schema import Document
from inkform.recognition import recognition_image

out=Path('tmp/pdfs/college-essay-review');d=Document.model_validate_json(Path('output/pdf/college-essay/document.json').read_text());g=get_generator();records=json.loads((out/'new-takes.json').read_text()) if (out/'new-takes.json').exists() else []
for number in tuple(map(int,sys.argv[1:])) or (74,96):
 original=d.lines[number-1];rows=[]
 for take in range(3):
  seed=(original.seed+104729*(take+1))%(2**31-1)
  with INFERENCE_LOCK:
   line,_=g.select_line(original.text,1.,seed,d.reference,True)
  records.append({'number':number,'take':take,'line':line.model_dump()})
  row=Image.new('RGB',(1800,160),'white');ImageDraw.Draw(row).text((10,5),f'Line {number} / take {take} / {line.readback.status} / {line.readback.recognized_text}',fill='black');im=recognition_image(line.offsets);im.thumbnail((1750,110));row.paste(im,(10,35));rows.append(row)
  print(number,take,line.readback.model_dump(),flush=True)
  (out/'new-takes.json').write_text(json.dumps(records))
 sheet=Image.new('RGB',(1800,160*len(rows)),'white')
 for i,row in enumerate(rows):sheet.paste(row,(0,i*160))
 sheet.save(out/f'new-takes-{number}.png')
