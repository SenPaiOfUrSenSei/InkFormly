"""Build visual proof sheets and content/geometry diagnostics for the generated essay."""
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
from inkform.schema import Document,InkSettings,PageSettings
from inkform.recognition import recognition_image
from inkform.render import layout_document,strokes_from_offsets

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/pdf/college-essay'
QA=ROOT/'tmp/pdfs/college-essay-review'
QA.mkdir(parents=True,exist_ok=True)


def main():
 document=Document.model_validate_json((OUT/'document.json').read_text())
 settings=json.loads((OUT/'settings.json').read_text())
 layout=layout_document(document,PageSettings.model_validate(settings['page']))
 items=[];rows=[]
 for index,line in enumerate(document.lines):
  if not line.text:continue
  strokes=strokes_from_offsets(line.offsets);points=np.concatenate(strokes);span=np.ptp(points,axis=0)
  items.append({'line':index+1,'text':line.text,'readback':line.readback.model_dump(),
                'height':float(span[1]),'width':float(span[0]),'seed':line.seed})
  row=Image.new('RGB',(1800,175),'white')
  ImageDraw.Draw(row).text((12,6),f'{index+1}: {line.text} | {line.readback.status} | OCR: {line.readback.recognized_text}',fill='black')
  im=recognition_image(line.offsets);im.thumbnail((1750,125));row.paste(im,(12,38));rows.append(row)
  if line.readback.status!='match':row.save(QA/f'line-{index+1:03d}.png')
 for start in range(0,len(rows),10):
  batch=rows[start:start+10];sheet=Image.new('RGB',(1800,175*len(batch)),'white')
  for y,row in enumerate(batch):sheet.paste(row,(0,y*175))
  sheet.save(QA/f'sheet-{start//10+1:02d}.png')
 report={'pages':len(layout.pages),'scale':layout.scale,'fitted_to_width':layout.fitted_to_width,
         'median_height':float(np.median([item['height'] for item in items])),
         'tallest':sorted(items,key=lambda x:x['height'],reverse=True)[:5],
         'review':[item for item in items if item['readback']['status']!='match']}
 (QA/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
