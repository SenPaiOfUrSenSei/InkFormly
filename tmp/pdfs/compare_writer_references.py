import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator,READABLE_MODEL_DIR
from inkform.schema import StyleReference
from inkform.recognition import recognition_image

out=Path('tmp/pdfs/angular-review');out.mkdir(parents=True,exist_ok=True)
g=ReadableGenerator();records=[]
texts=['My first hero was missing half his body.','Sometimes, even now, I wish I could forget.']
for style in (3,0,10):
 ref=StyleReference(text=np.load(READABLE_MODEL_DIR/f'styles/style-{style}-chars.npy',allow_pickle=False).tobytes().decode(),normalized_offsets=np.load(READABLE_MODEL_DIR/f'styles/style-{style}-strokes.npy',allow_pickle=False).tolist(),codec='vasquez-native-v1')
 rows=[]
 for i,text in enumerate(texts):
  result=g._trajectory(text,.85,42+i*1009,ref)
  row=Image.new('RGB',(1700,180),'white');ImageDraw.Draw(row).text((12,6),f'Reference {style}: {text}',fill='black')
  if result:
   offsets,_=result;im=recognition_image(offsets);im.thumbnail((1650,130));row.paste(im,(12,40));records.append({'style':style,'text':text,'offsets':offsets})
  rows.append(row)
 sheet=Image.new('RGB',(1700,360),'white')
 for i,row in enumerate(rows):sheet.paste(row,(0,i*180))
 sheet.save(out/f'reference-{style}.png');print(style,flush=True)
(out/'samples.json').write_text(json.dumps(records))
