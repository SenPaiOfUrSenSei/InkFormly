import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator,READABLE_MODEL_DIR
from inkform.schema import StyleReference
from inkform.recognition import recognition_image
out=Path('tmp/pdfs/cursive-style-search');out.mkdir(parents=True,exist_ok=True);g=ReadableGenerator();records=[]
text='I still want to be a hero. Only now, I stay.'
for style in (1,2,4,5,7,9,11,12):
 cp=READABLE_MODEL_DIR/f'styles/style-{style}-chars.npy';sp=READABLE_MODEL_DIR/f'styles/style-{style}-strokes.npy'
 if not cp.exists() or not sp.exists():continue
 raw=np.load(sp,allow_pickle=False);phrase=np.load(cp,allow_pickle=False).tobytes().decode()
 try:ref=StyleReference(text=phrase,normalized_offsets=raw.tolist(),codec='vasquez-native-v1')
 except ValueError as e:print(style,str(e),flush=True);continue
 rows=[]
 # Compare the actual reference's shape with this model's continuation.
 native=raw.copy();native[:,1]*=-1
 for label,offsets in [('actual reference: '+phrase,native.tolist())]:
  im=recognition_image(offsets);im.thumbnail((1600,110));row=Image.new('RGB',(1700,150),'white');ImageDraw.Draw(row).text((12,5),f'{style}: {label}',fill='black');row.paste(im,(12,32));rows.append(row)
 for bias in (.6,.9):
  result=g._trajectory(text,bias,817,ref)
  if not result:continue
  offsets,_=result;records.append({'style':style,'text':text,'bias':bias,'offsets':offsets})
  im=recognition_image(offsets);im.thumbnail((1600,110));row=Image.new('RGB',(1700,150),'white');ImageDraw.Draw(row).text((12,5),f'{style} generated / bias {bias}',fill='black');row.paste(im,(12,32));rows.append(row)
 sheet=Image.new('RGB',(1700,150*len(rows)),'white')
 for i,row in enumerate(rows):sheet.paste(row,(0,i*150))
 sheet.save(out/f'style-{style}.png');print(style,flush=True)
 (out/'samples.json').write_text(json.dumps(records))
