import json,numpy as np
from pathlib import Path
from inkform.t_crossbar import raise_t_crossbars
from inkform.recognition import recognition_image
from PIL import Image,ImageDraw
D=json.loads(Path('output/pdf/college-essay-clear-t/document.json').read_text());sheet=Image.new('RGB',(1600,1200),'white');draw=ImageDraw.Draw(sheet);n=0;total=0
for i,l in enumerate(D['lines']):
 a=(l.get('t_formation') or {}).get('adjustments',[])
 if not a:continue
 native=np.array(l['offsets'],float);native[:,:2]/=20;native[:,1]*=-1
 out,bars=raise_t_crossbars(native,a);total+=len(bars)
 if not bars or n>=6:continue
 out=np.array(out);out[:,:2]*=20;out[:,1]*=-1
 for k,v in enumerate((l['offsets'],out.tolist())):
  draw.text((10,n*200+k*95+4),f"{'Before' if k==0 else 'Higher bar'}: {i+1}: {l['text']} ({len(bars)} raised)",fill='black')
  ink=recognition_image(v);ink.thumbnail((1550,75));sheet.paste(ink,(10,n*200+k*95+20))
 n+=1
sheet.save('tmp/t-crossbar/higher-preview.png');print('Confident crossbars:',total)
