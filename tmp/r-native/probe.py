import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.recognition import recognition_image
path=Path('tmp/r-native')
g=ReadableGenerator()
for word in ('friend','river','around','hero','carry','really','paper','bright'):
 rows=[]
 for seed in (42,137,509,817,101,241,317,445,605,756,923,1079,1207,1499,1723,1861):
    take=g._trajectory(word,.85,seed,g.default_reference,.5)
    if take is None:continue
    offsets=take[0]
    _, labels=g.t_evaluator.character_trace(word,offsets,g.default_reference)
    rows.append(dict(seed=seed,offsets=offsets,labels=labels))
 image=Image.new('RGB',(960,128*((len(rows)+3)//4)),'white');draw=ImageDraw.Draw(image)
 for i,r in enumerate(rows):
    x=i%4*240;y=i//4*128
    draw.text((x+10,y+5),f"{word} {r['seed']}",fill='black')
    thumb=recognition_image(r['offsets']);thumb.thumbnail((230,100))
    image.paste(thumb,(x,y+23))
 (path/(word+'.json')).write_text(json.dumps(rows))
 image.save(path/(word+'.png'))
 print(word,len(rows),flush=True)
