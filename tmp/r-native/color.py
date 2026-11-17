import sys,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
for word in sys.argv[1:]:
 p=Path('tmp/r-native')/f'{word}.json'
 rows=json.loads(p.read_text())
 pic=Image.new('RGB',(1200,130*((len(rows)+1)//2)),'white');draw=ImageDraw.Draw(pic)
 for i,r in enumerate(rows):
    ox=i%2*600;oy=i//2*130
    draw.text((ox+15,oy+5),f"{word} {r['seed']}",fill='black')
    a=np.array(r['offsets']);xy=a[:,:2].cumsum(0);lab=np.array(r['labels']);span=np.ptp(xy,axis=0)
    xy=(xy-xy.min(0))*min(550/span[0],87/span[1])+[ox+24,oy+27]
    for j in range(1,len(a)):
      if a[j-1,2]:continue
      c=('navy','purple','teal','orange','darkgreen')[lab[j]%5]
      if lab[j] in [k for k,l in enumerate(word) if l=='r']:c='red'
      draw.line(tuple(xy[j-1])+tuple(xy[j]),fill=c,width=2)
 pic.save(p.with_name(word+'-color.png'))
