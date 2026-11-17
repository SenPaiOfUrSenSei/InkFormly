import sys,json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
sys.path.insert(0,'tmp/r-native')
from separation import shapes
from inkform.recognition import recognition_image

def shoulder(offsets,labels,text):
 a=np.array(offsets)
 xy=a[:,:2].cumsum(0);new=xy.copy()
 scores, spans=shapes(a,labels,text); changes=[]
 for char,span in zip((i for i,c in enumerate(text) if c=='r'),spans):
    if span is None:continue
    enter,peak,low,out,_=span
    if low-peak<7 or any(labels[p]!=char for p in range(peak,low+1)):continue
    leg=xy[peak:low+1]
    depth=leg[-1,1]-leg[0,1]
    t=(leg[:,1]-leg[0,1])/depth
    if np.min(np.diff(t))<-.005:continue
    y0=leg[0,1]; x0=leg[0,0]
    # Measured from the photo. Display slant is applied independently.
    u=.27
    shoulder_x=x0+.13*depth
    f=np.array([0.,.27,.65,.83,1.])
    desired=np.array([x0,shoulder_x,x0-.07*depth,leg[-1,0]-.025*depth,leg[-1,0]])
    x=np.interp(t,f,desired)
    x[-2:]=leg[-2:,0]
    if np.max(abs(x-leg[:,0]))>.45*depth:continue
    side=xy[low:out+1].copy()
    # Check the altered leg on both sides of the old entry and exit.
    # Example screenshot: the leg lives between the incoming and outgoing ink.
    new[peak:low+1,0]=x
    changes.append((char,peak,low))
 return np.column_stack((np.diff(new,axis=0,prepend=[[0,0]]),a[:,2])).tolist(),changes

if __name__=='__main__':
 rows=[]
 for word in ('friend','river','around','hero','carry','really','paper','bright'):
  for r in json.loads(Path(f'tmp/r-native/{word}.json').read_text()):
    alternate,changes=shoulder(r['offsets'],r['labels'],word)
    if changes:rows.append((word,r['seed'],r['offsets'],alternate,changes))
 image=Image.new('RGB',(860,93*len(rows)),'white');draw=ImageDraw.Draw(image)
 for i,(word,seed,before,after,changes) in enumerate(rows):
    draw.text((5,93*i+4),f'{word}, {seed} | {len(changes)} defined shoulders',fill='black')
    for j,offsets in enumerate((before,after)):
      im=recognition_image(offsets);im.thumbnail((420,66));image.paste(im,(6+j*430,93*i+23))
 image.save('tmp/joined-r/probe.png')
 print(len(rows),flush=True)
