import sys,json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import CubicHermiteSpline
sys.path.insert(0,'tmp/r-native')
from separation import shapes
from inkform.recognition import recognition_image

def remake(offsets, labels, text):
 a=np.asarray(offsets);xy=a[:,:2].cumsum(0);changes=[]
 _,spans=shapes(a,labels,text)
 for character,span in zip((i for i,c in enumerate(text) if c=='r'),spans):
    if not span:continue
    entry,peak,low,end,_=span
    if any(labels[i]!=character for i in range(peak,low+1)):continue
    depth=xy[low,1]-xy[peak,1]
    if low-peak<6:continue
    # Seek actual anchors further along the natural entry and exit.
    start=entry
    while (start>1 and not a[start-1,2] and labels[start-1]>=character-1
        and xy[start-1,1]>xy[start,1] and xy[start-1,0]<xy[start,0]
        and xy[start-1,1]-xy[peak,1]<.95*depth):
       start-=1
    stop=end
    while (stop<len(a)-2 and not a[stop,2] and labels[stop+1]<=character+1
        and xy[stop+1,1]<xy[stop,1] and xy[stop+1,0]>xy[stop,0]
        and xy[stop+1,1]-xy[peak,1]>.07*depth):
       stop+=1
    if xy[stop,0]-xy[start,0]<1.05*depth:continue
    if xy[stop,1] > xy[low,1]-.40*depth:continue
    first,last=xy[start],xy[stop]
    base=xy[low,1];anchor=xy[peak,1]; width=(last[0]-first[0])/depth
    crest_x=first[0]+depth*(width*.49)
    taper=depth*min(.21,width*.15)
    target=np.array([first,[crest_x-.20*depth,anchor+.34*depth],[crest_x,anchor],
        [crest_x+taper,anchor+.24*depth],[crest_x-.012*depth,base-.19*depth],
        [crest_x+.11*depth,base],[crest_x+.34*depth,base-.11*depth],last])
    if target[-1,0]<=target[-2,0]:continue
    speed=np.linalg.norm(np.diff(target,axis=0),axis=1)
    t=np.r_[0,speed.cumsum()]
    slope=np.empty_like(target);slope[1:-1]=(target[2:]-target[:-2])/(t[2:]-t[:-2])[:,None]
    slope[0]=(xy[start]-xy[start-1]);slope[0]/=max(np.linalg.norm(slope[0]),1e-6)
    slope[-1]=(xy[stop+1]-xy[stop]);slope[-1]/=max(np.linalg.norm(slope[-1]),1e-6)
    if a[start-1,2]:
        slope[0]=(xy[start+1]-xy[start]);slope[0]/=max(np.linalg.norm(slope[0]),1e-6)
    if a[stop,2]:
        slope[-1]=(xy[stop]-xy[stop-1]);slope[-1]/=max(np.linalg.norm(slope[-1]),1e-6)
    slope[2]*=.20
    slope[3]*=.32
    new=CubicHermiteSpline(t,target,slope)(np.linspace(0,t[-1],stop-start+1))
    changes.append((start,stop,new))
 result=xy.copy()
 for start,stop,new in changes:
    result[start:stop+1]=new
 return np.column_stack((np.diff(result,axis=0,prepend=[[0.,0.]]),a[:,2])).tolist(),changes

if __name__=='__main__':
 rows=[]
 for word in ('friend','river','around','hero','carry','really','paper','bright'):
    for rec in json.loads(Path(f'tmp/r-native/{word}.json').read_text()):
      alt,changes=remake(rec['offsets'],rec['labels'],word)
      if changes:rows.append((word,rec['seed'],rec['offsets'],alt,changes))
 print(len(rows))
 for page in range((len(rows)+13)//14):
    group=rows[14*page:14*(page+1)]
    image=Image.new('RGB',(830,94*len(group)),'white');draw=ImageDraw.Draw(image)
    for i,(word,seed,before,after,changes) in enumerate(group):
       draw.text((5,94*i+4),f'{word} / {seed} / {len(changes)} joins',fill='black')
       for j,offsets in enumerate((before,after)):
          im=recognition_image(offsets);im.thumbnail((412,70));image.paste(im,(6+j*414,94*i+23))
    image.save(f'tmp/joined-r/connect-{page}.png')
