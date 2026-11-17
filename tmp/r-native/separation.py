from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
from inkform.recognition import recognition_image

def shapes(offsets,labels,text):
    a=np.asarray(offsets);xy=a[:,:2].cumsum(0);labels=np.asarray(labels)
    score=[];details=[]
    for index,c in enumerate(text):
        if c!='r':continue
        owned=np.flatnonzero(labels==index)
        best=0.;best_points=None
        if len(owned)>=4:
            owned_span=np.ptp(xy[owned],axis=0);h=owned_span[1]
            if h<3:return [],[]
            # Ownership trace often starts early. Search near its end, and the
            # beginning of the next character, for an r crest.
            window=np.arange(owned[len(owned)//3],min(len(xy),owned[-1]+9))
            for peak in window:
                if peak < 2 or peak >= len(xy)-2:continue
                x,y=xy[peak]
                if y>xy[peak-1,1] or y>xy[peak+1,1]:continue
                lower_bound=max(0,peak-24)
                before=np.flatnonzero(a[lower_bound:peak,2])
                start=lower_bound+before[-1]+1 if len(before) else lower_bound
                end=min(len(xy),peak+28)
                pen_ends=np.flatnonzero(a[peak:end,2])
                if len(pen_ends):end=peak+pen_ends[0]+1
                if start>=peak or end<=peak+2:continue
                # Find the nearest preceding entry and first valley after the crest.
                b=xy[start:peak]
                above=np.flatnonzero(b[:,1]>=y+.53*h)
                if not len(above):continue
                entry=int(start+above[-1])
                valleys=np.flatnonzero((xy[peak+1:end-1,1]>xy[peak:end-2,1])&(xy[peak+1:end-1,1]>=xy[peak+2:end,1]))+peak+1
                valleys=valleys[xy[valleys,1]>=y+.45*h]
                if not len(valleys):continue
                low=int(valleys[0])
                exit_candidates=np.flatnonzero(xy[low+1:end,1]<=xy[low,1]-.28*h)
                if not len(exit_candidates):continue
                out=int(low+1+exit_candidates[0])
                height=xy[low,1]-y;rise=x-xy[entry,0];back=x-xy[low,0];advance=xy[out,0]-xy[low,0]
                if height>1.25*h or not .22<rise/height<2.3 or not -.02<back/height<.75 or not .12<advance/height<1.2:continue
                # Prefer a clearly advancing entry, a modest backstroke and
                # an exit that does not simply retrace that backstroke.
                descent=xy[peak:low+1];path=np.linalg.norm(np.diff(descent,axis=0),axis=1).sum()
                efficiency=float(np.linalg.norm(descent[-1]-descent[0])/max(path,.001))
                if efficiency<.7:continue
                level=xy[low,1]-.28*h
                entry_x=float(np.interp(level,xy[entry:peak+1,1][::-1],xy[entry:peak+1,0][::-1]))
                leg_x=float(np.interp(level,xy[peak:low+1,1],xy[peak:low+1,0]))
                out_frac=(level-xy[out-1,1])/(xy[out,1]-xy[out-1,1])
                out_x=xy[out-1,0]+out_frac*(xy[out,0]-xy[out-1,0])
                separation=min((out_x-leg_x)/height,.5)
                if separation<0 or leg_x<entry_x-.09*height:continue
                slope_penalty=abs(back/height-.25)*.8+abs(rise/height-.95)*.23
                usable=.65-slope_penalty+1.45*separation+.15*(1-efficiency)
                if usable>best:
                   best=usable;best_points=(entry,int(peak),low,out,round(back/height,2))
        score.append(round(best,3));details.append(best_points)
    return score,details

if __name__=='__main__':
 for p in sorted(Path('tmp/r-native').glob('*.json')):
    word=p.stem
    rows=json.loads(p.read_text())
    for r in rows:r['scores'],r['points']=shapes(r['offsets'],r['labels'],word)
    rows.sort(key=lambda r:(sum(s>0 for s in r['scores']),sum(r['scores'])),reverse=True)
    image=Image.new('RGB',(1200,126*((len(rows)+1)//2)),'white');draw=ImageDraw.Draw(image)
    for i,r in enumerate(rows):
      ox=i%2*600;oy=i//2*126
      draw.text((ox+15,oy+4),f"{word} {r['seed']} {r['scores']}",fill='black')
      a=np.array(r['offsets']);xy=a[:,:2].cumsum(0);span=np.ptp(xy,axis=0)
      xy=(xy-xy.min(0))*min(550/span[0],86/span[1])+[ox+24,oy+28]
      for j in range(1,len(a)):
       if a[j-1,2]:continue
       c='black'
       for pts in r['points']:
        if pts and pts[0]<j<=pts[3]:c='red' if j<=pts[1] else 'blue' if j<=pts[2] else 'green'
       draw.line(tuple(xy[j-1])+tuple(xy[j]),fill=c,width=2)
    image.save(p.with_name(word+'-separation.png'))
    print(word,[(r['seed'],r['scores']) for r in rows[:6]])
