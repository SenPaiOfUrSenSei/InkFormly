import json,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
p=Path('output/pdf/college-essay-clear-t/document.json');doc=json.loads(p.read_text());sheet=Image.new('RGB',(1400,1200),'white');draw=ImageDraw.Draw(sheet);n=0
for li,l in enumerate(doc['lines']):
 for a in (l.get('t_formation') or {}).get('adjustments',[]):
  native=np.array(l['offsets'],float);native[:,:2]/=20;native[:,1]*=-1;xy=np.cumsum(native[:,:2],0)
  up,pk,dn=[a[k] for k in ('upstroke_start','stem_peak','downstroke_end')];low=xy[dn,1];high=xy[pk,1];h=high-low
  if h<=0:continue
  cx=xy[pk,0];x0,y0=cx-1.5*h,low-.25*h;s=220/(2*h);col=n%5;row=n//5;ox=col*280;oy=row*300
  draw.text((ox+5,oy+4),f"{li+1}: {l['text']}"[:42],fill='black')
  def pos(pt):return ox+10+(pt[0]-x0)*s,oy+270-(pt[1]-y0)*s
  for j in range(max(0,up-30),min(len(xy)-1,dn+120)):
   if native[j,2]>.5:continue
   pa,pb=pos(xy[j]),pos(xy[j+1])
   if all(ox<=p[0]<=ox+275 and oy+30<=p[1]<=oy+290 for p in (pa,pb)):
    draw.line([pa,pb],fill='blue' if up<=j<=dn else 'grey',width=2)
  n+=1
  if n==20:break
 if n==20:break
sheet.save('tmp/t-crossbar/traces.png')
