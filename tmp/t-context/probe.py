import json,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.recognition import recognition_image,compare_readback

def score(native,labels,text):
 v=np.asarray(native);xy=np.cumsum(v[:,:2],axis=0);labels=np.asarray(labels);rows=[]
 for c,ch in enumerate(text):
  if ch!='t' or c==0 or not text[c-1].isalpha():continue
  idx=np.flatnonzero(labels==c)
  row=dict(character=c,score=0.,reason='alignment')
  if len(idx)<12:rows.append(row);continue
  blocks=np.split(idx,np.flatnonzero(np.diff(idx)>1)+1);b=max(blocks,key=len)
  y=xy[b,1];smooth=np.convolve(np.pad(y,(1,1),mode='edge'),[.25,.5,.25],mode='valid');pk=int(smooth.argmax());h=float(np.ptp(smooth))
  if pk<4 or pk>len(b)-4 or h<.5:rows.append(row);continue
  up=dn=pk
  while up>0 and smooth[up]-smooth[up-1]>=-.018*h:up-=1
  while dn+1<len(b) and smooth[dn+1]-smooth[dn]<=.018*h:dn+=1
  if pk-up<4 or dn-pk<3:rows.append(row);continue
  d=xy[b[pk:dn+1]];order=np.argsort(d[:,1]);sy,sx=d[order,1],d[order,0]
  low=max(float(xy[b[up],1]),float(sy[0]));high=float(sy[-1]);height=high-low
  if height<.4*h:rows.append(row);continue
  r=xy[b[up:pk+1]];u=(r[:,1]-low)/height;gap=(np.interp(r[:,1],sy,sx)-r[:,0])/height;core=(u>.2)&(u<.8)
  if sum(core)<2:rows.append(row);continue
  loop=float(np.median(np.abs(gap[core])));bars=[]
  for width in (3,5,9,13):
   for start in range(max(0,int(b[up])-20),min(len(xy)-width,int(b[dn])+110)):
    if np.any(v[start:start+width,2]>.5):continue
    seg=xy[start:start+width+1];dx,dy=seg[-1]-seg[0]
    if abs(dx)<.3*height or abs(dy)>.35*abs(dx):continue
    yy=float(np.median(seg[:,1]));rel=(yy-low)/height
    if not .15<rel<.92 or np.ptp(seg[:,1])>.22*height:continue
    xx=float(np.interp(yy,sy,sx))
    if seg[:,0].min()<xx-.06*height and seg[:,0].max()>xx+.06*height:bars.append(rel)
  bar=max(bars,default=0)
  q=1-.65*np.clip((loop-.06)/.4,0,1)-.35*np.clip((.52-bar)/.52,0,1)
  rows.append(dict(character=c,score=float(q),loop=loop,crossbar_height=bar,reason='measured'))
 return dict(score=float(np.mean([r['score'] for r in rows])) if rows else 1.,worst=min([r['score'] for r in rows],default=1.),rows=rows)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--text',default='My notes stayed neat.');p.add_argument('--seed',type=int,default=509);p.add_argument('--count',type=int,default=8);a=p.parse_args()
 out=Path('tmp/t-context')/('probe-'+str(a.seed)+'-'+a.text.split()[0]);out.mkdir(exist_ok=True)
 g=ReadableGenerator();reader=g.recognizer_factory();rs=[]
 for i in range(a.count):
  path=out/f'{i}.json'
  if path.exists():r=json.loads(path.read_text())
  else:
   bias=[.85,1.,1.15,.95,1.1,.85,1.05,1.2][i%8];s=a.seed+i*104729
   t=g._trajectory(a.text,bias,s,g.default_reference,.5)
   if t is None:continue
   offsets,native=t;native,labels=g.t_evaluator.character_trace(a.text,offsets,g.default_reference)
   r=dict(text=a.text,index=i,seed=s,bias=bias,offsets=offsets,quality=score(native,labels,a.text),readback=compare_readback(a.text,reader.read(offsets)).model_dump());path.write_text(json.dumps(r))
  rs.append(r);print(json.dumps({k:v for k,v in r.items() if k!='offsets'}),flush=True)
 sheet=Image.new('RGB',(1600,150*len(rs)),'white');draw=ImageDraw.Draw(sheet)
 for j,r in enumerate(rs):
  draw.text((10,j*150+3),f"{r['index']} score={r['quality']['score']:.3f} worst={r['quality']['worst']:.3f} OCR={r['readback']['recognized_text']} t:{[(round(x.get('loop',-1),2),round(x.get('crossbar_height',-1),2)) for x in r['quality']['rows']]}",fill='black')
  ink=recognition_image(r['offsets']);ink.thumbnail((1560,110));sheet.paste(ink,(10,j*150+30))
 sheet.save(out/'gallery.png')
