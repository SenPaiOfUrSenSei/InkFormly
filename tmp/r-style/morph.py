import json,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
from scipy.interpolate import CubicSpline
from inkform.recognition import recognition_image,get_recognizer

def morph(offsets,labels,text):
 data=np.asarray(offsets,float);xy=data[:,:2].cumsum(0);out=xy.copy();labels=np.asarray(labels);audit=[]
 for idx,ch in enumerate(text):
  if ch!='r':continue
  ids=np.flatnonzero(labels==idx)
  if len(ids)<12 or np.any(np.diff(ids)!=1) or np.any(data[ids[:-1],2]>.5):continue
  old=xy[ids];A,B=old[0],old[-1];width=B[0]-A[0];height=np.ptp(old[:,1])
  if height<2 or not .3*height<width<3*height:continue
  ref=np.array([[0.,0.],[.30,.35],[.57,.88],[.61,1.],[.69,.96],[.70,.85],[.63,.78],[.76,.74],[.71,.57],[.61,.26],[.61,.07],[.73,.05],[.94,.32],[1.06,.48]])
  t=np.r_[0,np.cumsum(np.linalg.norm(np.diff(ref,axis=0),axis=1))];s=np.linspace(0,1,len(ids));p=CubicSpline(t/t[-1],ref,axis=0,bc_type='natural')(s)
  target=np.column_stack((A[0]+p[:,0]/1.06*width,A[1]+s*(B[1]-A[1])-height*(p[:,1]-s*.48)))
  smooth=lambda z: np.clip(z,0,1)**3*(10-15*np.clip(z,0,1)+6*np.clip(z,0,1)**2)
  w=smooth(s/.16)*smooth((1-s)/.16)
  out[ids]=old*(1-w[:,None])+target*w[:,None];audit.append(idx)
 result=data.copy();result[:,:2]=np.diff(np.vstack((np.zeros((1,2)),out)),axis=0)
 return result.tolist(),audit

if __name__=='__main__':
 out=Path('tmp/r-style/morph');out.mkdir(exist_ok=True);reader=get_recognizer();rows=[]
 for p in Path('tmp/r-style').glob('*.json'):
  r=json.loads(p.read_text());off,a=morph(r['offsets'],r['labels'],r['text'])
  if not a:continue
  obs=reader.read(off);row=dict(**r,morphed=off,after_readback=obs,changed=a);rows.append(row);(out/p.name).write_text(json.dumps(row));print(r['text'],r['seed'],obs,flush=True)
 sheet=Image.new('RGB',(1400,260*len(rows)),'white');d=ImageDraw.Draw(sheet)
 for i,r in enumerate(rows):
  for n,key in enumerate(('offsets','morphed')):
   y=i*260+n*130;d.text((10,y+3),f"{r['text']} {r['seed']} {'original' if n==0 else r['after_readback']}",fill='black');im=recognition_image(r[key]);im.thumbnail((1300,90));sheet.paste(im,(10,y+28))
 sheet.save(out/'gallery.png')
