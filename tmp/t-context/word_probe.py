from pathlib import Path
import json,re,numpy as np,torch
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.recognition import recognition_image

def cropwords(offsets,labels,text):
 xy=np.cumsum(np.asarray(offsets)[:,:2],axis=0);labels=np.asarray(labels)
 # Blank vertical strips must separate entire trajectories, including delayed bars.
 lo,hi=xy[:,0].min(),xy[:,0].max();height=np.ptp(xy[:,1]);step=height/200;nb=int(np.ceil((hi-lo)/step))+1;occupied=np.zeros(nb,bool)
 for i in range(len(xy)-1):
  if offsets[i][2]>.5:continue
  a,b=sorted(((xy[i,0]-lo)/step,(xy[i+1,0]-lo)/step));occupied[max(0,int(np.floor(a))):min(nb,int(np.ceil(b))+1)]=True
 gaps=[];blank=np.flatnonzero(~occupied);runs=np.split(blank,np.flatnonzero(np.diff(blank)>1)+1)
 for run in runs:
  if len(run)>5 and run[0]>0 and run[-1]<nb-1:gaps.append((lo+run[0]*step,lo+run[-1]*step))
 cuts=[lo-height*.05]
 for m in re.finditer(r'\s+',text):
  idx=np.flatnonzero((labels>=m.start())&(labels<m.end()))
  if not len(idx):return []
  middle=np.median(xy[idx,0]);choices=[(abs((a+b)/2-middle),(a+b)/2) for a,b in gaps if a>cuts[-1] and abs((a+b)/2-middle)<height*.65]
  if not choices:return []
  cuts.append(min(choices)[1])
 cuts.append(hi+height*.05)
 words=list(re.finditer(r'\S+',text));result=[]
 if len(words)+1!=len(cuts) or any(a>=b for a,b in zip(cuts,cuts[1:])):return []
 for m,a,b in zip(words,cuts,cuts[1:]):
  ids=np.flatnonzero((xy[:,0]>a)&(xy[:,0]<b))
  # Retain whole pen segments and never slice an ink edge at a boundary.
  pts=[]
  for i in ids:
   if i>0 and not offsets[i-1][2] and not a<xy[i-1,0]<b:return []
   pts.append([float(xy[i,0]),float(xy[i,1]),int(offsets[i][2])])
  if not pts:return []
  arr=np.asarray(pts);arr[:,:2]=np.diff(np.vstack((np.zeros((1,2)),arr[:,:2])),axis=0);arr[-1,2]=1
  result.append((m.group(),arr.tolist()))
 return result

if __name__=='__main__':
 g=ReadableGenerator();r=g.recognizer_factory();paths=sorted(Path('tmp/t-context/probe-509-My').glob('[0-9].json'));sheet=Image.new('RGB',(1600,150*len(paths)),'white');draw=ImageDraw.Draw(sheet)
 for j,p in enumerate(paths):
  d=json.loads(p.read_text());_,labels=g.t_evaluator.character_trace(d['text'],d['offsets'],g.default_reference);words=cropwords(d['offsets'],labels,d['text']);scores=[]
  for w,off in words:
   if 't' not in w.lower():continue
   image=recognition_image(off);pixels=r.processor(images=image,return_tensors='pt').pixel_values;ids=r.processor.tokenizer(w,return_tensors='pt').input_ids
   decoder=torch.cat((torch.full((1,1),r.model.generation_config.decoder_start_token_id,dtype=torch.long),ids[:,:-1]),dim=1)
   with torch.inference_mode():loss=r.model(pixel_values=pixels,decoder_input_ids=decoder,labels=ids).loss
   obs=r.read(off);scores.append((w,obs,round(float(loss),3)))
  draw.text((10,j*150+4),f'{p.stem}: {scores}',fill='black');ink=recognition_image(d['offsets']);ink.thumbnail((1560,105));sheet.paste(ink,(10,j*150+28));print(p.stem,scores,flush=True)
 sheet.save('tmp/t-context/word-scores.png')
