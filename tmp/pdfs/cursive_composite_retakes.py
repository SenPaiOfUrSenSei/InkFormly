"""Short neural phrase retakes joined across pen-up word gaps, never glyph replacement."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from inkform.generator import get_generator,INFERENCE_LOCK
from inkform.schema import Document,Line
from inkform.recognition import get_recognizer,recognition_image,compare_readback
from inkform.render import strokes_from_offsets

OUT=Path('tmp/pdfs/cursive-A-review');OUT.mkdir(parents=True,exist_ok=True)
g=get_generator();reader=get_recognizer()
ref=Document.model_validate_json(Path('output/pdf/college-essay-cursive-A/paragraph-02.json').read_text()).reference
CASES=[(13,'shop for my usual Pepsi. Outside, a',['shop for my usual Pepsi.','Outside, a']),
 (13,'milk. As he ate, I petted him, knowing',['milk. As he ate,','I petted him, knowing']),
 (14,"The old question hadn't disappeared.",['The old question',"hadn't disappeared."]),
 (14,'I knew the dog in front of me was',['I knew the dog','in front of me was'])]

def join(components):
 strokes=[];cursor=0.;previous=np.zeros(2);offsets=[]
 for text,line in components:
  ss=strokes_from_offsets(line.offsets);points=np.concatenate(ss);cx=points[:,0]-points[:,0].mean();var=float(cx@cx)
  slope=float(cx@(points[:,1]-points[:,1].mean())/var) if var else 0
  for s in ss:s[:,1]-=.9*slope*(s[:,0]-points[0,0])
  points=np.concatenate(ss);left=points[:,0].min();right=points[:,0].max();baseline=np.quantile(points[:,1],.8)
  for s in ss:strokes.append(s-np.array([left,baseline])+np.array([cursor,0]))
  cursor+=right-left+max(40.,.6*(right-left)/max(len(text),1))
 for s in strokes:
  for i,point in enumerate(s):
   delta=point-previous;offsets.append((float(delta[0]),float(delta[1]),int(i==len(s)-1)));previous=point
 return offsets

records=[]
for case_index,(paragraph,text,parts) in enumerate(CASES):
 assert ' '.join(parts)==text
 best=None;start=time.perf_counter()
 for attempt in range(2):
  components=[]
  for j,part in enumerate(parts):
   with INFERENCE_LOCK:line,_=g.select_line(part,.95,817+500009+case_index*100003+attempt*104729+j*1009,ref,True)
   components.append((part,line))
  offsets=join(components);readback=compare_readback(text,reader.read(offsets))
  line=Line(text=text,offsets=offsets,seed=components[0][1].seed,seconds=time.perf_counter()-start,attempts=sum(l.attempts for _,l in components),primed=True,attention_complete=True,readback=readback)
  record={'paragraph':paragraph,'text':text,'line':line.model_dump(),'components':[l.model_dump() for _,l in components]}
  if best is None or readback.edit_distance<best['line']['readback']['edit_distance']:best=record
  if readback.status=='match':break
 records.append(best);(OUT/'composite-retakes.json').write_text(json.dumps(records));print(paragraph,text,best['line']['readback'],flush=True)
 im=recognition_image(best['line']['offsets']);im.thumbnail((1650,130));row=Image.new('RGB',(1700,185),'white');draw=ImageDraw.Draw(row);draw.text((10,4),text,fill='black');draw.text((10,22),best['line']['readback']['recognized_text'],fill='black');row.paste(im,(10,45));row.save(OUT/f'composite-{case_index+1}.png')
