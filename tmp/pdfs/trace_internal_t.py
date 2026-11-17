import json
from pathlib import Path
import torch,numpy as np
from torch.nn import functional as F
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.stroke_network import ALPHABET
root=Path('artifacts/evaluation/t-refinement/heldout/seed-904');r=next(x for x in json.loads((root/'samples.json').read_text()) if x['text']=='My notes stayed neat.')
g=ReadableGenerator(adapter_path=None);text=r['text'];prefix=g.default_reference.text+' ';full=prefix+text
ctx=F.one_hot(torch.tensor([[ALPHABET.index(c) for c in full]+[0]]),len(ALPHABET)).float();state=g.model.initial_state();native=torch.tensor(r['baseline']['offsets'],dtype=torch.float32);native[:,:2]/=20;native[:,1]*=-1
trace=[]
with torch.inference_mode():
 for point in g._primer(g.default_reference):state=g.model.step(point.reshape(1,3),state,ctx)
 pen=torch.tensor([[0.,0.,1.]])
 for point in native:
  state=g.model.step(pen,state,ctx);trace.append(int(state.attention.argmax())-len(prefix));pen=point.reshape(1,3)
xy=native[:,:2].cumsum(0).numpy();labels=np.array(trace);targets=[i for i,c in enumerate(text) if c=='t' and i and text[i-1].isalpha()]
sheet=Image.new('RGB',(1400,420),'white');draw=ImageDraw.Draw(sheet)
for col,k in enumerate(targets):
 indices=np.flatnonzero(labels==k);low,high=xy[indices].min(0),xy[indices].max(0);span=high-low;s=min(350/max(span[0],1),290/max(span[1],1));origin=np.array([30+col*450,70]);p=(xy[indices]-low)*[s,-s]+origin+[0,290]
 draw.text((30+col*450,10),f'{text} char {k}, steps {indices[0]}..{indices[-1]}',fill='black')
 start=0
 for j in range(len(indices)):
  if j and indices[j]!=indices[j-1]+1:start=j
  if j>start:draw.line([tuple(p[j-1]),tuple(p[j])],fill='blue',width=2)
  if j%4==0:draw.text(tuple(p[j]),str(j),fill='red')
  if native[indices[j],2]>.5:draw.ellipse((p[j,0]-3,p[j,1]-3,p[j,0]+3,p[j,1]+3),fill='black');start=j+1
 print(k,'range',indices[0],indices[-1],'points',len(indices),'span',span,'lifts',[(j,round(float(xy[i,0]),1),round(float(xy[i,1]),1)) for j,i in enumerate(indices) if native[i,2]>.5])
sheet.save('tmp/pdfs/internal-t-traces.png')
Path('tmp/pdfs/internal-t-traces.json').write_text(json.dumps(dict(text=text,native=native.tolist(),labels=trace)))
