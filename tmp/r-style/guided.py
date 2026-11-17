import json,torch,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
from scipy.interpolate import CubicSpline
from inkform.readable_generator import ReadableGenerator
from inkform.recognition import recognition_image
out=Path('tmp/r-style/guided');out.mkdir(exist_ok=True);g=ReadableGenerator();step=g.model.step;draw=g.model.draw;prefix=len(g.default_reference.text)+1;prime=len(g.default_reference.normalized_offsets);D={}
def reset(word,seed):
 D.clear();D.update(word=word,counter=0,xy=np.zeros(2),points=[],queue=[],active=None,finished=set(),finalize=None,rng=np.random.default_rng(seed))
def stepping(pen,state,text):
 i=D['counter'];D['counter']+=1
 if i>=prime:
  if i>prime:
   D['xy']+=pen[0,:2].numpy();D['points'].append(D['xy'].copy())
  if D['active'] is not None or D['finalize'] is not None:
   idx=D['active'] if D['active'] is not None else D['finalize'];target=prefix+idx
   if D['active'] is not None:target+=-.1+.75*(1-len(D['queue'])/32)
   else:target+=.9;D['finalize']=None
   if state.attention is not None:state.position=state.position+(target-float(state.attention.argmax()))
 return step(pen,state,text)
def drawing(state,bias,div=0.,focus=0.):
 idx=int(state.attention.argmax())-prefix
 if not D['queue'] and idx not in D['finished'] and 0<=idx<len(D['word']) and D['word'][idx]=='r':
  D['active']=idx;D['finished'].add(idx);p=D['xy'];rng=D['rng'];h=4.5*rng.uniform(.95,1.05);b=float(np.quantile(np.array(D['points'])[:,1],.15)) if len(D['points'])>5 else p[1]
  shape=np.array([[0.,np.clip((p[1]-b)/h,0,.7)],[.30,.35],[.57,.88],[.61,1.],[.69,.96],[.70,.85],[.63,.78],[.76,.74],[.71,.57],[.61,.26],[.61,.07],[.73,.05],[.94,.32],[1.06,.48]])
  shape[:,0]*=rng.uniform(.95,1.05)
  t=np.r_[0,np.cumsum(np.linalg.norm(np.diff(shape,axis=0),axis=1))];xy=CubicSpline(t,shape,axis=0,bc_type='natural')(np.linspace(0,t[-1],33))*h+np.array([p[0],b]);xy[0]=p
  diffs=np.diff(xy,axis=0);D['queue']=[torch.tensor([[float(x),float(y),0.]],dtype=torch.float32) for x,y in diffs]
 if D['queue']:
  pen=D['queue'].pop(0)
  if not D['queue']:D['finalize']=D['active'];D['active']=None
  return pen
 return draw(state,bias,div,focus)
g.model.step=stepping;g.model.draw=drawing;rows=[]
for word in ['around','hero','river','friend']:
 for seed in [42,509]:
  reset(word,seed);sample=g._trajectory(word,.95,seed,g.default_reference,.5)
  if sample is None:print('failed',word,seed,flush=True);continue
  r=dict(text=word,seed=seed,offsets=sample[0],readback=g.recognizer_factory().read(sample[0]));rows.append(r);(out/f'{word}-{seed}.json').write_text(json.dumps(r));print(word,seed,r['readback'],flush=True)
s=Image.new('RGB',(1400,150*len(rows)),'white');d=ImageDraw.Draw(s)
for i,r in enumerate(rows):
 d.text((10,i*150+3),f"{r['text']} / {r['seed']} / {r['readback']}",fill='black');im=recognition_image(r['offsets']);im.thumbnail((1300,100));s.paste(im,(10,i*150+30))
s.save(out/'gallery.png')
