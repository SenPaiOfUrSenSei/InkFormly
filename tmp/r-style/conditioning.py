import json,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.schema import StyleReference
from inkform.recognition import recognition_image

def reference(g,height=5.,steps=50):
 # Approximate the single continuous motion seen in the supplied r photo.
 landmarks=np.array([[0.,0.],[.30,.35],[.57,.88],[.61,1.],[.69,.96],[.70,.85],[.63,.78],[.76,.74],[.71,.57],[.61,.26],[.61,.07],[.73,.05],[.94,.32],[1.06,.48]])
 from scipy.interpolate import CubicSpline
 t=np.r_[0,np.cumsum(np.linalg.norm(np.diff(landmarks,axis=0),axis=1))];p=CubicSpline(t,landmarks,axis=0,bc_type='natural')(np.linspace(0,t[-1],steps))*height
 raw=np.asarray(g.default_reference.normalized_offsets);xy=raw[:,:2].cumsum(0);baseline=np.quantile(xy[:,1],.2)
 p+=np.array([xy[-1,0]+height*.75,baseline]);offsets=np.diff(np.vstack((xy[-1],p)),axis=0);ends=np.zeros(len(p));ends[-1]=1
 return StyleReference(text=g.default_reference.text+' r',normalized_offsets=np.vstack((raw,np.column_stack((offsets,ends)))).tolist(),codec='vasquez-native-v1')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--height',type=float,default=5.);p.add_argument('--steps',type=int,default=50);args=p.parse_args()
 out=Path(f'tmp/r-style/conditioning-{args.height}-{args.steps}');out.mkdir(exist_ok=True);g=ReadableGenerator();ref=reference(g,args.height,args.steps);(out/'reference.json').write_text(ref.model_dump_json());rows=[]
 for word in ['river','around','friend','hero']:
  for seed in [42,509]:
   path=out/f'{word}-{seed}.json'
   if path.exists():r=json.loads(path.read_text())
   else:
    s=g._trajectory(word,.95,seed,ref,.5)
    if s is None:print('failed',word,seed,flush=True);continue
    r=dict(text=word,seed=seed,offsets=s[0],readback=g.recognizer_factory().read(s[0]));path.write_text(json.dumps(r));print(word,seed,r['readback'],flush=True)
   rows.append(r)
 sheet=Image.new('RGB',(1400,150*len(rows)),'white');d=ImageDraw.Draw(sheet)
 for i,r in enumerate(rows):
  d.text((10,i*150+3),f"{r['text']} / {r['seed']} / {r['readback']}",fill='black');im=recognition_image(r['offsets']);im.thumbnail((1300,100));sheet.paste(im,(10,i*150+30))
 sheet.save(out/'gallery.png')
