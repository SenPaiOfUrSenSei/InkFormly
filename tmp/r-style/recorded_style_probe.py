from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.schema import StyleReference
from inkform.recognition import recognition_image
out=Path('tmp/r-style/recorded-9');out.mkdir(exist_ok=True);g=ReadableGenerator();folder=Path('models/vasquez/styles');idx=9
ref=StyleReference(text=np.load(folder/f'style-{idx}-chars.npy',allow_pickle=False).tobytes().decode(),normalized_offsets=np.load(folder/f'style-{idx}-strokes.npy',allow_pickle=False).tolist(),codec='vasquez-native-v1');rows=[]
for word in ['river','around','friend','hero']:
 for seed in [42,509]:
  sample=g._trajectory(word,.95,seed,ref,.5)
  if sample is None:print('failed',word,seed,flush=True);continue
  r=dict(text=word,seed=seed,offsets=sample[0],readback=g.recognizer_factory().read(sample[0]));rows.append(r);(out/f'{word}-{seed}.json').write_text(json.dumps(r));print(word,seed,r['readback'],flush=True)
s=Image.new('RGB',(1400,150*len(rows)),'white');d=ImageDraw.Draw(s)
for i,r in enumerate(rows):
 d.text((10,i*150+3),f"{r['text']} / {r['seed']} / {r['readback']}",fill='black');im=recognition_image(r['offsets']);im.thumbnail((1300,100));s.paste(im,(10,i*150+30))
s.save(out/'gallery.png')
