from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from inkform.recognition import recognition_image
p=Path('models/vasquez/styles');rows=[]
for source in sorted(p.glob('*-strokes.npy'),key=lambda q:int(q.name.split('-')[1])):
 index=int(source.name.split('-')[1]);text=np.load(p/f'style-{index}-chars.npy',allow_pickle=False).tobytes().decode();a=np.load(source,allow_pickle=False).copy();a[:,:2]*=20;a[:,1]*=-1;rows.append((index,text,a.tolist()))
s=Image.new('RGB',(1600,160*len(rows)),'white');d=ImageDraw.Draw(s)
for i,(idx,text,off) in enumerate(rows):
 d.text((10,i*160+3),f'{idx}: {text}',fill='black');im=recognition_image(off);im.thumbnail((1500,110));s.paste(im,(10,i*160+32))
s.save('tmp/r-style/real-references.png')
