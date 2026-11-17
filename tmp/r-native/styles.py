import json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator,READABLE_MODEL_DIR
from inkform.schema import StyleReference
from inkform.recognition import recognition_image
g=ReadableGenerator();word=sys.argv[1];rows=[]
for i in (2,3,4,5,6,8,9,11,12):
 directory=READABLE_MODEL_DIR/'styles'
 reference=StyleReference(text=np.load(directory/f'style-{i}-chars.npy').tobytes().decode(),normalized_offsets=np.load(directory/f'style-{i}-strokes.npy').tolist(),codec='vasquez-native-v1')
 for seed in (42,137,317,445):
    take=g._trajectory(word,.87,seed,reference,.5)
    if take:rows.append(dict(writer=i,seed=seed,offsets=take[0]))
path=Path('tmp/r-native')/f'styles-{word}';path.with_suffix('.json').write_text(json.dumps(rows))
im=Image.new('RGB',(1080,114*((len(rows)+3)//4)),'white');d=ImageDraw.Draw(im)
for i,r in enumerate(rows):
 x=i%4*270;y=i//4*114
 d.text((x+10,y+5),f"{word} / style {r['writer']} / {r['seed']}",fill='black')
 thumb=recognition_image(r['offsets']);thumb.thumbnail((257,84));im.paste(thumb,(x,y+25))
im.save(path.with_suffix('.png'))
