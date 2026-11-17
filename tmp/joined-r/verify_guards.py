import json
from pathlib import Path
from inkform.cursive_selection import _cursive_r_contours
from inkform.joined_r import fit_joined_r
from PIL import Image,ImageDraw
from inkform.recognition import recognition_image
words=('friend','river','around','hero','carry','really','paper','bright')
rows=[]
for word in words:
 counters=0
 for rec in json.loads(Path(f'tmp/r-native/{word}.json').read_text()):
    before,labels=rec['offsets'],rec['labels']
    after,spans=fit_joined_r(before,labels,word,_cursive_r_contours(before,labels,word))
    if spans:
        counters+=1
        if counters<=2:rows.append((word,rec['seed'],before,after,spans))
 print(word,counters,flush=True)
im=Image.new('RGB',(840,len(rows)*92),'white');d=ImageDraw.Draw(im)
for i,(word,seed,before,after,spans) in enumerate(rows):
    y=i*92;d.text((8,y+4),f'{word} / {seed}',fill='black')
    for j,offsets in enumerate((before,after)):
       thumb=recognition_image(offsets);thumb.thumbnail((406,65));im.paste(thumb,(6+j*414,y+24))
im.save('tmp/joined-r/guarded.png')
