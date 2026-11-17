import sys,json
from pathlib import Path
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.recognition import recognition_image,compare_readback
from separation import shapes
from inkform.word_clarity import word_regions
sentences=('A friend walked around the river.','The hero returned with courage.','I rarely carry a paper map.','Bright stars rose over the trees.')
start=int(sys.argv[1]);text=sentences[start]
g=ReadableGenerator();results=[]
for n in range(13):
 seed=517+n*7919
 bias=(.85,.95,1.05,.75,.85)[n%5];div=(.5,.5,.7,.9,.9)[n%5]
 take=g._trajectory(text,bias,seed,g.default_reference,div)
 if not take:continue
 offsets=take[0];_,labels=g.t_evaluator.character_trace(text,offsets,g.default_reference)
 readings=g.recognizer_factory().read(offsets)
 regions=word_regions(offsets,labels,text)
 ranks=[]
 if regions:
    for region in regions:
      word=region['text'];idx=region['source_indices']
      if 'r' in word:
        scores,_=shapes(region['offsets'],[labels[j]-region['start'] for j in idx],word)
        ranks.extend(float(x) for x in scores)
 results.append(dict(seed=seed,bias=bias,div=div,readback=readings,distance=compare_readback(text,readings).edit_distance,scores=ranks,offsets=offsets,labels=labels))
 print(start,seed,readings,ranks,flush=True)
path=Path('tmp/r-native')/f'sentence-{start}';path.with_suffix('.json').write_text(json.dumps(results))
results.sort(key=lambda x:(x['distance'], -sum(x['scores'])))
im=Image.new('RGB',(1170,120*len(results)),'white');d=ImageDraw.Draw(im)
for i,r in enumerate(results):
 d.text((10,i*120+5),f"{r['seed']} / {r['readback']} / {r['scores']}",fill='black')
 thumb=recognition_image(r['offsets']);thumb.thumbnail((1140,88));im.paste(thumb,(3,i*120+25))
im.save(path.with_suffix('.png'))
