from pathlib import Path
import sys,json
import numpy as np
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.recognition import recognition_image,compare_readback
from inkform.word_clarity import word_regions,place_complete_word,lexical_word
from eval_core import shapes
g=ReadableGenerator();reader=g.recognizer_factory();idx=int(sys.argv[1])
source=json.loads(Path(f'artifacts/evaluation/cursive-r-context-final/line-{idx}.json').read_text())['before']
text=source['text'];current=source['offsets'];original=current
_,labels=g.t_evaluator.character_trace(text,current,g.default_reference)
regions=word_regions(current,labels,text);audit=[]
def score(offsets,ls,txt):
 scores,_=shapes(offsets,ls,txt)
 return np.array(scores)
for index,region in reversed(list(enumerate(regions or []))):
 word=region['text']
 if 'r' not in word:continue
 before=score(region['offsets'],[labels[j]-region['start'] for j in region['source_indices']],word)
 print(word,'ownership',region['ownership_verified'],'baseline',before,flush=True)
 if not region['ownership_verified']:continue
 caption=word.strip('.,;:!?"()[]');original_evidence=reader.evidence(region['offsets'],caption)
 candidates=[]
 for n in range(16):
  seed=(source['seed']+(n+1)*104729+(index+1)*15485863)%(2**31-1)
  bias=(.85,.73,.95,.80)[n%4];div=(.5,.7,.5,.9)[n%4]
  draw=g._trajectory(word,bias,seed,g.default_reference,div)
  if draw is None:continue
  placed=place_complete_word(current,region,draw[0])
  if placed is None:continue
  _,local=g.t_evaluator.character_trace(word,draw[0],g.default_reference)
  after=score(draw[0],local,word)
  if before.shape!=after.shape or np.any(after<before-.08) or np.any(after<.38) or after.mean()<before.mean()+.12:continue
  candidates.append((float(after.mean()),seed,draw,placed,after))
 print(word,'candidate scores',[(s,seed) for s,seed,_,_,_ in candidates],flush=True)
 for rank,seed,draw,placed,after in sorted(candidates,key=lambda x:x[0],reverse=True)[:5]:
  values,transform=placed
  start=transform['start'];stop=start+transform['new_points']
  a=np.array(values);xy=a[:,:2].cumsum(0)[start:stop]
  crop=np.column_stack((np.diff(np.vstack((np.zeros((1,2)),xy)),axis=0),a[start:stop,2]))
  evidence=reader.evidence(crop.tolist(),caption)
  if lexical_word(evidence['recognized_text'])!=lexical_word(caption) or evidence['nll']>original_evidence['nll']+.1:
    print(word,'word evidence',original_evidence,evidence,flush=True);continue
  line_readback=compare_readback(text,reader.read(values))
  if line_readback.status!='match':
    print(word,'line evidence',line_readback.recognized_text,flush=True);continue
  current=values
  audit.append(dict(word=word,seed=seed,before=before.tolist(),after=after.tolist()))
  print(word,'selected',seed,after,flush=True)
  break
out=Path('tmp/r-native')/f'local-{idx}'
out.with_suffix('.json').write_text(json.dumps(dict(text=text,before=original,after=current,audit=audit)))
image=Image.new('RGB',(1200,244),'white');d=ImageDraw.Draw(image)
for k,(name,values) in enumerate((('Before',original),('After',current))):
 d.text((10,5+k*122),name,fill='black')
 thumb=recognition_image(values);thumb.thumbnail((1180,100));image.paste(thumb,(6,22+k*122))
image.save(out.with_suffix('.png'))
