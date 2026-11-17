from pathlib import Path
from inkform.schema import Document,GenerationSettings
from inkform.readable_generator import ReadableGenerator
from inkform.cursive_selection import select_cursive_words

g=ReadableGenerator()
old=Document.model_validate_json(Path('tmp/r-native/selection/before.json').read_text())
probe=g.generate(GenerationSettings(text='The bright stars rose over the barn.',seed=329),
                 lambda *message:print('baseline:',message[-1],flush=True))
cases=[(line,old.seed+i*1009) for i,line in enumerate(old.lines)]+[(probe.lines[0],329)]
before=[];after=[]
for line,seed in cases:
    print('checking:',line.text,flush=True)
    preferred,_=select_cursive_words(g,(line,None),old.reference,.85,.5,seed)
    print(line.text,preferred.readback.status,[(w['word'],w['placement'].get('in_place',False),len(w['contours'])) for w in (preferred.r_selection or {}).get('words',[])],flush=True)
    before.append(line);after.append(preferred)
    Path('tmp/joined-r/before.json').write_text(old.model_copy(update={'lines':before}).model_dump_json())
    Path('tmp/joined-r/after.json').write_text(old.model_copy(update={'lines':after,'prefer_cursive_r':True}).model_dump_json())
