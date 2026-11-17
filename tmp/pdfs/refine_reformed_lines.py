import sys,json
from pathlib import Path
from inkform.generator import get_generator
from inkform.schema import Line
from inkform.recognition import recognition_image
r=Path('output/pdf/college-essay-reformed');out=r/'qa-retakes';out.mkdir(exist_ok=True)
targets=[('intro',(r/'source.txt').read_text().split('\n\n')[0],700013),('p15-l01','I still want to fix things for the',810815),('p05-l01','One afternoon, only Oreo and Pie came',500021),('p07-l03',"for Oreo's. Within a month, Pie was",700033),('p02-l13','man beneath it, I turned back to my',200050)]
g=get_generator()
for i in map(int,sys.argv[1:]):
 key,text,seed=targets[i];path=out/(key+'.json')
 if path.exists():line=Line.model_validate_json(path.read_text())
 else:
  line,_=g.select_line(text,1.,seed,g.default_reference,True,.5)
  path.write_text(line.model_dump_json())
 recognition_image(line.offsets).save(out/(key+'.png'))
 print(key,line.readback.model_dump(),flush=True)
