import json
from pathlib import Path
from inkform.schema import Document,Line
from inkform.generator import GravesGenerator
r=Path('output/pdf/college-essay-reformed');qa=r/'qa-retakes';overrides=r/'overrides';records=[]
# Whole newly generated neural lines, visually reviewed. No glyph patches.
choices={1:[('My first hero was missing half his body.','intro')],2:[('man beneath it, I turned back to my','p02-l13')],5:[('One afternoon, only Oreo and Pie came','p05-l01')],7:[("for Oreo's. Within a month, Pie was",'p07-l03')],15:[('I still want to fix things for the','p15-l01-extra')]}
for paragraph,items in choices.items():
 name=f'paragraph-{paragraph:02d}.json';path=overrides/name
 source=path if paragraph==1 else r/name
 d=Document.model_validate_json(source.read_text())
 for text,key in items:
  line=Line.model_validate_json((qa/(key+'.json')).read_text());assert line.text==text
  i=next(i for i,l in enumerate(d.lines) if l.text==text)
  old=d.lines[i];d.lines[i]=line;d.generation_seconds+=line.seconds
  records.append(dict(paragraph=paragraph,text=text,selection=key,previous_readback=old.readback.model_dump(),selected_readback=line.readback.model_dump(),method='Visually reviewed unedited neural line, same reference A and requested form diversity .5.'))
 d.warnings=GravesGenerator.quality_warnings(d);path.write_text(d.model_dump_json())
(r/'manual-review.json').write_text(json.dumps(records,indent=2))
