"""Keep approved cursive strokes, apply reviewed neural retakes and a render preset."""
import json,re,shutil
from pathlib import Path
from inkform.schema import Document,Line,PageSettings,InkSettings
from inkform.generator import GravesGenerator
from inkform.render import export_document,layout_document
from inkform.variation_presets import VARIATION_PRESETS

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/pdf/college-essay-cursive-A';QA=ROOT/'tmp/pdfs/cursive-A-review'
source=Document.model_validate_json((OUT/'document.json').read_text())
records=json.loads((QA/'phrase-takes.json').read_text())
p2_path=OUT/'paragraph-02.json';original=QA/'paragraph-02-before-retakes.json'
if not original.exists():shutil.copyfile(p2_path,original)
p2=Document.model_validate_json(original.read_text())
def take(text,number):
 line=Line.model_validate(next(r['line'] for r in records if r['text']==text and r['take']==number))
 assert line.readback.status=='match'
 return line
replacement={
 'asked what I wanted to be, I shouted,':[take('asked what I wanted to be, I shouted,',0)],
 'man beneath it, I turned back to my':[take('man beneath it,',1),take('I turned back to my',0)]}
lines=[];changes=[]
for line in p2.lines:
 if line.text in replacement:
  new=replacement[line.text];assert ' '.join(l.text for l in new)==line.text
  lines.extend(new);changes.append({'source_line':line.text,'new_lines':[l.text for l in new],'method':'Visually selected, unedited neural retakes with the same reference.'})
 else:lines.append(line)
p2.lines=lines;p2.generation_seconds+=sum(r['line']['seconds'] for r in records)
p2.warnings=GravesGenerator.quality_warnings(p2);p2_path.write_text(p2.model_dump_json())
composites=json.loads((QA/'composite-retakes.json').read_text())
for paragraph in sorted(set(r['paragraph'] for r in composites)):
 path=OUT/f'paragraph-{paragraph:02d}.json'
 archive=QA/f'paragraph-{paragraph:02d}-before-composites.json'
 if not archive.exists():shutil.copyfile(path,archive)
 d=Document.model_validate_json(archive.read_text())
 by_text={r['text']:r for r in composites if r['paragraph']==paragraph}
 for i,line in enumerate(d.lines):
  if line.text in by_text:
   retake=Line.model_validate(by_text[line.text]['line']);assert retake.text==line.text
   d.lines[i]=retake;d.generation_seconds+=retake.seconds
   changes.append({'source_line':line.text,'new_lines':[retake.text],'method':'Same-reference neural phrases composed across pen-up word gaps.'})
 d.warnings=GravesGenerator.quality_warnings(d);path.write_text(d.model_dump_json())
paragraphs=[Document.model_validate_json((OUT/f'paragraph-{i:02d}.json').read_text()) for i in range(1,18)]
source.lines=[]
rows_per_page=int((297-2*18)//12)
def blank():return Line(text='',offsets=[],seed=817,seconds=0,attempts=0,primed=False,attention_complete=True)
for i,d in enumerate(paragraphs):
 if source.lines and len(source.lines)%rows_per_page:source.lines.append(blank())
 remaining=rows_per_page-len(source.lines)%rows_per_page
 count=len(d.lines)
 # Keep short paragraphs together, and avoid one-line paragraph fragments.
 padding=0
 if (count<=3 and count>remaining) or (remaining==1 and count>1):padding=remaining
 elif count>remaining and (count-remaining)%rows_per_page==1:padding=1
 source.lines.extend(blank() for _ in range(padding))
 source.lines.extend(d.lines)
assert re.sub(r'\s+',' ',source.source_text).strip()==' '.join(l.text for l in source.lines if l.text)
source.generation_seconds=sum(d.generation_seconds for d in paragraphs);source.warnings=GravesGenerator.quality_warnings(source)
(OUT/'document.json').write_text(source.model_dump_json())
preset=VARIATION_PRESETS['lively']
page=PageSettings(**{**preset['page'],'margin_mm':18,'writing_height_mm':8,'line_spacing_mm':12,'layout_seed':29})
ink=InkSettings(**{**preset['ink'],'color':'#203f78','width_mm':.3,'opacity':.98,'roughness':.3})
(OUT/'settings.json').write_text(json.dumps({'page':page.model_dump(),'ink':ink.model_dump()},indent=2))
report={'backend':source.backend,'preset':'lively','written_lines':sum(bool(l.text) for l in source.lines),
 'readback_matches':sum(l.readback.status=='match' for l in source.lines if l.text),
 'review':[{'line':i+1,'text':l.text,'readback':l.readback.model_dump()} for i,l in enumerate(source.lines) if l.text and l.readback.status!='match'],
 'retakes':changes,'note':'Readback is not human-labeled accuracy. No font/glyph replacements or wording edits.'}
(OUT/'readback.json').write_text(json.dumps(report,indent=2))
layout=layout_document(source,page)
manifest=export_document(source,page,ink,OUT,300)
manifest['variation_preset']='lively'
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copyfile(OUT/'inkform.pdf',OUT.parent/'DraftCollegeEssay2-handwritten-cursive-A.pdf')
print(json.dumps({'manifest':manifest,'readback_matches':report['readback_matches'],'written_lines':report['written_lines'],'scale':layout.scale},indent=2))
