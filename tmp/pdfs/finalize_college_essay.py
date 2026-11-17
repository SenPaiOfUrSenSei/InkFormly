import json
from pathlib import Path
import re
import shutil

from inkform.schema import Document,Line,PageSettings,InkSettings
from inkform.generator import GravesGenerator
from inkform.render import export_document

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/pdf/college-essay'
QA=ROOT/'tmp/pdfs/college-essay-review'
initial=OUT/'document-initial.json'
if not initial.exists():shutil.copyfile(OUT/'document.json',initial)
document=Document.model_validate_json(initial.read_text())
records=json.loads((QA/'new-takes.json').read_text())
changes=[]
for number in (74,80,96):
 record=next(r for r in records if r['number']==number and r['take']==0)
 replacement=Line.model_validate(record['line'])
 original=document.lines[number-1]
 assert replacement.text==original.text and replacement.readback.status=='match'
 document.lines[number-1]=replacement
 changes.append({'line':number,'old_seed':original.seed,'new_seed':replacement.seed,
                 'text':replacement.text,'reason':'Clearer capital I in a visually inspected, unedited neural take.'})
document.generation_seconds+=sum(r['line']['seconds'] for r in records)
document.warnings=GravesGenerator.quality_warnings(document)
assert re.sub(r'\s+',' ',document.source_text).strip()==' '.join(l.text for l in document.lines if l.text)
(OUT/'document.json').write_text(document.model_dump_json())
report=json.loads((OUT/'readback.json').read_text())
report.update(visual_rewrites=changes,seconds=document.generation_seconds,warnings=document.warnings,
              visual_review='All four pages and all six OCR-flagged lines inspected; no wording edits or manual glyph substitutions.')
(OUT/'readback.json').write_text(json.dumps(report,indent=2))
for path in sorted(OUT.glob('paragraph-*.json')):
 paragraph=Document.model_validate_json(path.read_text())
 changed=False
 for i,line in enumerate(paragraph.lines):
  for update in changes:
   if line.text==update['text'] and line.seed==update['old_seed']:
    paragraph.lines[i]=document.lines[update['line']-1]
    paragraph.generation_seconds+=sum(r['line']['seconds'] for r in records if r['number']==update['line'])
    changed=True
 if changed:
  paragraph.warnings=GravesGenerator.quality_warnings(paragraph)
  path.write_text(paragraph.model_dump_json())
settings=json.loads((OUT/'settings.json').read_text())
manifest=export_document(document,PageSettings.model_validate(settings['page']),InkSettings.model_validate(settings['ink']),OUT,300)
shutil.copyfile(OUT/manifest['pdf'],OUT.parent/'DraftCollegeEssay2-handwritten.pdf')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'manifest':manifest,'visual_rewrites':changes},indent=2))
