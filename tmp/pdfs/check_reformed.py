import hashlib,json
from pathlib import Path
import numpy as np
from pypdf import PdfReader
from inkform.schema import Document,PageSettings,InkSettings
from inkform.render import layout_document
root=Path('output/pdf/college-essay-reformed')
d=Document.model_validate_json((root/'document.json').read_text())
s=json.loads((root/'settings.json').read_text());p=PageSettings(**s['page']);ink=InkSettings(**s['ink'])
assert d.form_diversity==.5 and d.backend=='vasquez-17900-style2'
assert ' '.join(d.source_text.split())==' '.join(l.text for l in d.lines if l.text)
assert all(l.attention_complete and l.sampling_form_diversity is not None for l in d.lines if l.text)
assert all(l.readback.status in ('match','review') for l in d.lines if l.text)
layout=layout_document(d,p)
for page in layout.pages:
 for line in page:
  if not line.strokes:continue
  a=np.concatenate(line.strokes);assert np.isfinite(a).all()
  assert a[:,0].min()>=p.margin_mm-.01 and a[:,0].max()<=210-p.margin_mm+.01
  assert a[:,1].min()>=p.margin_mm-.01 and a[:,1].max()<=297-p.margin_mm+.01
pdfpath=Path('output/pdf/DraftCollegeEssay2-handwritten-reformed.pdf');pdf=PdfReader(pdfpath)
assert len(pdf.pages)==len(layout.pages)
for page in pdf.pages:
 assert abs(float(page.mediabox.width)-595.2756)<.01
 assert abs(float(page.mediabox.height)-841.8898)<.01
report=dict(words=len(d.source_text.split()),written_lines=sum(bool(l.text) for l in d.lines),pages=len(pdf.pages),source_text_preserved=True,bounds_verified=True,form_diversity=d.form_diversity,readback_matches=sum(l.readback.status=='match' for l in d.lines if l.text),review_lines=sum(l.readback.status=='review' for l in d.lines if l.text),pdf_sha256=hashlib.sha256(pdfpath.read_bytes()).hexdigest(),note='Source text equality verifies generation requests, not every visible character. Final page images also require visual review.')
(root/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
