"""Larger upright essay with independently fitted lines and varied page edges."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from pypdf import PdfReader

from inkform.generator import GravesGenerator
from inkform.render import layout_document,export_document
from inkform.schema import Document,Line,PageSettings,InkSettings


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'output/pdf/college-essay-coherent-upright'
    out=root/'output/pdf/college-essay-larger-relaxed'
    out.mkdir(parents=True,exist_ok=True)
    original=Document.model_validate_json((source/'document.json').read_text())
    doc=original.model_copy(deep=True)
    written=[l for l in original.lines if l.text]
    # Recover actual paragraph boundaries and remove padding for the old page
    # size. The new spacing requires its own pagination, including widow care.
    paragraphs=[];cursor=0
    for text in original.source_text.split('\n\n'):
        target=' '.join(text.split());lines=[]
        while ' '.join(l.text for l in lines)!=target:
            assert cursor<len(written)
            lines.append(written[cursor].model_copy(deep=True));cursor+=1
            assert target.startswith(' '.join(l.text for l in lines))
        paragraphs.append(lines)
    assert cursor==len(written)
    intro=Document.model_validate_json((out/'intro.json').read_text())
    assert intro.reference==original.reference
    assert ' '.join(l.text for l in intro.lines)==original.source_text.split('\n\n')[0]
    paragraphs[0]=intro.lines
    saved=json.loads((source/'settings.json').read_text())
    page=PageSettings(**{**saved['page'],'writing_height_mm':8.8,'line_spacing_mm':13.5,
        'fit_lines_individually':True,'naturalness':.65,'line_start_variation':1.,
        'line_end_variation':.7,'line_spacing_variation':.3,'line_tilt_variation':.2,
        'baseline_variation':.3})
    ink=InkSettings(**saved['ink'])
    rows=int((297-2*page.margin_mm)//page.line_spacing_mm)
    def blank():return Line(text='',offsets=[],seed=817,seconds=0,attempts=0,primed=False,attention_complete=True)
    doc.lines=[]
    for lines in paragraphs:
        if doc.lines and len(doc.lines)%rows:doc.lines.append(blank())
        remaining=rows-len(doc.lines)%rows;count=len(lines);padding=0
        if (count<=3 and count>remaining) or (remaining==1 and count>1):padding=remaining
        elif count>remaining and (count-remaining)%rows==1:padding=1
        doc.lines.extend(blank() for _ in range(padding));doc.lines.extend(lines)
    assert ' '.join(doc.source_text.split())==' '.join(l.text for l in doc.lines if l.text)
    doc.generation_seconds+=intro.generation_seconds
    doc.warnings=GravesGenerator.quality_warnings(doc)
    layout=layout_document(doc,page)
    actual=[]
    for rows_in_page in layout.pages:
        for row,line in enumerate(rows_in_page):
            if not line.strokes:continue
            points=np.concatenate(line.strokes);actual.append(points)
            assert np.isfinite(points).all()
            assert points[:,0].min()>=page.margin_mm-.01 and points[:,0].max()<=210-page.margin_mm+.01
            assert points[:,1].min()>=page.margin_mm+row*page.line_spacing_mm-.01
            assert points[:,1].max()<=page.margin_mm+(row+1)*page.line_spacing_mm+.01
    old_page=PageSettings(**saved['page']);old_layout=layout_document(original,old_page)
    old_points=[np.concatenate(l.strokes) for p in old_layout.pages for l in p if l.strokes]
    old_height=float(np.median([np.ptp(p[:,1]) for p in old_points]))
    new_height=float(np.median([np.ptp(p[:,1]) for p in actual]))
    assert new_height>old_height*1.12
    manifest=export_document(doc,page,ink,out,300)
    target=out.parent/'DraftCollegeEssay2-handwritten-larger-relaxed.pdf'
    shutil.copyfile(out/manifest['pdf'],target)
    (out/'document.json').write_text(doc.model_dump_json())
    (out/'settings.json').write_text(json.dumps(dict(page=page.model_dump(),ink=ink.model_dump()),indent=2))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    reader=PdfReader(target);assert len(reader.pages)==len(layout.pages)
    for p in reader.pages:
        assert abs(float(p.mediabox.width)-595.2756)<.01 and abs(float(p.mediabox.height)-841.8898)<.01
    report=dict(pages=len(reader.pages),source_text_preserved=True,source_paragraphs=len(paragraphs),
        upright_adjustment_preserved=page.slant_adjustment_deg,previous_median_line_height_mm=old_height,
        median_line_height_mm=new_height,height_increase_percent=100*(new_height/old_height-1),
        start_range_mm=[float(min(p[:,0].min() for p in actual)),float(max(p[:,0].min() for p in actual))],
        margins_and_row_bounds_verified=True,newly_generated_intro_readbacks=[l.readback.model_dump() for l in intro.lines],
        visually_reviewed_pages=[],pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        note='All previously reviewed takes are retained except the opening sentence, newly synthesized as two lines. Whole lines are fitted independently; no glyphs were replaced.')
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(pdf=str(target),report=report),indent=2))


if __name__=='__main__':main()
