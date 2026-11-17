"""Reproduce the rejected higher-crossbar experiment; excluded from generation."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from pypdf import PdfReader

from inkform.generator import GravesGenerator
from inkform.recognition import get_recognizer,compare_readback
from inkform.render import export_document,layout_document
from inkform.schema import Document,PageSettings,InkSettings
from inkform.t_crossbar import raise_t_crossbars


def main():
    source=Path('output/pdf/college-essay-clear-t')
    out=Path('output/pdf/college-essay-higher-t-crossbars');out.mkdir(exist_ok=True)
    original=Document.model_validate_json((source/'document.json').read_text())
    doc=original.model_copy(deep=True);reader=get_recognizer();changes=[]
    rank=lambda line:(not line.readback.numbers_match,line.readback.edit_distance)
    for index,line in enumerate(doc.lines):
        stems=(line.t_formation or {}).get('adjustments',[])
        if not stems:continue
        native=np.asarray(line.offsets,dtype=float);native[:,:2]/=20;native[:,1]*=-1
        for strength in (.85,.55):
            corrected,audit=raise_t_crossbars(native,stems,strength)
            if not audit:break
            rendered=np.asarray(corrected);rendered[:,:2]*=20;rendered[:,1]*=-1
            candidate=line.model_copy(deep=True)
            candidate.offsets=[(float(x),float(y),int(end)) for x,y,end in rendered]
            candidate.readback=compare_readback(line.text,reader.read(candidate.offsets))
            if rank(candidate)>rank(line):continue
            candidate.t_formation['crossbar_adjustments']=audit
            doc.lines[index]=candidate
            changes.append(dict(line=index+1,text=line.text,crossbars=audit,
                before_readback=line.readback.model_dump(),after_readback=candidate.readback.model_dump()))
            break
        print(f'Checked line {index+1}/{len(doc.lines)}',flush=True)
    assert doc.source_text==original.source_text and doc.reference==original.reference
    assert [l.text for l in doc.lines]==[l.text for l in original.lines]
    doc.warnings=GravesGenerator.quality_warnings(doc)
    settings=json.loads((source/'settings.json').read_text())
    page=PageSettings(**settings['page']);ink=InkSettings(**settings['ink'])
    layout=layout_document(doc,page)
    for rows in layout.pages:
        for row,line in enumerate(rows):
            if not line.strokes:continue
            xy=np.concatenate(line.strokes)
            assert xy[:,0].min()>=page.margin_mm-.01 and xy[:,0].max()<=210-page.margin_mm+.01
            assert xy[:,1].min()>=page.margin_mm+row*page.line_spacing_mm-.01
            assert xy[:,1].max()<=page.margin_mm+(row+1)*page.line_spacing_mm+.01
    manifest=export_document(doc,page,ink,out,300)
    target=out.parent/'DraftCollegeEssay2-handwritten-higher-t-crossbars.pdf'
    shutil.copyfile(out/manifest['pdf'],target)
    assert len(PdfReader(target).pages)==len(layout.pages)==7
    report=dict(pages=7,source_text_and_paragraphs_preserved=True,writer_reference_preserved=True,
        layout_settings_preserved=True,changed_lines=len(changes),raised_crossbars=sum(len(c['crossbars']) for c in changes),
        matches_before=sum(l.readback.status=='match' for l in original.lines if l.text),
        matches_after=sum(l.readback.status=='match' for l in doc.lines if l.text),
        changes=changes,visually_reviewed_pages=[],pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    for name,value in [('document',doc.model_dump()),('settings',settings),('manifest',manifest),('verification',report)]:
        (out/f'{name}.json').write_text(json.dumps(value))
    print(json.dumps({k:v for k,v in report.items() if k!='changes'}),flush=True)


if __name__=='__main__':main()
