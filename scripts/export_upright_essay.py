"""Render the reviewed coherent essay with less rightward letter lean."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from pypdf import PdfReader

from inkform.render import export_document,layout_document
from inkform.schema import Document,PageSettings,InkSettings


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'output/pdf/college-essay-coherent'
    out=root/'output/pdf/college-essay-coherent-upright'
    out.mkdir(parents=True,exist_ok=True)
    document_json=(source/'document.json').read_text()
    document=Document.model_validate_json(document_json)
    original=json.loads((source/'settings.json').read_text())
    page=PageSettings(**{**original['page'],'slant_adjustment_deg':-18})
    ink=InkSettings(**original['ink'])
    layout=layout_document(document,page)
    assert len(layout.pages)==6
    for rows in layout.pages:
        for row,line in enumerate(rows):
            if not line.strokes:continue
            points=np.concatenate(line.strokes)
            assert np.isfinite(points).all()
            assert points[:,0].min()>=page.margin_mm-.01
            assert points[:,0].max()<=210-page.margin_mm+.01
            assert points[:,1].min()>=page.margin_mm+row*page.line_spacing_mm-.01
            assert points[:,1].max()<=page.margin_mm+(row+1)*page.line_spacing_mm+.01
    manifest=export_document(document,page,ink,out,300)
    target=out.parent/'DraftCollegeEssay2-handwritten-coherent-upright.pdf'
    shutil.copyfile(out/manifest['pdf'],target)
    (out/'document.json').write_text(document_json)
    (out/'settings.json').write_text(json.dumps(dict(page=page.model_dump(),ink=ink.model_dump()),indent=2))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    assert document.model_dump_json()==Document.model_validate_json(document_json).model_dump_json()
    reader=PdfReader(target)
    assert len(reader.pages)==6
    for p in reader.pages:
        assert abs(float(p.mediabox.width)-595.2756)<.01
        assert abs(float(p.mediabox.height)-841.8898)<.01
    report=dict(pages=6,relative_slant_adjustment_degrees=-18,source_text_and_trajectories_unchanged=True,
        margins_and_row_bounds_verified=True,pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        visually_reviewed_pages=[],note='A common horizontal shear reduces rightward letter lean. The angle is a relative adjustment, not the measured final letter slant. Readback remains the base-trajectory estimate.')
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(target)


if __name__=='__main__':main()
