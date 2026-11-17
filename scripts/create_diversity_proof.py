"""One-page vector proof from the controlled checked comparison's saved strokes."""
import json
from pathlib import Path

import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from inkform.render import strokes_from_offsets


def main():
    records = json.loads(Path('artifacts/evaluation/form-diversity/checked/samples.json').read_text())
    output = Path('output/pdf/InkForm-letter-form-variation-proof.pdf')
    output.parent.mkdir(parents=True,exist_ok=True)
    pdf = canvas.Canvas(str(output),pagesize=A4)
    pdf.setTitle('InkForm — learned letter-form variation proof')
    pdf.setAuthor('InkForm')
    _,height = A4
    pdf.setFillColorRGB(.1,.15,.23)
    pdf.setFont('Helvetica-Bold',17)
    pdf.drawString(18*mm,height-21*mm,'More variation in the pen movements')
    pdf.setFont('Helvetica',9)
    for i,line in enumerate(('Same connected-cursive reference A. Original sampler vs. 50% form diversity.',
                             'Neural trajectories only: shape warps, layout irregularity and ink texture are off.')):
        pdf.drawString(18*mm,height-(29+i*5)*mm,line)
    texts = ['Little letters look a little different.', 'Seven students sit beside Sarah.',
             'I remember the man beneath it.']
    for index,text in enumerate(texts):
        top = height-(48+index*66)*mm
        pdf.setFillColorRGB(.1,.15,.23)
        pdf.setFont('Helvetica-Bold',10)
        pdf.drawString(18*mm,top,text)
        lines = [next(r['line'] for r in records if r['text']==text and r['diversity']==d) for d in (0.,.5)]
        prepared = [strokes_from_offsets(line['offsets']) for line in lines]
        bounds = [(np.concatenate(strokes).min(0),np.concatenate(strokes).max(0)) for strokes in prepared]
        scale = min(174*mm/max(high[0]-low[0] for low,high in bounds),
                    11*mm/max(high[1]-low[1] for low,high in bounds))
        for row,(strokes,(low,high)) in enumerate(zip(prepared,bounds)):
            y = top-(12+row*24)*mm
            pdf.setFont('Helvetica',8)
            pdf.setFillColorRGB(.4,.43,.48)
            pdf.drawString(18*mm,y,'Original' if row==0 else 'More learned movement choices')
            pdf.setStrokeColorRGB(.125,.247,.47)
            pdf.setLineWidth(.45)
            pdf.setLineCap(1)
            pdf.setLineJoin(1)
            for stroke in strokes:
                xy = (stroke-low)*scale
                path = pdf.beginPath()
                for j,(x,dy) in enumerate(xy):
                    if j==0:path.moveTo(18*mm+float(x),y-4*mm-float(dy))
                    else:path.lineTo(18*mm+float(x),y-4*mm-float(dy))
                pdf.drawPath(path)
    pdf.setFillColorRGB(.36,.39,.44)
    pdf.setFont('Helvetica',8)
    notes = ['Small development comparison; the samples use different retained seeds after readback retries.',
             'All three new samples matched OCR. The original final sentence missed its period in OCR.',
             'OCR cannot guarantee well-formed letters. No model weights were retrained.',
             'This is a comparison proof; the earlier full essay PDF has not been regenerated.']
    for i,line in enumerate(notes):
        pdf.drawString(18*mm,(28-i*4.5)*mm,line)
    pdf.save()
    print(output.resolve())


if __name__=='__main__':
    main()
