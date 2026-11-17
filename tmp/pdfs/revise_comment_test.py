"""Visually reviewed repair of the three reported passages in the test PDF.

Words use the selected neural phrases. The SAT acronym and :) are explicitly
drawn vector strokes; this is a sample correction, not a claim of model accuracy.
"""
import json
from pathlib import Path
import numpy as np

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from inkform.generator import get_generator
from inkform.recognition import compare_readback, get_recognizer
from inkform.schema import Document, PageSettings, InkSettings
from inkform.render import layout_document, render_page

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/pdf/comment-test"
probe = FOLDER / "phrase-probe"
document = Document.model_validate_json((FOLDER / "document.json").read_text())
choices = json.loads((probe / "candidates.json").read_text())
acronyms = json.loads((probe / "acronyms.json").read_text())


def choose(text, index, data=choices):
    return next(item["offsets"] for item in data if item["text"] == text and item["index"] == index)


def as_offsets(strokes):
    result = []
    previous = np.zeros(2)
    for stroke in strokes:
        for i, point in enumerate(stroke):
            result.append((float(point[0]-previous[0]), float(point[1]-previous[1]), int(i==len(stroke)-1)))
            previous = point
    return result


def literal_strokes(symbol):
    if symbol == ":)":
        dots = [np.array([[0, -0.55], [0, -0.54]]), np.array([[0, -0.16], [0, -0.15]])]
        t = np.linspace(0, 1, 35)
        bracket = np.column_stack((0.34 + 0.24*np.sin(np.pi*t), -0.92 + 1.1*t))
        return as_offsets(dots + [bracket])
    if symbol == "SAT":
        def bezier(a, b, c, d):
            t = np.linspace(0, 1, 25)[:, None]
            return (1-t)**3*np.array(a)+3*(1-t)**2*t*np.array(b)+3*(1-t)*t*t*np.array(c)+t**3*np.array(d)
        s = np.vstack([bezier((.52,-.82),(.05,-1.08),(-.13,-.58),(.25,-.46)),
                       bezier((.25,-.46),(.85,-.20),(.41,.11),(0,-.04))[1:]])
        a = np.array([[.75,0],[1,-.88],[1.28,0]])
        bar = np.array([[.86,-.34],[1.17,-.34]])
        top = np.array([[1.45,-.87],[2,-.87]])
        stem = np.array([[1.725,-.87],[1.725,0]])
        return as_offsets([s,a,bar,top,stem])
    raise ValueError(symbol)


def expected_span(text):
    upper = 1.5 if any(c.isupper() or c.isdigit() or c in "bdfhklt" for c in text) else 1.0
    lower = .65 if any(c in "gjpqy" for c in text) else 0
    return upper + lower


def compose(fragments, xheight):
    """Place independent pen fragments on a common baseline, preserving lifts."""
    output, previous = [], np.zeros(2)
    cursor = 0
    for offsets, is_literal, text in fragments:
        data = np.asarray(offsets, dtype=float)
        xy = np.cumsum(data[:,:2], axis=0)
        if is_literal:
            scale = xheight * (1.5/.88 if text == "SAT" else 1.4/1.1)
            baseline = 0
        else:
            scale = xheight * expected_span(text) / max(float(np.ptp(xy[:,1])), 1)
            baseline = float(np.quantile(xy[:,1], .8 if any(c in "gjpqy" for c in text) else .95))
        xy[:,0] = (xy[:,0] - xy[:,0].min()) * scale + cursor
        xy[:,1] = (xy[:,1] - baseline) * scale
        if output:
            output[-1] = (output[-1][0], output[-1][1], 1)
        for point, end in zip(xy, data[:,2]):
            output.append((float(point[0]-previous[0]),float(point[1]-previous[1]),int(end)))
            previous = point
        cursor = float(xy[:,0].max()) + xheight * .6
    output[-1] = (output[-1][0],output[-1][1],1)
    return output


heights = [float(np.ptp(np.cumsum(np.asarray(line.offsets)[:,:2],axis=0)[:,1])) / expected_span(line.text) for line in document.lines]
height = float(np.median(heights))
document.lines[0].offsets = compose([
    (choose("Hey Eric!",1),False,"Hey Eric!"),
    (choose("Former",1,acronyms),False,"Former"),
    (literal_strokes("SAT"),True,"SAT"),
    (choose("masterclass",0,acronyms),False,"masterclass"),
    (choose("student here,",0),False,"student here,"),
],height)

prefix_path = probe / "last-prefix.json"
if prefix_path.is_file():
    prefix = json.loads(prefix_path.read_text())["offsets"]
else:
    generator = get_generator()
    prefix_line, _ = generator.select_line("you for helping me get over a 1500",.8,817,document.reference,True)
    prefix = prefix_line.offsets
    prefix_path.write_text(json.dumps({"offsets":prefix,"readback":prefix_line.readback.model_dump()}))
document.lines[-1].offsets = compose([
    (prefix,False,"you for helping me get over a 1500"), (choose("on the sat",0),False,"on the sat"), (literal_strokes(":)"),True,":)")
],height)

reader = get_recognizer()
for i in (0,len(document.lines)-1):
    document.lines[i].readback = compare_readback(document.lines[i].text,reader.read(document.lines[i].offsets))
document.warnings.append("Visually repaired sample: selected neural phrases; SAT block capitals and :) are explicit vector strokes. OCR is not visual certification.")
(FOLDER / "document-v2.json").write_text(document.model_dump_json())
page, ink = PageSettings(margin_mm=25, writing_height_mm=6.5), InkSettings(width_mm=.25)
layout = layout_document(document,page)
pdf_path = ROOT / "output/pdf/handwriting-comment-test-v2.pdf"
mm=72/25.4
pdf=canvas.Canvas(str(pdf_path),pagesize=(210*mm,297*mm))
pdf.setTitle("InkForm comment test - visually revised")
for page_index,lines in enumerate(layout.pages):
    for line in lines:
        for stroke in line.strokes:
            stroke[:,1] += 20
        line.baseline_mm += 20
    im=render_page(lines,page,ink,300,page_index)
    im.save(FOLDER / f"revised-page-{page_index+1:03d}.png",dpi=(300,300))
    pdf.drawImage(ImageReader(im),0,0,width=210*mm,height=297*mm)
    if page_index==0:
        pdf.setFillColorRGB(.20,.25,.29)
        pdf.setFont("Helvetica-Bold",11)
        pdf.drawString(25*mm,273*mm,"@Chelumpkeif")
        pdf.setFont("Helvetica",9)
        pdf.setFillColorRGB(.46,.49,.50)
        pdf.drawString(25*mm,268*mm,"1 year ago")
    pdf.showPage()
    im.close()
pdf.save()
(FOLDER / "revision-notes.json").write_text(json.dumps({
    "selected_neural_phrases":{"Hey Eric!":1,"student here,":0,"on the sat":0},
    "explicit_vector_corrections":["SAT",":)"],
    "reason":"Reported ambiguous initial s, malformed capital E, and invented trailing punctuation.",
},indent=2))
print(pdf_path,flush=True)
