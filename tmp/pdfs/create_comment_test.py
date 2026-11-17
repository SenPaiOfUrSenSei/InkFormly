"""Make the requested InkForm comment test with exact source metadata."""
from pathlib import Path
import json

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from inkform.generator import get_generator
from inkform.schema import GenerationSettings, PageSettings, InkSettings
from inkform.render import layout_document, render_page

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/pdf/comment-test"
BODY = ("Hey Eric! Former SAT masterclass student here, you are truly the goat!! "
        "Even if you can't afford the classes, these videos and the free resources "
        "are legitimately life changing! Thank you for helping me get over a 1500 "
        "on the sat :)")
FOLDER.mkdir(parents=True, exist_ok=True)
source = "@Chelumpkeif\n1 year ago\n" + BODY
(FOLDER / "source.txt").write_text(source)
settings = GenerationSettings(text=BODY, seed=42, page=PageSettings(margin_mm=25, writing_height_mm=6.5),
                              ink=InkSettings(width_mm=0.25))
document_path = FOLDER / "document.json"
if document_path.is_file():
    from inkform.schema import Document
    document = Document.model_validate_json(document_path.read_text())
else:
    document = get_generator().generate(settings, lambda i, n, text: print(f"Writing {i+1}/{n}: {text}", flush=True))
    document_path.write_text(document.model_dump_json())

layout = layout_document(document, settings.page)
pdf_path = ROOT / "output/pdf/handwriting-comment-test.pdf"
mm = 72 / 25.4
pdf = canvas.Canvas(str(pdf_path), pagesize=(210 * mm, 297 * mm))
pdf.setTitle("InkForm handwritten comment test")
pdf.setAuthor("InkForm")
for index, lines in enumerate(layout.pages):
    # Leave room for the source's two metadata lines at the top of page one.
    if index == 0:
        for line in lines:
            for stroke in line.strokes:
                stroke[:, 1] += 20
            line.baseline_mm += 20
    image = render_page(lines, settings.page, settings.ink, 300, index)
    image.save(FOLDER / f"page-{index+1:03d}.png", dpi=(300, 300))
    pdf.drawImage(ImageReader(image), 0, 0, width=210*mm, height=297*mm)
    if index == 0:
        pdf.setFillColorRGB(0.20, 0.25, 0.29)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(25*mm, (297-24)*mm, "@Chelumpkeif")
        pdf.setFont("Helvetica", 9)
        pdf.setFillColorRGB(0.46, 0.49, 0.50)
        pdf.drawString(25*mm, (297-29)*mm, "1 year ago")
    pdf.showPage()
    image.close()
pdf.save()
review = [{"text": line.text, "readback": line.readback.model_dump(), "attempts": line.attempts}
          for line in document.lines if line.text]
(FOLDER / "readback.json").write_text(json.dumps(review, indent=2))
print(json.dumps({"pdf": str(pdf_path), "pages": len(layout.pages), "review": review}, indent=2), flush=True)
