from pathlib import Path
import json
from PIL import Image, ImageDraw
from inkform.generator import get_generator, INFERENCE_LOCK
from inkform.recognition import get_recognizer, recognition_image
from inkform.schema import StyleReference

ROOT = Path(__file__).resolve().parents[2]
folder = ROOT / "output/pdf/comment-test/phrase-probe"
g, r = get_generator(), get_recognizer()
ref = StyleReference.model_validate_json((ROOT / "inkform/assets/default_style.json").read_text())
rows, results = [], []
with INFERENCE_LOCK:
    for text in ("SAT", "A", "Former", "masterclass"):
        for i, reference in enumerate((ref, None)):
            sampled = g._trajectory(text, 0.8, 817, reference)
            if sampled is None:
                continue
            offsets, normalized = sampled
            read = r.read(offsets)
            entry = {"text": text, "index": i, "offsets": offsets, "recognized": read}
            results.append(entry)
            row = Image.new("RGB", (1200, 145), "white")
            ImageDraw.Draw(row).text((12, 4), f"{text} | #{i} | read: {read}", fill="black")
            im = recognition_image(offsets)
            im.thumbnail((1100, 110))
            row.paste(im, (12, 28))
            rows.append(row)
            print(text, i, read, flush=True)
sheet = Image.new("RGB", (1200, 145 * len(rows)), "white")
for i, row in enumerate(rows):
    sheet.paste(row, (0, i * 145))
sheet.save(folder / "acronyms.png")
(folder / "acronyms.json").write_text(json.dumps(results))
