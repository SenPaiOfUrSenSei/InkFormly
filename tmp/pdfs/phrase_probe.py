"""Inspect short model-generated phrases instead of trusting sentence OCR."""
from pathlib import Path
import json

from PIL import Image, ImageDraw

from inkform.generator import get_generator, INFERENCE_LOCK
from inkform.recognition import get_recognizer, recognition_image
from inkform.schema import StyleReference

ROOT = Path(__file__).resolve().parents[2]
folder = ROOT / "output/pdf/comment-test/phrase-probe"
folder.mkdir(parents=True, exist_ok=True)
generator = get_generator()
recognizer = get_recognizer()
reference = StyleReference.model_validate_json((ROOT / "inkform/assets/default_style.json").read_text())
phrases = ["Hey Eric!", "Former SAT masterclass", "student here,", "on the sat"]
results = []
rows = []
with INFERENCE_LOCK:
    for phrase in phrases:
        for index, bias in enumerate((0.8, 1.0, 1.15, 0.65)):
            seed = 817 + index * 7919
            sampled = generator._trajectory(phrase, bias, seed, reference)
            if sampled is None:
                print(phrase, index, "incomplete", flush=True)
                continue
            offsets, normalized = sampled
            readback = recognizer.read(offsets)
            entry = {"text": phrase, "index": index, "bias": bias, "seed": seed,
                     "offsets": offsets, "normalized": normalized, "recognized": readback}
            results.append(entry)
            preview = recognition_image(offsets)
            row = Image.new("RGB", (1300, 145), "white")
            draw = ImageDraw.Draw(row)
            draw.text((12, 4), f"{phrase} | #{index} bias={bias} | readback: {readback}", fill="#454545")
            preview.thumbnail((1270, 110))
            row.paste(preview, (12, 28))
            rows.append(row)
            print(phrase, index, readback, flush=True)
            (folder / "candidates.json").write_text(json.dumps(results))
sheet = Image.new("RGB", (1300, len(rows) * 145), "white")
for i, row in enumerate(rows):
    sheet.paste(row, (0, i * 145))
sheet.save(folder / "comparison.png")
