"""Inspect normal checked generation on reported cases and new text.

This uses the production factory: no phrase overrides or hand-edited strokes.
Readback scores describe the selection mechanism, not independent accuracy.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw

from inkform.generator import get_generator
from inkform.recognition import recognition_image
from inkform.schema import GenerationSettings

OUT = Path(__file__).resolve().parents[1] / "artifacts/evaluation/formation"
TEXTS = [
    "student here,",
    "Hey Eric! Former SAT masterclass student here,",
    "on the sat :)",
    "Seven students sit beside Sarah.",
    "Every student studies science.",
    "This lesson ends at 1500 on Friday.",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    generator = get_generator()
    records = []
    rows = []
    for text in TEXTS:
        for seed in (42, 817):
            document = generator.generate(GenerationSettings(text=text, seed=seed))
            (OUT / f"checked-{len(records):02d}.json").write_text(document.model_dump_json())
            line = document.lines[0]
            assert len(document.lines) == 1
            record = {"text": text, "seed": seed, "chosen_seed": line.seed,
                      "attempts": line.attempts, "seconds": line.seconds,
                      "readback": line.readback.model_dump()}
            records.append(record)
            print(json.dumps(record), flush=True)
            row = Image.new("RGB", (1500, 165), "white")
            ImageDraw.Draw(row).text((15, 8), f"Input: {text} | seed {seed} | {line.readback.status}", fill="black")
            preview = recognition_image(line.offsets)
            preview.thumbnail((1450, 115))
            row.paste(preview, (15, 40))
            preview.close()
            rows.append(row)
    for index in range(len(TEXTS)):
        pair = Image.new("RGB", (1500, 330), "white")
        pair.paste(rows[index*2], (0, 0))
        pair.paste(rows[index*2+1], (0, 165))
        pair.save(OUT / f"checked-pair-{index}.png")
    sheet = Image.new("RGB", (1500, 165*len(rows)), "white")
    for i, row in enumerate(rows):
        sheet.paste(row, (0, i*165))
    sheet.save(OUT / "checked-formation.png")
    report = {"backend": generator.backend, "samples": records,
              "readback_matches": sum(r["readback"]["status"] == "match" for r in records),
              "limitations": "Same OCR used for selection and scoring; visually inspect shapes. No training, manual glyphs, or per-word overrides."}
    (OUT / "checked-summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
