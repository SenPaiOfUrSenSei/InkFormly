"""Raw terminal-letter/punctuation stress samples, without OCR-based selection."""
import json
from pathlib import Path

from PIL import Image, ImageDraw

from inkform.readable_generator import ReadableGenerator
from inkform.recognition import compare_readback, get_recognizer, recognition_image

OUT = Path(__file__).resolve().parents[1] / "artifacts/evaluation/formation"
TEXTS = ["A quiet visit", "Write it", "Eric and Sarah rest", "A small red dot",
         "Read this", "Easy to see", "We met at 1500", "History lesson: 1947",
         "A pencil and a pen!", "Meet me here,"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    generator = ReadableGenerator()
    reader = get_recognizer()
    records = []
    for index, text in enumerate(TEXTS):
        sheet = Image.new("RGB", (1500, 300), "white")
        for row, seed in enumerate((43, 818)):
            result = generator._trajectory(text, .8, seed, generator.default_reference)
            record = {"text": text, "seed": seed, "complete": result is not None}
            if result:
                offsets, _ = result
                observed = reader.read(offsets)
                record["readback"] = compare_readback(text, observed).model_dump()
                ImageDraw.Draw(sheet).text((12, row*150+6),
                    f"Input: {text} / seed {seed} / OCR: {observed}", fill="black")
                preview = recognition_image(offsets)
                preview.thumbnail((1450, 105))
                sheet.paste(preview, (12, row*150+35))
                preview.close()
            records.append(record)
            print(json.dumps(record), flush=True)
        sheet.save(OUT / f"terminal-stress-{index}.png")
    (OUT / "terminal-stress.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
