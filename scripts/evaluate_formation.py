"""Compare raw neural letter formation across held-out texts and fixed seeds."""
import json
import argparse
from pathlib import Path
import time

from PIL import Image, ImageDraw

from inkform.generator import GravesGenerator
from inkform.recognition import get_recognizer, recognition_image, compare_readback
from inkform.schema import StyleReference
from inkform.readable_generator import ReadableGenerator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/formation"
TEXTS = [
    "Every student studies science.",
    "Eric writes neatly in English.",
    "Seven students sit beside Sarah.",
    "A small seed starts to sprout.",
    "SAT exams test reading and writing.",
    "This lesson ends at 1500 on Friday.",
]
SEEDS = [42, 817]


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--readable",action="store_true")
    args=parser.parse_args()
    reader = get_recognizer()
    reference = StyleReference.model_validate_json((ROOT / "inkform/assets/default_style.json").read_text())
    records, summary, pictures = [], {}, {}
    for epoch in (52,"readable") if args.readable else (52,46):
        generator = ReadableGenerator() if epoch=="readable" else GravesGenerator(ROOT / f"models/graves/Epoch_{epoch}")
        current_reference = generator.default_reference if epoch=="readable" else reference
        completed = matched = distance = 0
        started = time.perf_counter()
        for index,text in enumerate(TEXTS):
            for seed in SEEDS:
                result = generator._trajectory(text,.8,seed,current_reference)
                record = {"epoch":epoch,"text":text,"seed":seed,"complete":result is not None}
                if result:
                    completed += 1
                    offsets, _ = result
                    observed = reader.read(offsets)
                    review = compare_readback(text,observed)
                    matched += review.status=="match"
                    distance += review.edit_distance
                    record.update(offsets=offsets, readback=review.model_dump())
                    image = recognition_image(offsets)
                    row = Image.new("RGB",(1550,175),"white")
                    ImageDraw.Draw(row).text((12,5),f"Epoch {epoch} / seed {seed} | {observed}",fill="black")
                    image.thumbnail((1500,125))
                    row.paste(image,(14,35))
                    pictures[(epoch,index,seed)] = row
                records.append(record)
                print(json.dumps({k:v for k,v in record.items() if k!="offsets"}),flush=True)
                (OUT / "heldout-samples.json").write_text(json.dumps(records))
        summary[str(epoch)] = {"completed":completed,"total":len(TEXTS)*len(SEEDS),
                               "matching_readbacks":matched,"ocr_edit_distance":distance,
                               "seconds":time.perf_counter()-started}
    for index,text in enumerate(TEXTS):
        alternative="readable" if args.readable else 46
        rows = [pictures[key] for key in [(52,index,42),(52,index,817),(alternative,index,42),(alternative,index,817)] if key in pictures]
        sheet = Image.new("RGB",(1550,175*len(rows)),"white")
        for i,row in enumerate(rows):
            sheet.paste(row,(0,175*i))
        sheet.save(OUT / f"{'readable-' if args.readable else ''}heldout-{index}.png")
    summary["limitations"] = "Readback is not human-labeled accuracy. Inspect sheets for malformed letters even where OCR agrees. No manual substitutions or candidate selection used."
    (OUT / ("readable-heldout-summary.json" if args.readable else "heldout-summary.json")).write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
