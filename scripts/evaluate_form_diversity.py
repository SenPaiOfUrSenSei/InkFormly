"""Controlled, resumable comparison of neural sampling (no rendering warps)."""
import argparse
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw

from inkform.readable_generator import ReadableGenerator
from inkform.recognition import get_recognizer, compare_readback, recognition_image


TEXTS = ["Every student studies science.", "Eric writes a little letter.",
         "Little letters look a little different.", "Seven students sit beside Sarah.",
         "The year was 2026, with 25 samples.", "I remember the man beneath it."]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/evaluation/form-diversity"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "samples.json"
    records = json.loads(path.read_text()) if path.exists() else []
    generator = ReadableGenerator()
    reader = get_recognizer()
    keys = {(r['text'], r['seed'], r['diversity']) for r in records}
    for text in TEXTS:
        for seed in (42, 817):
            for diversity in (0., .5, 1.):
                if (text, seed, diversity) in keys:
                    continue
                start = time.perf_counter()
                result = generator._trajectory(text, .85, seed, generator.default_reference, diversity)
                record = dict(text=text, seed=seed, diversity=diversity, complete=result is not None)
                if result:
                    record.update(offsets=result[0], readback=compare_readback(text, reader.read(result[0])).model_dump())
                record['seconds'] = time.perf_counter() - start
                records.append(record)
                path.write_text(json.dumps(records))
                print(json.dumps({k:v for k,v in record.items() if k!='offsets'}), flush=True)
    summary = {}
    for diversity in (0., .5, 1.):
        group = [r for r in records if r['diversity']==diversity]
        summary[str(diversity)] = dict(total=len(group), completed=sum(r['complete'] for r in group),
            matching_readbacks=sum(r.get('readback',{}).get('status')=='match' for r in group),
            edit_distance=sum(r.get('readback',{}).get('edit_distance',len(r['text'])) for r in group))
    summary['limitations'] = 'Small fixed-writer probe; blind OCR is not human legibility or writer-identity validation. No retries or render deformations.'
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    for index,text in enumerate(TEXTS):
        rows = [r for r in records if r['text']==text and r['complete']]
        sheet = Image.new('RGB',(1550,len(rows)*155),'white')
        draw = ImageDraw.Draw(sheet)
        for i,row in enumerate(rows):
            draw.text((16,i*155+5),f"Form diversity {row['diversity']} / seed {row['seed']} / OCR: {row['readback']['recognized_text']}",fill='black')
            ink = recognition_image(row['offsets'])
            ink.thumbnail((1500,115))
            sheet.paste(ink,(16,i*155+28))
        sheet.save(args.out/f'comparison-{index+1}.png')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
