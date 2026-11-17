"""Compare the checked pipeline with the original and broader motion sampler."""
import json
from pathlib import Path

from PIL import Image, ImageDraw

from inkform.readable_generator import ReadableGenerator
from inkform.recognition import recognition_image
from scripts.evaluate_form_diversity import TEXTS


def main():
    out = Path('artifacts/evaluation/form-diversity/checked')
    out.mkdir(parents=True,exist_ok=True)
    path = out/'samples.json'
    records = json.loads(path.read_text()) if path.exists() else []
    keys = {(r['text'],r['diversity']) for r in records}
    generator = ReadableGenerator()
    for text in TEXTS:
        for diversity in (0.,.5):
            if (text,diversity) in keys:
                continue
            line,_ = generator.select_line(text,.85,817,generator.default_reference,True,diversity)
            record = dict(text=text,diversity=diversity,line=line.model_dump())
            records.append(record)
            path.write_text(json.dumps(records))
            print(json.dumps({**record,'line':{k:v for k,v in record['line'].items() if k!='offsets'}}),flush=True)
    summary = {str(d): dict(lines=len([r for r in records if r['diversity']==d]),
        matches=sum(r['line']['readback']['status']=='match' for r in records if r['diversity']==d),
        edit_distance=sum(r['line']['readback']['edit_distance'] for r in records if r['diversity']==d),
        seconds=sum(r['line']['seconds'] for r in records if r['diversity']==d)) for d in (0.,.5)}
    summary['limitations'] = 'Six development sentences, one seed, same writer primer. Selected OCR is not an independent test of readability, writer identity or human likeness. No render augmentation.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    for i,text in enumerate(TEXTS):
        sheet = Image.new('RGB',(1550,350),'white')
        draw = ImageDraw.Draw(sheet)
        for n,d in enumerate((0.,.5)):
            row = next(r for r in records if r['text']==text and r['diversity']==d)
            line = row['line']
            draw.text((15,n*175+5),f"Requested {d}, retained {line['sampling_form_diversity']} / readback: {line['readback']['recognized_text']}",fill='black')
            ink = recognition_image(line['offsets'])
            ink.thumbnail((1500,130))
            sheet.paste(ink,(15,n*175+32))
        sheet.save(out/f'comparison-{i+1}.png')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
