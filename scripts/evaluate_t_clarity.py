"""Paired raw neural t/T probes. No glyph edits, retries or rendering warps."""
import argparse
import json
from pathlib import Path
import time

from PIL import Image,ImageDraw

from inkform.readable_generator import ReadableGenerator
from inkform.recognition import compare_readback,get_recognizer,recognition_image

DEV = ['The little kitten sat quietly.','She wrote a better letter.',
       'I thought about it.','A SAT test starts at ten.']
HELDOUT = ['We met at the station.','Nothing matters until later.',
           'Write STATE in the notes.','My notes stayed neat.',
           'He counted fifteen steps.','A patient artist painted it.']
CONTROLS = ['My pen moves slowly.','Our calm blue sea.']


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=202)
    p.add_argument('--split',choices=['dev','heldout','controls'],default='dev')
    p.add_argument('--strength',type=float,default=.8)
    p.add_argument('--adapter',action='store_true')
    args=p.parse_args()
    texts={'dev':DEV,'heldout':HELDOUT,'controls':CONTROLS}[args.split]
    out=Path('artifacts/evaluation/t-adapter' if args.adapter else 'artifacts/evaluation/t-clarity')/args.split/f'seed-{args.seed}-strength-{args.strength}'
    out.mkdir(parents=True,exist_ok=True)
    path=out/'samples.json';records=json.loads(path.read_text()) if path.exists() else []
    g=ReadableGenerator(adapter_path=Path('models/vasquez/t_adapter-experimental.pt') if args.adapter else None);reader=get_recognizer()
    for text in texts:
        for strength in (0.,args.strength):
            if any(r['text']==text and r['strength']==strength for r in records):continue
            started=time.perf_counter()
            result=g._trajectory(text,.85,args.seed,g.default_reference,.5,t_clarity=strength,
                t_strategy='adapter' if args.adapter else 'precision')
            record=dict(text=text,seed=args.seed,strength=strength,complete=result is not None)
            if result:
                record.update(offsets=result[0],readback=compare_readback(text,reader.read(result[0])).model_dump())
            record['seconds']=time.perf_counter()-started;records.append(record)
            path.write_text(json.dumps(records))
            print(json.dumps({k:v for k,v in record.items() if k!='offsets'}),flush=True)
    for i,text in enumerate(texts):
        sheet=Image.new('RGB',(1550,340),'white');draw=ImageDraw.Draw(sheet)
        for row,strength in enumerate((0.,args.strength)):
            r=next(r for r in records if r['text']==text and r['strength']==strength)
            draw.text((12,row*170+5),f"T clarity {strength} / {text} / OCR: {r.get('readback',{}).get('recognized_text','Incomplete')}",fill='black')
            if r['complete']:
                ink=recognition_image(r['offsets']);ink.thumbnail((1500,125));sheet.paste(ink,(15,row*170+32))
        sheet.save(out/f'comparison-{i+1}.png')
    summary={str(s):dict(completed=sum(r['complete'] for r in records if r['strength']==s),
        exact_readbacks=sum(r.get('readback',{}).get('status')=='match' for r in records if r['strength']==s),
        edit_distance_with_incomplete_penalty=sum(r.get('readback',{}).get('edit_distance',len(r['text'])) for r in records if r['strength']==s)) for s in (0.,args.strength)}
    summary['limitations']='Small fixed-writer probe; independent OCR is not character-shape validation. Inspect the saved raw trajectories and comparison sheets.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
