"""Compare the guarded production t policy against the complete original pipeline."""
import argparse
import json
from pathlib import Path

from PIL import Image,ImageDraw

from inkform.readable_generator import ReadableGenerator
from inkform.recognition import recognition_image
from scripts.evaluate_t_clarity import DEV,HELDOUT,CONTROLS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--split',choices=['dev','heldout','controls'],default='heldout')
    parser.add_argument('--rerank-experiment',action='store_true')
    args=parser.parse_args()
    texts={'dev':DEV,'heldout':HELDOUT,'controls':CONTROLS+['SAT STATE TITLE']}[args.split]
    root=Path('artifacts/evaluation/t-refinement' if args.rerank_experiment else 'artifacts/evaluation/t-stem-refinement')/args.split/f'seed-{args.seed}'
    root.mkdir(parents=True,exist_ok=True);path=root/'samples.json'
    records=json.loads(path.read_text()) if path.exists() else []
    g=ReadableGenerator(adapter_path=Path('models/vasquez/t_candidate_adapter.pt') if args.rerank_experiment else None)
    for text in texts:
        if any(r['text']==text for r in records):continue
        baseline=g.select_line(text,.85,args.seed,g.default_reference,True,.5,False)
        refined=(g._experimental_refine_t_line if args.rerank_experiment else g._refine_t_line)(text,.85,args.seed,g.default_reference,.5,baseline)
        a,b=baseline[0],refined[0]
        rank=lambda line:(not line.readback.numbers_match,line.readback.edit_distance)
        assert rank(b)<=rank(a)
        if b.offsets!=a.offsets and args.rerank_experiment:
            assert b.t_formation['margin']>g.t_evaluator.score(text,a.offsets,g.default_reference)['margin']+.025
        record=dict(text=text,baseline=a.model_dump(),refined=b.model_dump(),changed=b.offsets!=a.offsets)
        records.append(record);path.write_text(json.dumps(records))
        print(json.dumps(dict(text=text,baseline_readback=a.readback.model_dump(),refined_readback=b.readback.model_dump(),changed=record['changed'],formation=b.t_formation,adapter=b.t_adapter_version)),flush=True)
    for index,row in enumerate(records):
        sheet=Image.new('RGB',(1550,340),'white');draw=ImageDraw.Draw(sheet)
        for n,key in enumerate(('baseline','refined')):
            line=row[key];draw.text((12,n*170+4),f"{key}: {row['text']} | readback: {line['readback']['recognized_text']}",fill='black')
            ink=recognition_image(line['offsets']);ink.thumbnail((1500,125));sheet.paste(ink,(15,n*170+32))
        sheet.save(root/f'comparison-{index+1}.png')
    summary={key:dict(exact_readbacks=sum(r[key]['readback']['status']=='match' for r in records),edit_distance=sum(r[key]['readback']['edit_distance'] for r in records),seconds=sum(r[key]['seconds'] for r in records)) for key in ('baseline','refined')}
    summary.update(total=len(records),changed=sum(r['changed'] for r in records),adapter_selected=sum(bool(r['refined']['t_adapter_version']) for r in records),limitations=(
        'Experimental selection readback and self-model t contrast are not human-labeled legibility.' if args.rerank_experiment else
        'Attention-guided geometric correction, not weight training. Changed stems require an existing crossbar and non-worsening readback. OCR is not human-labeled legibility; visually inspect the paired sheets.'))
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
