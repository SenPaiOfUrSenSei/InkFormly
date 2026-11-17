"""Score already generated paired t probes, keeping OCR and model evidence distinct."""
import argparse
import json
from pathlib import Path

from inkform.readable_generator import ReadableGenerator
from inkform.t_formation import TFormationEvaluator


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--split',default='dev',choices=['dev','heldout'])
    args=parser.parse_args()
    root=Path('artifacts/evaluation/t-adapter')/args.split/f'seed-{args.seed}-strength-1.0'
    records=json.loads((root/'samples.json').read_text());path=root/'formation-scores.json'
    scores=json.loads(path.read_text()) if path.exists() else []
    g=ReadableGenerator(adapter_path=None);critic=TFormationEvaluator(g.model,g._primer)
    for row in records:
        if any(r['text']==row['text'] and r['strength']==row['strength'] for r in scores):continue
        if not row['complete']:continue
        result=critic.score(row['text'],row['offsets'],g.default_reference)
        record=dict(text=row['text'],strength=row['strength'],readback=row['readback'],formation=result)
        scores.append(record);path.write_text(json.dumps(scores,indent=2))
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
