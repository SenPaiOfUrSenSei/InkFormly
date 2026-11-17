"""Keep reviewed cursive A, retake uncertain lines, and render calmer variations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'output/pdf/college-essay-reformed'
OUT = ROOT/'output/pdf/college-essay-coherent'


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value));temporary.replace(path)


def worker(indices):
    from inkform.generator import get_generator,GenerationError
    from inkform.schema import Document
    doc=Document.model_validate_json((BASE/'document.json').read_text())
    generator=get_generator()
    assert generator.default_reference==doc.reference
    for index in indices:
        path=OUT/'retakes'/f'line-{index+1:03d}.json'
        if path.exists():continue
        old=doc.lines[index]
        try:
            candidate,_=generator.select_line(old.text,1.05,(old.seed+370003)%(2**31-1),
                doc.reference,True,.5)
            rank=lambda line:(not line.readback.numbers_match,line.readback.edit_distance)
            accepted=rank(candidate)<=rank(old)
            record=dict(index=index,original=old.model_dump(),candidate=candidate.model_dump(),accepted=accepted)
        except GenerationError as exc:
            record=dict(index=index,original=old.model_dump(),accepted=False,error=str(exc))
        save(path,record)
        print(f'Line {index+1}: retained {"new" if record["accepted"] else "previous"} take',flush=True)


def compose():
    from inkform.generator import GravesGenerator
    from inkform.schema import Document,Line,PageSettings,InkSettings
    from inkform.variation_presets import VARIATION_PRESETS
    from inkform.render import export_document,layout_document
    doc=Document.model_validate_json((BASE/'document.json').read_text())
    changes=[]
    for path in sorted((OUT/'retakes').glob('*.json')):
        record=json.loads(path.read_text())
        if record['accepted']:
            index=record['index'];line=Line.model_validate(record['candidate'])
            assert line.text==doc.lines[index].text
            changes.append(dict(line=index+1,old=doc.lines[index].readback.model_dump(),new=line.readback.model_dump()))
            doc.lines[index]=line
        doc.generation_seconds+=record.get('candidate',{}).get('seconds',0)
    assert ' '.join(doc.source_text.split())==' '.join(l.text for l in doc.lines if l.text)
    doc.warnings=GravesGenerator.quality_warnings(doc)
    old=json.loads((BASE/'settings.json').read_text());preset=VARIATION_PRESETS['readable']
    page=PageSettings(**{**old['page'],**preset['page']})
    ink=InkSettings(**{**old['ink'],**preset['ink'],'width_mm':.28})
    save(OUT/'document.json',doc.model_dump())
    save(OUT/'settings.json',dict(page=page.model_dump(),ink=ink.model_dump()))
    report=dict(preset='readable',fixed_reference='cursive A / writer 2',changed_lines=changes,
        written_lines=sum(bool(l.text) for l in doc.lines),
        matches_before=sum(l.readback.status=='match' for l in Document.model_validate_json((BASE/'document.json').read_text()).lines),
        matches_after=sum(l.readback.status=='match' for l in doc.lines),
        review=[dict(line=i+1,text=l.text,readback=l.readback.model_dump())
            for i,l in enumerate(doc.lines) if l.text and l.readback.status!='match'],
        limitations='Readback checks base trajectories, not the final rendering. Minimum gaps apply to detected blank intervals before fitting. No human-likeness validation or weight retraining.')
    save(OUT/'readback.json',report)
    layout=layout_document(doc,page)
    import numpy as np
    for rows in layout.pages:
        for line in rows:
            if line.strokes:
                points=np.concatenate(line.strokes)
                assert np.isfinite(points).all()
                assert points[:,0].min()>=page.margin_mm-.01 and points[:,0].max()<=210-page.margin_mm+.01
                assert points[:,1].min()>=page.margin_mm-.01 and points[:,1].max()<=297-page.margin_mm+.01
    manifest=export_document(doc,page,ink,OUT,300)
    save(OUT/'manifest.json',manifest)
    target=OUT.parent/'DraftCollegeEssay2-handwritten-coherent.pdf'
    shutil.copyfile(OUT/manifest['pdf'],target)
    print(json.dumps(dict(pdf=str(target),pages=manifest['pages'],matches_before=report['matches_before'],
        matches_after=report['matches_after'],retained_new_lines=len(changes))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',nargs='+',type=int)
    parser.add_argument('--compose-only',action='store_true');args=parser.parse_args()
    if args.worker is not None:worker(args.worker);return
    OUT.mkdir(exist_ok=True);(OUT/'retakes').mkdir(exist_ok=True)
    if args.compose_only:compose();return
    from inkform.schema import Document
    doc=Document.model_validate_json((BASE/'document.json').read_text())
    indices=[i for i,l in enumerate(doc.lines) if l.text and l.readback.status!='match']
    processes=[]
    for slot in range(3):
        log=(OUT/f'worker-{slot+1}.log').open('a')
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
        p=subprocess.Popen([sys.executable,'-m','scripts.improve_essay_readability','--worker',*map(str,indices[slot::3])],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        processes.append((p,log))
    print(f'Checking {len(indices)} uncertain lines; keeping {sum(bool(l.text) for l in doc.lines)-len(indices)} matching lines.',flush=True)
    last=-1
    while any(p.poll() is None for p,_ in processes):
        done=len(list((OUT/'retakes').glob('*.json')))
        if done!=last:print(f'Checked {done}/{len(indices)} uncertain lines',flush=True);last=done
        time.sleep(2)
    for p,log in processes:
        log.close()
        if p.returncode:raise RuntimeError('A worker failed; inspect worker logs before resuming.')
    compose()


if __name__=='__main__':main()
