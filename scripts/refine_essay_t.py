"""Reproduce the experimental stem-adjusted essay, superseded after user review."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'output/pdf/college-essay-larger-relaxed'
OUT=ROOT/'output/pdf/college-essay-clear-t'


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value));temporary.replace(path)


def worker(indices):
    from inkform.schema import Document
    from inkform.readable_generator import ReadableGenerator
    doc=Document.model_validate_json((SOURCE/'document.json').read_text());g=ReadableGenerator(adapter_path=None)
    for index in indices:
        path=OUT/'lines'/f'line-{index+1:03d}.json'
        if path.exists():continue
        original=doc.lines[index]
        line,_=g._refine_t_line(original.text,doc.bias,original.seed,doc.reference,doc.form_diversity,(original,None))
        rank=lambda l:(not l.readback.numbers_match,l.readback.edit_distance)
        assert line.text==original.text and rank(line)<=rank(original)
        changed=line.offsets!=original.offsets
        save(path,dict(index=index,line=line.model_dump(),changed=changed,
            before_readback=original.readback.model_dump()))
        print(f'Line {index+1}: {"refined" if changed else "retained"}',flush=True)


def compose():
    import numpy as np
    from pypdf import PdfReader
    from inkform.schema import Document,Line,PageSettings,InkSettings
    from inkform.generator import GravesGenerator
    from inkform.render import layout_document,export_document
    original=Document.model_validate_json((SOURCE/'document.json').read_text());doc=original.model_copy(deep=True)
    changes=[]
    for path in sorted((OUT/'lines').glob('*.json')):
        row=json.loads(path.read_text());line=Line.model_validate(row['line']);index=row['index']
        doc.generation_seconds+=max(0.,line.seconds-doc.lines[index].seconds)
        doc.lines[index]=line
        if row['changed']:changes.append(dict(line=index+1,text=line.text,formation=line.t_formation,
            before_readback=row['before_readback'],after_readback=line.readback.model_dump()))
    doc.t_refinement=True;doc.warnings=GravesGenerator.quality_warnings(doc)
    assert [l.text for l in doc.lines]==[l.text for l in original.lines]
    assert doc.source_text==original.source_text and doc.reference==original.reference
    settings=json.loads((SOURCE/'settings.json').read_text());page=PageSettings(**settings['page']);ink=InkSettings(**settings['ink'])
    layout=layout_document(doc,page)
    for rows in layout.pages:
        for row,line in enumerate(rows):
            if not line.strokes:continue
            a=np.concatenate(line.strokes)
            assert a[:,0].min()>=page.margin_mm-.01 and a[:,0].max()<=210-page.margin_mm+.01
            assert a[:,1].min()>=page.margin_mm+row*page.line_spacing_mm-.01
            assert a[:,1].max()<=page.margin_mm+(row+1)*page.line_spacing_mm+.01
    manifest=export_document(doc,page,ink,OUT,300)
    target=OUT.parent/'DraftCollegeEssay2-handwritten-clear-t.pdf';shutil.copyfile(OUT/manifest['pdf'],target)
    save(OUT/'document.json',doc.model_dump());save(OUT/'settings.json',settings);save(OUT/'manifest.json',manifest)
    assert len(PdfReader(target).pages)==len(layout.pages)==7
    report=dict(pages=7,source_text_and_paragraphs_preserved=True,writer_reference_preserved=True,
        layout_settings_preserved=True,changed_lines=len(changes),changed_t_stems=sum(len(x['formation']['adjustments']) for x in changes),
        matches_before=sum(l.readback.status=='match' for l in original.lines if l.text),
        matches_after=sum(l.readback.status=='match' for l in doc.lines if l.text),
        changes=changes,visually_reviewed_pages=[],pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        limitations='This is an attention-guided stroke-decoder correction; pretrained network weights are unchanged. Crossbars are never inserted. Independent readback cannot guarantee letter legibility.')
    save(OUT/'verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='changes'}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',nargs='+',type=int);parser.add_argument('--compose-only',action='store_true');args=parser.parse_args()
    if args.worker is not None:worker(args.worker);return
    OUT.mkdir(exist_ok=True);(OUT/'lines').mkdir(exist_ok=True)
    if args.compose_only:compose();return
    from inkform.schema import Document
    doc=Document.model_validate_json((SOURCE/'document.json').read_text())
    indices=[i for i,l in enumerate(doc.lines) if any(c=='t' and j and l.text[j-1].isalpha() for j,c in enumerate(l.text))]
    processes=[]
    for slot in range(3):
        log=(OUT/f'worker-{slot+1}.log').open('a')
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
        p=subprocess.Popen([sys.executable,'-m','scripts.refine_essay_t','--worker',*map(str,indices[slot::3])],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        processes.append((p,log))
    last=-1
    while any(p.poll() is None for p,_ in processes):
        done=len(list((OUT/'lines').glob('*.json')))
        if done!=last:print(f'Checked {done}/{len(indices)} lines with internal t',flush=True);last=done
        time.sleep(2)
    for p,log in processes:
        log.close()
        if p.returncode:raise RuntimeError('A decoder worker failed; inspect worker logs before resuming.')
    compose()


if __name__=='__main__':main()
