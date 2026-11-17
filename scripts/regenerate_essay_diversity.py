"""Regenerate the user's essay with fixed cursive A and learned form diversity.

Each new neural line is cached separately. Three independent CPU processes
share no model state; readback and same-reference candidate selection run in
each. Original PDFs and earlier trajectory documents are preserved.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/pdf/college-essay-reformed'
SOURCE = Path('/home/bifrost/Downloads/DraftCollegeEssay2.pdf')
DIVERSITY = .5


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False))
    temporary.replace(path)


def source_text():
    from pypdf import PdfReader
    text = (ROOT/'output/pdf/college-essay-cursive-A/source.txt').read_text()
    raw = '\n'.join(p.extract_text(extraction_mode='layout') for p in PdfReader(SOURCE).pages)
    converted = raw.translate(str.maketrans({'“':'"','”':'"','’':"'",'‘':"'",'—':' - ','–':' - '}))
    assert ' '.join(converted.split())==' '.join(text.split()),'Source words differ'
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'source.txt').write_text(text)
    return text


def worker(indices,repair=False):
    from inkform.generator import GenerationError, get_generator, plan_lines
    from inkform.schema import Document, Line
    text = (OUT/'source.txt').read_text()
    paragraphs = text.split('\n\n')
    generator = get_generator()
    generator.validate_text(text)
    cache = OUT/'lines'
    cache.mkdir(exist_ok=True)
    for index in indices:
        paragraph = paragraphs[index]
        planned = plan_lines(paragraph,38)
        lines = []
        for i,content in enumerate(planned):
            path = cache/f'p{index+1:02d}-l{i+1:02d}.json'
            if path.exists():
                record = json.loads(path.read_text())
                assert record['requested_form_diversity']==DIVERSITY
                line = Line.model_validate(record['line'])
                assert line.text==content
                if repair and line.readback.edit_distance>=2 and not record.get('repair_completed'):
                    original = line
                    try:
                        candidate,_ = generator.select_line(content,.85,(line.seed+104729)%(2**31-1),
                            generator.default_reference,True,DIVERSITY)
                        rank = lambda l:(not l.readback.numbers_match,l.readback.edit_distance)
                        if rank(candidate)<rank(line):line=candidate
                        record['additional_readback']=candidate.readback.model_dump()
                        line.seconds=original.seconds+candidate.seconds
                    except GenerationError as exc:
                        record['additional_error']=str(exc)
                    record['repair_completed']=True
                    record['original_line']=original.model_dump()
                    record['line']=line.model_dump()
                    write_json(path,record)
            else:
                seed = (817+index*100003+i*1009)%(2**31-1)
                for retry in range(3):
                    try:
                        line,_ = generator.select_line(content,.85,(seed+retry*104729)%(2**31-1),
                            generator.default_reference,True,DIVERSITY)
                        break
                    except GenerationError:
                        if retry==2:raise
                write_json(path,dict(requested_form_diversity=DIVERSITY,line=line.model_dump()))
            lines.append(line)
            print(f'Paragraph {index+1:02d}: {i+1}/{len(planned)} | {line.readback.status} | diversity {line.sampling_form_diversity}',flush=True)
        document = Document(backend=generator.backend,source_text='\n'.join(planned),seed=817+index*100003,
            bias=.85,form_diversity=DIVERSITY,lines=lines,reference=generator.default_reference,
            generation_seconds=sum(l.seconds for l in lines),quality_checks=True,consistent_style=True)
        document.warnings = generator.quality_warnings(document)
        write_json(OUT/f'paragraph-{index+1:02d}.json',document.model_dump())


def compose():
    from inkform.generator import GravesGenerator
    from inkform.schema import Document,Line,PageSettings,InkSettings
    from inkform.render import export_document
    text = (OUT/'source.txt').read_text()
    paragraphs = []
    for i in range(len(text.split('\n\n'))):
        name = f'paragraph-{i+1:02d}.json'
        path = OUT/'overrides'/name
        if not path.exists():path = OUT/name
        paragraphs.append(Document.model_validate_json(path.read_text()))
    # Retain the existing relaxed formatting and ink so the generation change
    # can be judged against the previous essay.
    settings = json.loads((ROOT/'output/pdf/college-essay-cursive-A/settings.json').read_text())
    page,ink = PageSettings(**settings['page']),InkSettings(**settings['ink'])
    rows = int((297-2*page.margin_mm)//page.line_spacing_mm)
    lines = []
    def blank():return Line(text='',offsets=[],seed=817,seconds=0,attempts=0,primed=False,attention_complete=True)
    for paragraph in paragraphs:
        assert paragraph.form_diversity==DIVERSITY and paragraph.reference==paragraphs[0].reference
        if lines and len(lines)%rows:lines.append(blank())
        remaining = rows-len(lines)%rows
        count = len(paragraph.lines)
        padding = 0
        if (count<=3 and count>remaining) or (remaining==1 and count>1):padding=remaining
        elif count>remaining and (count-remaining)%rows==1:padding=1
        lines.extend(blank() for _ in range(padding))
        lines.extend(paragraph.lines)
    assert re.sub(r'\s+',' ',text).strip()==' '.join(l.text for l in lines if l.text)
    document = Document(backend=paragraphs[0].backend,source_text=text,seed=817,bias=.85,
        form_diversity=DIVERSITY,lines=lines,reference=paragraphs[0].reference,
        generation_seconds=sum(p.generation_seconds for p in paragraphs),quality_checks=True,consistent_style=True)
    document.warnings = GravesGenerator.quality_warnings(document)
    write_json(OUT/'document.json',document.model_dump())
    write_json(OUT/'settings.json',settings)
    (OUT/'reference.json').write_text(document.reference.model_dump_json())
    report = dict(source=str(SOURCE),source_words=len(text.split()),paragraphs=len(paragraphs),
        form_diversity=DIVERSITY,written_lines=sum(bool(l.text) for l in lines),
        readback_matches=sum(l.readback.status=='match' for l in lines if l.text),
        review=[dict(line=i+1,text=l.text,readback=l.readback.model_dump())
                for i,l in enumerate(lines) if l.text and l.readback.status!='match'],
        retained_diversities={str(d):sum(l.sampling_form_diversity==d for l in lines if l.text)
                              for d in sorted({l.sampling_form_diversity for l in lines if l.text})},
        note='All lines newly synthesized with the updated sampler and fixed reference A. OCR is not human-labeled accuracy.')
    write_json(OUT/'readback.json',report)
    manifest = export_document(document,page,ink,OUT,300)
    write_json(OUT/'manifest.json',manifest)
    target = OUT.parent/'DraftCollegeEssay2-handwritten-reformed.pdf'
    shutil.copyfile(OUT/manifest['pdf'],target)
    print(json.dumps(dict(pdf=str(target),manifest=manifest,report=report),indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,nargs='+')
    parser.add_argument('--compose-only',action='store_true')
    parser.add_argument('--repair',action='store_true',help='Try additional takes for readback differences of at least two characters')
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker,args.repair)
        return
    if args.compose_only:
        compose()
        return
    text = source_text()
    from inkform.generator import plan_lines
    paragraphs = text.split('\n\n')
    # Balance workers by line count, placing the longest paragraphs first.
    assignments = [[],[],[]]
    loads = [0,0,0]
    for index in sorted(range(len(paragraphs)),key=lambda i:len(plan_lines(paragraphs[i],38)),reverse=True):
        slot = min(range(3),key=lambda i:loads[i])
        assignments[slot].append(index)
        loads[slot] += len(plan_lines(paragraphs[index],38))
    print(json.dumps(dict(paragraphs=len(paragraphs),lines=sum(loads),worker_lines=loads)),flush=True)
    processes = []
    for i,indices in enumerate(assignments):
        log = (OUT/f'worker-{i+1}.log').open('a')
        env = {**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
        process = subprocess.Popen([sys.executable,'-m','scripts.regenerate_essay_diversity',
            *(['--repair'] if args.repair else []),'--worker',*map(str,indices)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        processes.append((process,log))
    last = -1
    while any(p.poll() is None for p,_ in processes):
        done = len(list((OUT/'lines').glob('*.json')))
        if done!=last:
            print(f'Completed neural lines: {done}/{sum(loads)}',flush=True)
            last = done
        time.sleep(2)
    for p,log in processes:
        log.close()
        if p.returncode:raise RuntimeError('A generation worker failed; inspect worker logs. Cached lines can be resumed.')
    compose()


if __name__=='__main__':
    main()
