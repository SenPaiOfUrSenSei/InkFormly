"""Select coherent, intact neural takes using word-local t evidence."""
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
SOURCE=ROOT/'output/pdf/college-essay-restored-context'
OUT=ROOT/'output/pdf/college-essay-t-context-words'
CACHE=ROOT/'output/pdf/college-essay-t-word-clarity/lines'


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value));temporary.replace(path)


def worker(indices):
    from inkform.schema import Document,Line
    from inkform.word_clarity import prefer_complete_take,lexical_word
    from inkform.readable_generator import ReadableGenerator
    doc=Document.model_validate_json((SOURCE/'document.json').read_text());g=ReadableGenerator(adapter_path=None)
    for index in indices:
        path=OUT/'lines'/f'line-{index+1:03d}.json'
        if path.exists():continue
        original=doc.lines[index]
        cached=json.loads((CACHE/f'line-{index+1:03d}.json').read_text())
        base=Line.model_validate(cached['line']);proof=base.t_formation
        if cached['changed'] and not prefer_complete_take(proof['baseline_words'],proof['selected_words']):
            base=original.model_copy(deep=True);proof['selected_words']=proof['baseline_words']
        evidence=proof['selected_words']
        if evidence and evidence['exact_words']==evidence['target_words'] and evidence['mean_nll']<.75:
            line=base
        else:
            line,_=g._select_t_words(original.text,doc.bias,original.seed,doc.reference,doc.form_diversity,(base,None),cached_evidence=evidence)
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
    from inkform.word_clarity import prefer_complete_take,lexical_word
    from inkform.render import layout_document,export_document
    original=Document.model_validate_json((SOURCE/'document.json').read_text());doc=original.model_copy(deep=True)
    changes=[]
    for path in sorted((OUT/'lines').glob('*.json')):
        row=json.loads(path.read_text());line=Line.model_validate(row['line']);index=row['index']
        formation=line.t_formation or {}
        # Validate cached observations using final word normalization rules.
        evidence=formation.get('selected_words')
        if evidence:
            for word in evidence['words']:
                word['exact']=word.get('boundary_verified',True) and lexical_word(word['recognized_text'])==lexical_word(word['word'])
            evidence['exact_words']=sum(w['exact'] for w in evidence['words'] if w['target'])
        placements=formation.get('word_placements',[])
        if evidence and any(not evidence['words'][p['word_index']]['exact'] for p in placements):
            prior=json.loads((CACHE/path.name).read_text())
            line=Line.model_validate(prior['line']) if prior['changed'] else original.lines[index].model_copy(deep=True)
            row.update(line=line.model_dump(),changed=line.offsets!=original.lines[index].offsets)
            save(path,row)
        doc.generation_seconds+=max(0.,line.seconds-doc.lines[index].seconds)
        doc.lines[index]=line
        if row['changed']:
            if line.t_formation['method']=='whole-line-word-clarity-v1':
                assert prefer_complete_take(line.t_formation['baseline_words'],line.t_formation['selected_words'])
            else:
                assert line.t_formation['individual_strokes_deformed'] is False
                assert all(p['outside_points_preserved'] and p['uniform_word_placement'] and p.get('ownership_verified') for p in line.t_formation['word_placements'])
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
    target=OUT.parent/'DraftCollegeEssay2-handwritten-contextual-t.pdf';shutil.copyfile(OUT/manifest['pdf'],target)
    save(OUT/'document.json',doc.model_dump());save(OUT/'settings.json',settings);save(OUT/'manifest.json',manifest)
    assert len(PdfReader(target).pages)==len(layout.pages)==7
    report=dict(pages=7,source_text_and_paragraphs_preserved=True,writer_reference_preserved=True,
        layout_settings_preserved=True,changed_lines=len(changes),t_words_in_selected_lines=sum(x['formation']['selected_words']['target_words'] for x in changes),
        regenerated_complete_words=sum(len(x['formation'].get('word_placements',[])) for x in changes),
        matches_before=sum(l.readback.status=='match' for l in original.lines if l.text),
        matches_after=sum(l.readback.status=='match' for l in doc.lines if l.text),
        changes=changes,visually_reviewed_pages=[],pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        limitations='Complete words are regenerated at verified blank gaps and pen lifts, with translation and uniform scaling only. Neighboring raw pen paths are preserved. Pretrained weights are unchanged; readback is not a human legibility guarantee.')
    save(OUT/'verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='changes'}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',nargs='+',type=int);parser.add_argument('--compose-only',action='store_true');args=parser.parse_args()
    if args.worker is not None:worker(args.worker);return
    OUT.mkdir(exist_ok=True);(OUT/'lines').mkdir(exist_ok=True)
    if args.compose_only:compose();return
    from inkform.schema import Document
    doc=Document.model_validate_json((SOURCE/'document.json').read_text())
    from inkform.word_clarity import internal_t_words
    indices=[i for i,l in enumerate(doc.lines) if internal_t_words(l.text)]
    processes=[]
    for slot in range(3):
        log=(OUT/f'worker-{slot+1}.log').open('a')
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
        p=subprocess.Popen([sys.executable,'-m','scripts.refine_essay_word_blocks','--worker',*map(str,indices[slot::3])],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
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
