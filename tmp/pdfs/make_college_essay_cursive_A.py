"""Generate the user's full essay with the production neural handwriting pipeline."""
import json
import numpy as np
from pathlib import Path
import re
import shutil

from pypdf import PdfReader
from inkform.generator import get_generator, plan_lines
from inkform.schema import Document, GenerationSettings, InkSettings, Line, PageSettings, StyleReference
from inkform.render import export_document

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/pdf/college-essay-cursive-A'
SOURCE = Path('/home/bifrost/Downloads/DraftCollegeEssay2.pdf')
STARTS = [
 'My first hero', 'On our old', 'My first chance', 'I thought they',
 'One afternoon', 'It was trash', 'The next morning', 'I cried for',
 'Why bother,', 'Then the power', 'For a while,', 'For years,',
 'Years later,', 'The old question', 'I still want to fix',
 'I think I understand', 'I still want to be a hero.',
]


def prepare():
    raw = '\n'.join(page.extract_text(extraction_mode='layout') for page in PdfReader(SOURCE).pages)
    pieces, current = [], []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(line.startswith(start) for start in STARTS):
            if current:
                pieces.append(' '.join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        pieces.append(' '.join(current))
    assert len(pieces) == len(STARTS), len(pieces)
    assert ' '.join(raw.split()) == ' '.join(' '.join(pieces).split()), 'Extraction changed words'
    conversion = str.maketrans({'“':'"', '”':'"', '’':"'", '‘':"'", '—':' - ', '–':' - '})
    paragraphs = [piece.translate(conversion) for piece in pieces]
    text = '\n\n'.join(paragraphs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'source-extracted.txt').write_text(raw)
    (OUT/'source.txt').write_text(text)
    print(json.dumps({'characters':len(text),'words':len(raw.split()),'paragraphs':len(paragraphs),
                      'planned_lines':sum(len(plan_lines(p,48)) for p in paragraphs)+len(paragraphs)-1}),flush=True)
    return paragraphs, text


def main():
    paragraphs, text = prepare()
    generator = get_generator()
    model_dir = ROOT / 'models/vasquez/styles'
    generator.default_reference = StyleReference(
        text=np.load(model_dir/'style-2-chars.npy',allow_pickle=False).tobytes().decode(),
        normalized_offsets=np.load(model_dir/'style-2-strokes.npy',allow_pickle=False).tolist(),
        codec='vasquez-native-v1')
    generator.backend = 'vasquez-17900-style2'
    (OUT/'reference.json').write_text(generator.default_reference.model_dump_json())
    generator.validate_text(text)
    page = PageSettings(margin_mm=18, writing_height_mm=8., line_spacing_mm=12., paper='ruled',naturalness=.45,letter_variation=.25)
    ink = InkSettings(color='#203f78', width_mm=.28, opacity=.98, roughness=.35)
    documents = []
    for index, paragraph in enumerate(paragraphs):
        generation_text = paragraph if index==0 else '\n'.join(plan_lines(paragraph,38))
        cached = OUT/f'paragraph-{index+1:02d}.json'
        if cached.exists():
            document = Document.model_validate_json(cached.read_text())
            assert document.source_text == generation_text and document.backend == generator.backend
            print(f'Paragraph {index+1}/{len(paragraphs)}: reused checked trajectories',flush=True)
        else:
            settings = GenerationSettings(text=generation_text, seed=817+index*100003, bias=.85, ink=ink, page=page)
            document = generator.generate(settings, lambda i,n,line: print(
                f'Paragraph {index+1}/{len(paragraphs)} | line {i+1}/{n}: {line}',flush=True))
            cached.write_text(document.model_dump_json())
            print(f'Paragraph {index+1} done: '+json.dumps([
                {'text':line.text,'readback':line.readback.model_dump(),'attempts':line.attempts}
                for line in document.lines if line.readback.status!='match']),flush=True)
        documents.append(document)
    lines=[]
    for index,document in enumerate(documents):
        if index:
            lines.append(Line(text='',offsets=[],seed=42,seconds=0,attempts=0,primed=False,attention_complete=True))
        lines.extend(document.lines)
    combined = Document(backend=generator.backend, source_text=text, seed=817,bias=.85,
                        lines=lines,reference=generator.default_reference,
                        generation_seconds=sum(d.generation_seconds for d in documents),
                        quality_checks=True,consistent_style=True)
    assert re.sub(r'\s+',' ',text).strip() == ' '.join(line.text for line in lines if line.text)
    combined.warnings=generator.quality_warnings(combined)
    (OUT/'document.json').write_text(combined.model_dump_json())
    (OUT/'settings.json').write_text(json.dumps({'page':page.model_dump(),'ink':ink.model_dump()},indent=2))
    report = {'source':str(SOURCE),'backend':combined.backend,'written_lines':sum(bool(l.text) for l in lines),
              'matches':sum(l.readback.status=='match' for l in lines if l.text),
              'review':[{'line':i+1,'text':l.text,'readback':l.readback.model_dump()} for i,l in enumerate(lines) if l.text and l.readback.status!='match'],
              'seconds':combined.generation_seconds,'warnings':combined.warnings}
    (OUT/'readback.json').write_text(json.dumps(report,indent=2))
    manifest=export_document(combined,page,ink,OUT,300)
    target=OUT.parent/'DraftCollegeEssay2-handwritten-cursive-A.pdf'
    shutil.copyfile(OUT/manifest['pdf'],target)
    print(json.dumps({'pdf':str(target),'manifest':manifest,'review':report},indent=2),flush=True)


if __name__=='__main__':
    main()
