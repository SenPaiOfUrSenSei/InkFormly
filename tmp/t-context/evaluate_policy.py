import argparse,json,time
from pathlib import Path
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.schema import Line
from inkform.recognition import recognition_image
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();out=Path('artifacts/evaluation/t-word-clarity')/f'seed-{a.seed}';out.mkdir(parents=True,exist_ok=True)
source=Path('artifacts/evaluation/t-stem-refinement/heldout')/f'seed-{a.seed}'/'samples.json';old=json.loads(source.read_text());g=ReadableGenerator();records=[]
for i,r in enumerate(old):
 path=out/f'line-{i+1}.json'
 if path.exists():record=json.loads(path.read_text())
 else:
  original=Line.model_validate(r['baseline']);start=time.perf_counter();line,_=g._select_t_context(original.text,.85,a.seed,g.default_reference,.5,(original,None))
  assert line.text==original.text
  if line.offsets!=original.offsets:
   assert line.readback.status=='match' and line.t_formation['strokes_modified'] is False
  record=dict(text=original.text,baseline=original.model_dump(),selected=line.model_dump(),changed=line.offsets!=original.offsets,extra_seconds=time.perf_counter()-start);path.write_text(json.dumps(record))
 records.append(record);print(json.dumps({k:v for k,v in record.items() if k not in ('baseline','selected')}),flush=True)
 sheet=Image.new('RGB',(1650,320),'white');draw=ImageDraw.Draw(sheet)
 for n,key in enumerate(('baseline','selected')):
  line=record[key];draw.text((10,n*160+4),f"{key}: {line['text']} | {line['readback']['recognized_text']}",fill='black');im=recognition_image(line['offsets']);im.thumbnail((1600,120));sheet.paste(im,(10,n*160+30))
 sheet.save(out/f'comparison-{i+1}.png')
summary=dict(total=len(records),changed=sum(r['changed'] for r in records),baseline_exact=sum(r['baseline']['readback']['status']=='match' for r in records),selected_exact=sum(r['selected']['readback']['status']=='match' for r in records),extra_seconds=sum(r['extra_seconds'] for r in records))
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
