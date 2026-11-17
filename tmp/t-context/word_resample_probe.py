from pathlib import Path
import json
from PIL import Image,ImageDraw
from inkform.readable_generator import ReadableGenerator
from inkform.schema import Line
from inkform.recognition import recognition_image
out=Path('tmp/t-context/word-resampling');out.mkdir(exist_ok=True);g=ReadableGenerator()
for i in (8,30,44):
 p=Path(f'output/pdf/college-essay-t-word-clarity/lines/line-{i:03d}.json');row=json.loads(p.read_text());old=Line.model_validate(row['line']);e=old.t_formation['selected_words']
 line,_=g._select_t_words(old.text,.85,old.seed,g.default_reference,.5,(old,None),cached_evidence=e)
 (out/f'line-{i:03d}.json').write_text(line.model_dump_json())
 print(i,line.t_formation['applied'],line.t_formation['word_placements'],flush=True)
 sheet=Image.new('RGB',(1700,340),'white');d=ImageDraw.Draw(sheet)
 for n,l in enumerate((old,line)):
  d.text((10,n*170+5),f"{'Original' if n==0 else 'Complete-word generation'}: {l.text}",fill='black');im=recognition_image(l.offsets);im.thumbnail((1660,125));sheet.paste(im,(10,n*170+35))
 sheet.save(out/f'comparison-{i}.png')
