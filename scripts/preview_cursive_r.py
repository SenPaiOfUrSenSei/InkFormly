"""Render the historical, user-rejected r experiment for comparison only."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from inkform.render import strokes_from_offsets,_adjust_slant,PlacedLine,render_page
from inkform.schema import PageSettings,InkSettings

def prepared(offsets):
 strokes=strokes_from_offsets(offsets);points=np.concatenate(strokes)
 centered=points[:,0]-points[:,0].mean();variance=float(centered@centered)
 if variance:
  slope=float(centered@(points[:,1]-points[:,1].mean())/variance)
  for stroke in strokes:stroke[:,1]-=.9*slope*(stroke[:,0]-points[0,0])
 return _adjust_slant(strokes,-18)

def main():
 out=Path('output/pdf/cursive-r-preview');out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for path in sorted(Path('artifacts/evaluation/cursive-r').glob('*.json')):
  r=json.loads(path.read_text())
  if 'before' in r and r.get('changed'):rows.append(r)
 assert rows
 rows=rows[:2]
 contexts=[json.loads(path.read_text()) for path in sorted(Path('artifacts/evaluation/cursive-r-context-final').glob('line-*.json'))][:2]
 page=PageSettings(margin_mm=18,writing_height_mm=8.8,line_spacing_mm=13.5,paper='plain',naturalness=0,letter_variation=0)
 ink=InkSettings(width_mm=.28,roughness=.3,opacity=.98)
 lines=[]
 for i,r in enumerate(rows):
  variants=[prepared(r[key]['offsets']) for key in ('before','after')]
  bounds=[(np.concatenate(v).min(0),np.concatenate(v).max(0)) for v in variants]
  scale=min(8.8/max(float(hi[1]-lo[1]) for lo,hi in bounds),70/max(float(hi[0]-lo[0]) for lo,hi in bounds))
  for col,strokes in enumerate(variants):
   low,high=bounds[col];origin=np.array([23+89*col,79+43*i])
   placed=[(s-low)*scale+origin for s in strokes]
   lines.append(PlacedLine(r['text'],placed,origin[1]+8.8))
 for i,r in enumerate(contexts):
  variants=[prepared(r[key]['offsets']) for key in ('before','after')]
  bounds=[(np.concatenate(v).min(0),np.concatenate(v).max(0)) for v in variants]
  robust=max(float(np.diff(np.quantile(np.concatenate(v)[:,1],[.01,.99]))[0]) for v in variants)
  scale=min(9/max(robust,.01),12/max(float(hi[1]-lo[1]) for lo,hi in bounds),168/max(float(hi[0]-lo[0]) for lo,hi in bounds))
  for n,strokes in enumerate(variants):
   low,high=bounds[n];origin=np.array([23,181+44*i+18*n])
   lines.append(PlacedLine(r['text'],[(s-low)*scale+origin for s in strokes],origin[1]+7.2))
 image=render_page(lines,page,ink,300,0);image.save(out/'writing.png')
 target=out.parent/'InkForm-reference-cursive-r-preview.pdf';c=canvas.Canvas(str(target),pagesize=(210*mm,297*mm))
 c.drawImage(ImageReader(image),0,0,width=210*mm,height=297*mm)
 c.setFillColorRGB(.12,.2,.3);c.setFont('Helvetica-Bold',17);c.drawString(20*mm,274*mm,'Rejected r experiment')
 c.setFont('Helvetica',10);c.drawString(20*mm,265*mm,'Historical comparison: withdrawn after user review.')
 c.setFont('Helvetica',9);c.drawString(20*mm,255*mm,'OCR passed these edits, but words and joins were not visually readable.')
 c.setFont('Helvetica-Bold',10);c.drawString(23*mm,231*mm,'Current form');c.drawString(112*mm,231*mm,'Reference cursive r')
 for i,r in enumerate(rows):
  c.setFont('Helvetica',9);c.setFillColorRGB(.4,.45,.5);c.drawString(23*mm,(297-(72+43*i))*mm,f"{r['text']} - take {r['seed']}")
 for i,r in enumerate(contexts):
  c.setFont('Helvetica',8);c.setFillColorRGB(.4,.45,.5)
  c.drawString(23*mm,(297-(177+44*i))*mm,'In context - current form')
  c.drawString(23*mm,(297-(195+44*i))*mm,'In context - reference cursive r')
 c.setFillColorRGB(.4,.45,.5);c.setFont('Helvetica',8);c.drawString(20*mm,21*mm,'Rejected experiment. Production uses native neural letter forms.')
 c.showPage();c.save();image.close()
 report=dict(status='rejected by user: words containing cursive r are not understandable',production_enabled=False,pdf=str(target),pages=1,accepted_pairs=len(rows),context_pairs=len(contexts),words=[r['text'] for r in rows],visually_reviewed=False,pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
 (out/'verification.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':main()
