from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image

out=Path('output/pdf');samples=out/'cursive-options'
mm=72/25.4
pdf=out/'InkForm-cursive-style-options.pdf'
c=canvas.Canvas(str(pdf),pagesize=(210*mm,297*mm))
c.setTitle('InkForm - cursive handwriting options')
c.setAuthor('InkForm')
c.setFillColorRGB(.12,.18,.25);c.setFont('Helvetica-Bold',20)
c.drawString(20*mm,274*mm,'Two cursive directions')
c.setFont('Helvetica',10);c.setFillColorRGB(.4,.44,.48)
c.drawString(20*mm,265*mm,'The same passage from your essay, with two handwriting references.')
for label,title,description,top in [
 ('A','Connected cursive','Continuous joins and a stronger slant.',240),
 ('B','Casual joined handwriting','Looser connections and a more upright rhythm.',153)]:
 c.setFillColorRGB(.12,.18,.25);c.setFont('Helvetica-Bold',12)
 c.drawString(20*mm,top*mm,f'{label}  /  {title}')
 c.setFont('Helvetica',9);c.setFillColorRGB(.4,.44,.48)
 c.drawString(20*mm,(top-7)*mm,description)
 with Image.open(samples/f'style-{label}.png') as im:
  width=174*mm;height=width*im.height/im.width
  c.drawImage(ImageReader(im),18*mm,(top-14)*mm-height,width=width,height=height)
c.setStrokeColorRGB(.84,.88,.91);c.line(20*mm,48*mm,190*mm,48*mm)
c.setFillColorRGB(.4,.44,.48);c.setFont('Helvetica',9)
c.drawString(20*mm,39*mm,'Choose a direction for the full essay. Letter deformation is off in these samples.')
c.drawString(20*mm,33*mm,'These are neural stroke samples; final pages still need proofreading.')
c.save();print(pdf.resolve())
