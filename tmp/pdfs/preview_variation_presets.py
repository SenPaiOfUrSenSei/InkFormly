import json
from pathlib import Path
from PIL import Image,ImageDraw
from inkform.schema import Document,PageSettings,InkSettings
from inkform.variation_presets import VARIATION_PRESETS
from inkform.render import layout_document,render_page
out=Path('output/pdf/cursive-variation-options');out.mkdir(parents=True,exist_ok=True)
d=Document.model_validate_json(Path('output/pdf/cursive-options/style-2.json').read_text());rows=[]
for name,preset in VARIATION_PRESETS.items():
 page=PageSettings(**{**preset['page'],'margin_mm':18,'writing_height_mm':8,'line_spacing_mm':12,'layout_seed':29})
 ink=InkSettings(**{**preset['ink'],'color':'#203f78','width_mm':.3,'opacity':.98,'roughness':.3})
 lines=layout_document(d,page).pages[0];im=render_page(lines,page,ink,150)
 px=150/25.4;crop=im.crop((round(17*px),round(17*px),round(194*px),round((18+12*3+1)*px)))
 row=Image.new('RGB',(1100,crop.height+35),'white');ImageDraw.Draw(row).text((12,6),preset['label'],fill='black');row.paste(crop,(12,30));row.save(out/f'{name}.png');rows.append(row)
sheet=Image.new('RGB',(1100,sum(r.height for r in rows)),'white');y=0
for row in rows:sheet.paste(row,(0,y));y+=row.height
sheet.save(out/'presets.png');print(out/'presets.png')
