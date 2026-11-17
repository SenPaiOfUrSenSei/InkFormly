import json
from pathlib import Path
import numpy as np
import torch
from inkform.schema import Document,PageSettings
from inkform.render import layout_document
from inkform.recognition import get_recognizer,compare_readback

torch.set_num_threads(1)
folder=Path('output/pdf/college-essay-varied')
d=Document.model_validate_json((folder/'document.json').read_text());s=json.loads((folder/'settings.json').read_text());p=PageSettings.model_validate(s['page'])
versions={name:[line for page in layout_document(d,settings).pages for line in page] for name,settings in [('before',p.model_copy(update={'letter_variation':0})),('after',p)]}
reader=get_recognizer();report=[]
for index in (0,14,24,34,73,95):
 result={'line':index+1,'source':d.lines[index].text}
 for name,lines in versions.items():
  previous=np.zeros(2);offsets=[]
  for stroke in lines[index].strokes:
   for i,point in enumerate(stroke):
    delta=point-previous;offsets.append([float(delta[0]),float(delta[1]),int(i==len(stroke)-1)]);previous=point
  result[name]=compare_readback(d.lines[index].text,reader.read(offsets)).model_dump()
 report.append(result);print(json.dumps(result),flush=True)
(folder/'rendered-readback-check.json').write_text(json.dumps(report,indent=2))
