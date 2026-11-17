import hashlib,json,urllib.request
from pathlib import Path
revision='61544f003893c99cbcb9013b33d0a437bbe1ac6f'
folder=Path('models/vasquez/styles');records=[]
for style in (6,8):
 for kind in ('chars','strokes'):
  name=f'style-{style}-{kind}.npy';path=folder/name
  url=f'https://raw.githubusercontent.com/niccolodevries/svg-handwriting-generator/{revision}/synth/styles/{name}'
  if not path.exists():
   with urllib.request.urlopen(url,timeout=60) as response: data=response.read(2_000_000)
   if not data.startswith(b'\x93NUMPY'):raise RuntimeError(f'Invalid array: {name}')
   path.write_bytes(data)
  records.append({'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'url':url})
Path('models/vasquez/cursive-reference-provenance.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
