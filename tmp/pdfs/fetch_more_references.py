import concurrent.futures,hashlib,json,urllib.request,urllib.error
from pathlib import Path
revision='61544f003893c99cbcb9013b33d0a437bbe1ac6f';folder=Path('models/vasquez/styles')
def fetch(pair):
 style,kind=pair;name=f'style-{style}-{kind}.npy';p=folder/name
 url=f'https://raw.githubusercontent.com/niccolodevries/svg-handwriting-generator/{revision}/synth/styles/{name}'
 try:
  if not p.exists():
   with urllib.request.urlopen(url,timeout=60) as r:data=r.read(2_000_000)
   if not data.startswith(b'\x93NUMPY'):raise RuntimeError(name)
   p.write_bytes(data)
  return {'file':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'url':url}
 except urllib.error.HTTPError as e:
  if e.code==404:return {'file':name,'unavailable':True}
  raise
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 records=list(pool.map(fetch,[(i,k) for i in (1,2,4,5,7,9,11,12) for k in ('chars','strokes')]))
Path('models/vasquez/additional-reference-provenance.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
