from pathlib import Path
from inkform.schema import Document
from inkform.readable_generator import ReadableGenerator
from inkform.cursive_selection import select_cursive_words

folder=Path('tmp/joined-r')
baseline=Document.model_validate_json((folder/'before.json').read_text())
updated=Document.model_validate_json((folder/'after.json').read_text())
line=baseline.lines[2]
generator=ReadableGenerator()
result,_=select_cursive_words(generator,(line,None),baseline.reference,.85,.5,baseline.seed+2018)
updated.lines[2]=result
print(line.text, result.r_selection, flush=True)
(folder/'after.json').write_text(updated.model_dump_json())
