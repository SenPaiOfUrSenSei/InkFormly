"""Historical upstream-port probe using explicitly supplied audited source.

The app uses InkForm's own implementation. This optional comparison requires
model.py and drawing.py from the conversion revision in the asset provenance.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from inkform.recognition import get_recognizer, recognition_image, compare_readback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/formation"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-source", type=Path, required=True)
parser.add_argument("--drawing-source", type=Path, required=True)
parser.add_argument("--styles", type=int, nargs="+", default=[3])
args = parser.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("vasquez_reference", args.model_source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tree = ast.parse(args.drawing_source.read_text())
alphabet = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                and any(isinstance(target,ast.Name) and target.id=="alphabet" for target in node.targets))
torch.set_num_threads(1)
model = module.HandwritingSynthesisModel(vocab_size=len(alphabet))
model.load_state_dict(torch.load(ROOT/"models/vasquez/model.pt",map_location="cpu",weights_only=True))
model.eval()
reader = get_recognizer()
cases = ["student here,", "Hey Eric! Former SAT masterclass student here,", "on the sat :)",
         "Every student studies science."]
records, pictures = [], {}
for style in args.styles:
    prime = np.load(ROOT/f"models/vasquez/styles/style-{style}-strokes.npy",allow_pickle=False)
    primetext = np.load(ROOT/f"models/vasquez/styles/style-{style}-chars.npy",allow_pickle=False).tobytes().decode("utf-8")
    print("Style",style,repr(primetext),flush=True)
    for index,text in enumerate(cases):
        encoded = [alphabet.index(c) for c in primetext+" "+text]+[0]
        start=time.perf_counter()
        with torch.random.fork_rng(devices=[]),torch.inference_mode():
            torch.manual_seed(817)
            sample = model.sample(torch.tensor([encoded]),torch.tensor([len(encoded)]),torch.tensor([.75]),
                                  60*len(text),True,torch.tensor(prime).float().unsqueeze(0),torch.tensor([len(prime)]))[0]
        sample[:,1] *= -1
        observed=reader.read(sample.tolist())
        record={"style":style,"text":text,"seed":817,"seconds":time.perf_counter()-start,
                "offsets":sample.tolist(),"points":len(sample),"readback":compare_readback(text,observed).model_dump()}
        records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!="offsets"}),flush=True)
        preview=recognition_image(sample.tolist())
        row=Image.new("RGB",(1600,190),"white")
        ImageDraw.Draw(row).text((12,5),f"Vasquez / style {style} / {observed}",fill="black")
        preview.thumbnail((1550,140))
        row.paste(preview,(14,35))
        pictures[(style,index)]=row
        (OUT/"vasquez-probe.json").write_text(json.dumps(records))
        if style==args.styles[-1]:
            sheet=Image.new("RGB",(1600,190*len(args.styles)),"white")
            for j,s in enumerate(args.styles):
                sheet.paste(pictures[(s,index)],(0,j*190))
            sheet.save(OUT/f"vasquez-case{index}.png")
