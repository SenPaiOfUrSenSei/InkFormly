"""Run the upstream sampler before committing to an application architecture."""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import time
from xml.sax.saxutils import escape

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("graves_models", ROOT / "vendor/graves/models.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/graves/Epoch_56")
    parser.add_argument("--label", default="epoch56")
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = Path(args.checkpoint)
    meta = json.loads((folder / "meta.json").read_text())
    started = time.perf_counter()
    model = module.SynthesisNetwork.get_default_model(len(meta["charset"]) + 1, torch.device("cpu"), bias=0.8)
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    load_seconds = time.perf_counter() - started
    text = "Natural handwriting has rhythm and variation."
    ids = [meta["charset"].index(c) + 1 for c in text + "  "]
    context = torch.eye(len(meta["charset"]) + 1)[ids].unsqueeze(0)
    rows, results = [], []
    for bias, seed in [(0.5, 42), (0.8, 42), (1.1, 42), (0.8, 43)]:
        torch.manual_seed(seed)
        model.mixture.bias = bias
        start = time.perf_counter()
        with torch.inference_mode():
            points, attention = model.sample_means_with_attention(context, steps=4000, stochastic=True)
        elapsed = time.perf_counter() - start
        points = points * torch.tensor(meta["std"]) + torch.tensor(meta["mu"])
        absolute = points[:, :2].cumsum(0)
        xy = absolute.tolist()
        min_x, min_y = absolute.min(0).values.tolist()
        max_x, max_y = absolute.max(0).values.tolist()
        scale = min(1450 / max(max_x - min_x, 1), 130 / max(max_y - min_y, 1))
        row = [f'<text x="20" y="20" font-size="15" fill="#444">{escape(args.label)} | bias {bias} | seed {seed} | {elapsed:.2f}s | {len(points)} points</text>']
        for i in range(1, len(points)):
            if points[i - 1, 2] < 0.5:
                a, b = xy[i - 1], xy[i]
                row.append(f'<path d="M {20+(a[0]-min_x)*scale:.2f} {45+(a[1]-min_y)*scale:.2f} L {20+(b[0]-min_x)*scale:.2f} {45+(b[1]-min_y)*scale:.2f}" fill="none" stroke="#163763" stroke-width="2" stroke-linecap="round"/>')
        rows.append(row)
        result = {"bias": bias, "seed": seed, "seconds": elapsed, "points": len(points),
                  "attention_last_index": int(attention[-1].argmax()), "text_length": len(ids)}
        results.append(result)
        out = ROOT / "artifacts/evaluation" / f"{args.label}-b{bias}-s{seed}.json"
        out.write_text(json.dumps({"text": text, "offsets": points.tolist(), "metrics": result}))
        print(json.dumps(result), flush=True)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="1540" height="{len(rows)*200}"><rect width="100%" height="100%" fill="white"/>'
    svg += ''.join(f'<g transform="translate(0,{i*200})">'+''.join(row)+'</g>' for i, row in enumerate(rows)) + '</svg>'
    (ROOT / "artifacts/evaluation" / f"{args.label}-comparison.svg").write_text(svg)
    report = {"checkpoint": args.label, "torch": torch.__version__, "threads": 1,
              "load_seconds": load_seconds, "parameters": sum(p.numel() for p in model.parameters()),
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "samples": results}
    (ROOT / "artifacts/evaluation" / f"{args.label}-benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
