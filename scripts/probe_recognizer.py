"""Check the independent handwriting recognizer against existing generated lines."""
import json
from pathlib import Path
import time

import torch

from inkform.recognition import get_recognizer, recognition_image


if __name__ == "__main__":
    torch.set_num_threads(1)
    start = time.perf_counter()
    recognizer = get_recognizer()
    print(f"Loaded readback model in {time.perf_counter()-start:.2f}s", flush=True)
    data = json.loads(Path("output/demo-fixed/document.json").read_text())
    results = []
    for i, line in enumerate(data["lines"]):
        if not line["offsets"]:
            continue
        start = time.perf_counter()
        image = recognition_image(line["offsets"])
        image.save(f"artifacts/evaluation/readback-line-{i}.png")
        actual = recognizer.read(line["offsets"])
        result = {"requested": line["text"], "recognized": actual, "seconds": time.perf_counter()-start}
        results.append(result)
        print(json.dumps(result), flush=True)
    Path("artifacts/evaluation/readback-baseline.json").write_text(json.dumps(results, indent=2))
