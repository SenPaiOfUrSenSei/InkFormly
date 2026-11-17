"""Compare blind OCR disagreement on a fixed project-style text, not true CER."""
import json
import argparse
from pathlib import Path
import time

from inkform.generator import get_generator
from inkform.recognition import compare_readback, get_recognizer
from inkform.schema import GenerationSettings

ROOT = Path(__file__).resolve().parents[1]
TEXT = "\n".join([
    "The water cycle includes evaporation.",
    "Plants use sunlight to make their food.",
    "The experiment began on 15 July 2026.",
    "We recorded 25 samples over 7 days.",
    "The result was 12.5 percent.",
    "Clean water supports every living organism.",
    "Repeated letters occur in little and letter.",
    "Conclusion: small changes can save water.",
])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checked-only", action="store_true", help="Reuse the saved initial baseline")
    args = parser.parse_args()
    generator = get_generator()
    recognizer = get_recognizer()
    report_path = ROOT / "artifacts/evaluation/quality-comparison.json"
    reports = json.loads(report_path.read_text()) if args.checked_only else {}
    for checked in (True,) if args.checked_only else (False, True):
        label = "checked" if checked else "baseline"
        start = time.perf_counter()
        document = generator.generate(GenerationSettings(text=TEXT, quality_checks=checked),
                                      lambda i, n, text: print(f"{label} {i+1}/{n}: {text}", flush=True))
        if not checked:
            for line in document.lines:
                line.readback = compare_readback(line.text, recognizer.read(line.offsets))
        lines = [{"text": line.text, "recognized": line.readback.recognized_text,
                  "distance": line.readback.edit_distance, "status": line.readback.status,
                  "candidates": line.attempts} for line in document.lines]
        reports[label] = {"seconds_including_readback": time.perf_counter()-start,
                          "matching_lines": sum(line["status"] == "match" for line in lines),
                          "total_lines": len(lines), "ocr_edit_distance": sum(line["distance"] for line in lines),
                          "lines": lines}
        (ROOT / f"artifacts/evaluation/project-{label}.json").write_text(document.model_dump_json())
        print(json.dumps(reports[label], indent=2), flush=True)
    reports["limitation"] = "OCR disagreement measures a recognizer's reading, not guaranteed true transcription error. Checked samples were selected using this recognizer. Reused initial baselines compare full pipeline revisions, including the reference style."
    report_path.write_text(json.dumps(reports, indent=2))
