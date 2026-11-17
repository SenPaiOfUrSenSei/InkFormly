"""Exercise the running localhost studio with the real pretrained model."""
import hashlib
import json
from pathlib import Path
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"
TEXT = ("Natural handwriting has rhythm and variation. Every letter should feel a little different.\n\n"
        "Small details make a page feel personal: the space between words, the curve of a letter, "
        "and the way the ink meets the paper.")


def main():
    def wait(client, job_id):
        deadline = time.monotonic() + 180
        last = None
        while time.monotonic() < deadline:
            job = client.get(f"/api/jobs/{job_id}").raise_for_status().json()
            current = (job["state"], job["completed_lines"])
            if current != last:
                print(f"{job['operation']}: {current}", flush=True)
                last = current
            if job["state"] == "error":
                raise RuntimeError(job["error"])
            if job["state"] == "ready":
                return job
            time.sleep(0.2)
        raise TimeoutError("Local handwriting job timed out")

    with httpx.Client(base_url=BASE, timeout=30) as client:
        model = client.get("/api/model").raise_for_status().json()
        assert model["available"] and model["readback_available"]
        for path in ("/", "/app.js", "/style.css"):
            assert client.get(path).status_code == 200
        assert client.post("/api/generate", json={"text": "Unsupported 🖊"}).status_code == 422
        response = client.post("/api/generate", json={"text": TEXT, "seed": 42}).raise_for_status()
        generation = wait(client, response.json()["id"])
        job_id = generation["id"]
        original = client.get(f"/api/jobs/{job_id}/files/document.json").raise_for_status().content
        assert json.loads(original)["source_text"] == TEXT
        assert all(line["attention_complete"] for line in json.loads(original)["lines"])
        review = client.get(f"/api/jobs/{job_id}/review").raise_for_status().json()
        assert review["quality_checks"]
        assert all(line["readback"]["status"] in ("match", "review") for line in review["lines"])
        target = next((line["index"] for line in review["lines"] if line["readback"]["status"] == "review"), review["lines"][-1]["index"])
        old_document = json.loads(original)
        response = client.post(f"/api/jobs/{job_id}/lines/{target}/rewrite", json={}).raise_for_status()
        rewritten = wait(client, response.json()["id"])
        current_id = rewritten["id"]
        original = client.get(f"/api/jobs/{current_id}/files/document.json").raise_for_status().content
        new_document = json.loads(original)
        assert new_document["source_text"] == old_document["source_text"]
        assert all(line == new_document["lines"][i] for i, line in enumerate(old_document["lines"]) if i != target)
        preview = client.get(f"/api/jobs/{job_id}/files/preview-001.png").raise_for_status().content
        assert preview.startswith(b"\x89PNG")
        response = client.post(f"/api/jobs/{current_id}/render", json={"ink": {"color": "#1c232a", "roughness": 0.7}}).raise_for_status()
        rendered = wait(client, response.json()["id"])
        changed = client.get(f"/api/jobs/{rendered['id']}/files/document.json").raise_for_status().content
        assert original == changed
        response = client.post(f"/api/jobs/{rendered['id']}/export", json={}).raise_for_status()
        exported = wait(client, response.json()["id"])
        pdf = client.get(f"/api/jobs/{exported['id']}/files/inkform.pdf").raise_for_status().content
        png_zip = client.get(f"/api/jobs/{exported['id']}/files/pages.zip").raise_for_status().content
        assert pdf.startswith(b"%PDF") and png_zip.startswith(b"PK")
        report = {"base_url": BASE, "model": model["backend"], "generation_id": job_id,
                  "render_id": rendered["id"], "export_id": exported["id"], "pages": generation["pages"],
                  "generation_seconds": generation["seconds"], "text_characters": len(TEXT),
                  "rewritten_line": target + 1, "other_lines_preserved": True,
                  "readback_matches_before_rewrite": generation["readback_matches"],
                  "review_lines_before_rewrite": generation["review_lines"],
                  "trajectories_unchanged_after_ink_edit": True,
                  "trajectory_sha256": hashlib.sha256(original).hexdigest(),
                  "pdf_bytes": len(pdf), "png_zip_bytes": len(png_zip)}
        (ROOT / "artifacts/evaluation/quality-live-smoke.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
