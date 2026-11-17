from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import threading
import uuid
import zipfile

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .generator import get_generator, ROOT
from .readable_generator import READABLE_MODEL_DIR, DISPLAY_NAME, DEFAULT_STYLE
from .stroke_network import ALPHABET
from .render import export_document, layout_document, render_page
from .schema import Document, GenerationSettings, RenderSettings
from .recognition import OCR_DIR
from .word_clarity import line_word_review,CLARITY_MODE
from .variation_presets import VARIATION_PRESETS

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    state: str = "queued"
    operation: str = "generate"
    completed_lines: int = 0
    total_lines: int = 0
    current_line: str = ""
    error: str | None = None
    pages: int = 0
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0
    export_ready: bool = False
    readback_matches: int = 0
    review_lines: list[int] = field(default_factory=list)


class JobManager:
    def __init__(self, output: Path, factory=get_generator):
        self.output = output
        self.factory = factory
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.capacity = threading.BoundedSemaphore(3)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inkform")

    def folder(self, job_id: str) -> Path:
        if job_id not in self.jobs:
            raise HTTPException(404, "Document not found in this session.")
        return self.output / job_id

    def submit(self, operation, function) -> Job:
        if not self.capacity.acquire(blocking=False):
            raise HTTPException(429, "The local queue is full. Wait for the current documents to finish.")
        job = Job(id=uuid.uuid4().hex, operation=operation)
        with self.lock:
            # Keep memory bounded; saved artifacts remain on disk.
            if len(self.jobs) >= 40:
                old = next((key for key, value in self.jobs.items() if value.state in ("ready", "error")), None)
                if old:
                    self.jobs.pop(old)
            self.jobs[job.id] = job

        def run():
            try:
                folder = self.output / job.id
                folder.mkdir(parents=True, exist_ok=True)
                function(job, folder)
                with self.lock:
                    job.state = "ready"
            except Exception as exc:
                log.exception("InkForm job failed")
                with self.lock:
                    job.state = "error"
                    job.error = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "The local job failed. See the server log for details."
            finally:
                self.capacity.release()

        self.executor.submit(run)
        return job

    def previews(self, job, folder, document, settings):
        with self.lock:
            job.state = "rendering"
        layout = layout_document(document, settings.page)
        line_index = 0
        for i, lines in enumerate(layout.pages):
            image = render_page(lines, settings.page, settings.ink, 120, i)
            image.save(folder / f"preview-{i + 1:03d}.png")
            px = 120 / 25.4
            for row, placed in enumerate(lines):
                if placed.text:
                    top = (settings.page.margin_mm + row * settings.page.line_spacing_mm) * px
                    crop = image.crop((round((settings.page.margin_mm - 1) * px), round(top),
                                       round((211 - settings.page.margin_mm) * px),
                                       round(top + settings.page.line_spacing_mm * px)))
                    crop.save(folder / f"line-{line_index + 1:03d}.png")
                    crop.close()
                line_index += 1
            image.close()
        (folder / "document.json").write_text(document.model_dump_json())
        (folder / "reference.json").write_text(document.reference.model_dump_json())
        (folder / "settings.json").write_text(settings.model_dump_json())
        with self.lock:
            job.pages = len(layout.pages)
            job.warnings = list(document.warnings)
            rejected_r = document.withdrawn_r_warning()
            if rejected_r and rejected_r not in job.warnings:
                job.warnings.append(rejected_r)
            if layout.fitted_to_width:
                job.warnings.append("Writing was reduced uniformly to fit the page width.")
            job.seconds = document.generation_seconds
            job.readback_matches = sum(line.readback.status == "match" for line in document.lines if line.text)
            job.review_lines = [i + 1 for i, line in enumerate(document.lines) if line.text and
                (line.readback.status != "match" or (line_word_review(line) or {}).get('status')=='review')]

    def generate(self, settings):
        def work(job, folder):
            with self.lock:
                job.state = "generating"

            def progress(i, total, text):
                with self.lock:
                    job.completed_lines, job.total_lines, job.current_line = i, total, text

            document = self.factory().generate(settings, progress)
            with self.lock:
                job.completed_lines = job.total_lines
            self.previews(job, folder, document, RenderSettings(ink=settings.ink, page=settings.page))

        return self.submit("generate", work)

    def render(self, parent_id, settings):
        folder = self.folder(parent_id)
        if self.jobs[parent_id].state != "ready":
            raise HTTPException(409, "Wait for the document to finish.")
        document = Document.model_validate_json((folder / "document.json").read_text())
        return self.submit("render", lambda job, path: self.previews(job, path, document, settings))

    def rewrite(self, parent_id, index):
        source = self.folder(parent_id)
        if self.jobs[parent_id].state != "ready":
            raise HTTPException(409, "Wait for the document to finish.")
        document = Document.model_validate_json((source / "document.json").read_text())
        if not 0 <= index < len(document.lines) or not document.lines[index].text:
            raise HTTPException(422, "Choose a written line to rewrite.")
        settings = RenderSettings.model_validate_json((source / "settings.json").read_text())

        def work(job, folder):
            with self.lock:
                job.state, job.total_lines, job.current_line = "generating", 1, document.lines[index].text
            updated = self.factory().rewrite_line(document, index)
            with self.lock:
                job.completed_lines = 1
            self.previews(job, folder, updated, settings)

        return self.submit("rewrite", work)

    def export(self, parent_id):
        source = self.folder(parent_id)
        if self.jobs[parent_id].state != "ready":
            raise HTTPException(409, "Wait for the document to finish.")
        document = Document.model_validate_json((source / "document.json").read_text())
        settings = RenderSettings.model_validate_json((source / "settings.json").read_text())

        def work(job, folder):
            with self.lock:
                job.state = "rendering"
            manifest = export_document(document, settings.page, settings.ink, folder, 300)
            (folder / "manifest.json").write_text(json.dumps(manifest))
            (folder / "document.json").write_text(document.model_dump_json())
            with zipfile.ZipFile(folder / "pages.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for name in manifest["png"]:
                    archive.write(folder / name, name)
            with self.lock:
                job.pages = manifest["pages"]
                job.export_ready = True
                job.warnings = list(document.warnings)
                rejected_r = document.withdrawn_r_warning()
                if rejected_r and rejected_r not in job.warnings:
                    job.warnings.append(rejected_r)

        return self.submit("export", work)


def create_app(output: Path = ROOT / "output/jobs", factory=get_generator) -> FastAPI:
    manager = JobManager(output, factory)

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.executor.shutdown(wait=True)

    app = FastAPI(title="InkForm", lifespan=lifespan)
    app.state.manager = manager

    @app.get("/api/variation-presets")
    def variation_presets():
        return VARIATION_PRESETS

    @app.get("/api/model")
    def model():
        return {"backend": DISPLAY_NAME, "device": "CPU", "parameters": 3637231,
                "available": all((READABLE_MODEL_DIR / name).is_file() for name in
                                 ("model.pt", f"styles/style-{DEFAULT_STYLE}-chars.npy", f"styles/style-{DEFAULT_STYLE}-strokes.npy")),
                "readback_available": (OCR_DIR / "pytorch_model.bin").is_file(),
                "t_refinement_available": True,
                "t_refinement_default": True,
                "t_refinement_mode": CLARITY_MODE,
                "cursive_r_available": False,
                "cursive_r_selection_available": True,
                "charset": ALPHABET[1:]}

    @app.post("/api/generate", status_code=202)
    def generate(settings: GenerationSettings):
        if not settings.text.strip():
            raise HTTPException(422, "Enter some text to generate.")
        if factory is get_generator and not model()["available"]:
            raise HTTPException(503, "Pretrained checkpoint or writer sample missing. Run scripts/download_readable_model.py.")
        if factory is get_generator and settings.quality_checks and not (OCR_DIR / "pytorch_model.bin").is_file():
            raise HTTPException(503, "Handwriting readback model missing. Run scripts/download_recognizer.py, or turn off wording checks for a faster unchecked draft.")
        # Reject unsupported characters before queuing, without loading PyTorch weights.
        if factory is get_generator:
            charset = ALPHABET[1:]
            text = settings.text + (settings.reference.text if settings.reference else "")
            missing = sorted(set(text) - set(charset) - set("\r\n\t"))
            if missing:
                raise HTTPException(422, f"Unsupported characters: {', '.join(repr(c) for c in missing)}")
        return {"id": manager.generate(settings).id}

    @app.get("/api/jobs/{job_id}")
    def status(job_id: str):
        manager.folder(job_id)
        with manager.lock:
            job = manager.jobs[job_id]
            return {"id": job.id, "state": job.state, "operation": job.operation,
                    "completed_lines": job.completed_lines, "total_lines": job.total_lines,
                    "current_line": job.current_line, "error": job.error, "pages": job.pages,
                    "warnings": list(job.warnings), "seconds": job.seconds, "export_ready": job.export_ready,
                    "readback_matches": job.readback_matches, "review_lines": list(job.review_lines)}

    @app.get("/api/jobs/{job_id}/review")
    def review(job_id: str):
        folder = manager.folder(job_id)
        if manager.jobs[job_id].state != "ready" or not (folder / "document.json").is_file():
            raise HTTPException(409, "Wait for the document to finish.")
        document = Document.model_validate_json((folder / "document.json").read_text())
        return {"quality_checks": document.quality_checks,
                "lines": [{"index": i, "text": line.text, "readback": line.readback.model_dump(),
                           "word_clarity": line_word_review(line),
                           "r_style": line.r_formation,
                           "attempts": line.attempts, "preview": f"line-{i + 1:03d}.png"}
                          for i, line in enumerate(document.lines) if line.text]}

    @app.post("/api/jobs/{job_id}/lines/{index}/rewrite", status_code=202)
    def rewrite(job_id: str, index: int):
        return {"id": manager.rewrite(job_id, index).id}

    @app.post("/api/jobs/{job_id}/render", status_code=202)
    def rerender(job_id: str, settings: RenderSettings):
        return {"id": manager.render(job_id, settings).id}

    @app.post("/api/jobs/{job_id}/export", status_code=202)
    def export(job_id: str):
        return {"id": manager.export(job_id).id}

    @app.get("/api/jobs/{job_id}/files/{name}")
    def download(job_id: str, name: str):
        folder = manager.folder(job_id)
        # No user-supplied filesystem paths; serve only concrete output filenames.
        allowed = {"inkform.pdf", "pages.zip", "document.json", "reference.json"}
        preview = name.startswith("preview-") and name.endswith(".png") and name[8:-4].isdigit()
        line_preview = name.startswith("line-") and name.endswith(".png") and name[5:-4].isdigit()
        if name not in allowed and not preview and not line_preview:
            raise HTTPException(404, "File not found.")
        path = folder / name
        if not path.is_file():
            raise HTTPException(404, "File not found.")
        return FileResponse(path, filename=name if name in allowed else None)

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
    return app


app = create_app()
