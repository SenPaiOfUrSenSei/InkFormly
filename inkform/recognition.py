"""Independent offline readback. The recognizer never sees the requested text."""
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

import numpy as np
from PIL import Image, ImageDraw
import torch

from .render import strokes_from_offsets
from .schema import Readback

OCR_DIR = Path(__file__).resolve().parents[1] / "models/trocr-small-handwritten"


def canonical_text(text: str) -> str:
    """Ignore OCR token spacing, but preserve case, numbers and punctuation."""
    text = unicodedata.normalize("NFC", text).translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    text = " ".join(text.split())
    text = re.sub(r"\s+([.,:;!?\)\]])", r"\1", text)
    return re.sub(r"([\(\[])\s+", r"\1", text)


def edit_distance(expected: str, observed: str) -> int:
    previous = list(range(len(observed) + 1))
    for i, a in enumerate(expected, 1):
        current = [i]
        for j, b in enumerate(observed, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def compare_readback(expected: str, observed: str) -> Readback:
    source, recognized = canonical_text(expected), canonical_text(observed)
    distance = edit_distance(source, recognized)
    number_pattern = r"[+-]?\d+(?:[.,]\d+)*"
    numbers_match = re.findall(number_pattern, source) == re.findall(number_pattern, recognized)
    return Readback(status="match" if source == recognized else "review", recognized_text=observed,
                    edit_distance=distance, character_error_rate=distance / max(1, len(source)), numbers_match=numbers_match)


def recognition_image(offsets) -> Image.Image:
    strokes = strokes_from_offsets(offsets)
    if not strokes:
        raise ValueError("There are no strokes to read.")
    points = np.concatenate(strokes)
    centered = points[:, 0] - points[:, 0].mean()
    variance = float(centered @ centered)
    slope = float(centered @ (points[:, 1] - points[:, 1].mean()) / variance) if variance else 0
    for stroke in strokes:
        stroke[:, 1] -= 0.9 * slope * (stroke[:, 0] - points[0, 0])
    points = np.concatenate(strokes)
    low, high = points.min(0), points.max(0)
    scale = min(80 / max(float(high[1] - low[1]), 1), 2600 / max(float(high[0] - low[0]), 1))
    size = np.ceil((high - low) * scale).astype(int) + 32
    image = Image.new("RGB", tuple(size.tolist()), "white")
    draw = ImageDraw.Draw(image)
    for stroke in strokes:
        xy = [(float(x), float(y)) for x, y in (stroke - low) * scale + 16]
        if len(xy) > 1:
            draw.line(xy, fill="black", width=2, joint="curve")
        else:
            x, y = xy[0]
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="black")
    return image


class HandwritingRecognizer:
    def __init__(self, model_dir: Path = OCR_DIR):
        if not (model_dir / "pytorch_model.bin").is_file():
            raise RuntimeError("Handwriting readback model missing. Run scripts/download_recognizer.py.")
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.processor = TrOCRProcessor.from_pretrained(str(model_dir), local_files_only=True, use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(str(model_dir), local_files_only=True,
                                                             use_safetensors=False).eval()

    def read(self, offsets) -> str:
        image = recognition_image(offsets)
        try:
            pixels = self.processor(images=image, return_tensors="pt").pixel_values
            with torch.inference_mode():
                ids = self.model.generate(pixels, max_new_tokens=160, num_beams=1, do_sample=False)
            return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        finally:
            image.close()

    def evidence(self, offsets, expected: str) -> dict:
        """Blind decoding plus a separate, teacher-forced caption likelihood.

        Expected text is used only for the likelihood calculation, never passed
        to generate(). Reuse the visual encoder and bound short-word decoding.
        This measures recognizer confidence, not a calibrated legibility score.
        """
        from torch.nn import functional as F
        image = recognition_image(offsets)
        try:
            pixels = self.processor(images=image, return_tensors="pt").pixel_values
            # This checkpoint starts at EOS and immediately predicts content;
            # the tokenizer's default BOS would shift every score by one token.
            token_ids = self.processor.tokenizer(expected, add_special_tokens=False).input_ids
            labels = torch.tensor([token_ids+[self.model.generation_config.eos_token_id]], dtype=torch.long)
            start = self.model.generation_config.decoder_start_token_id
            if start is None:
                start = self.model.decoder.config.decoder_start_token_id
            decoder = torch.cat((torch.full((1, 1), start, dtype=torch.long), labels[:, :-1]), dim=1)
            with torch.inference_mode():
                encoded = self.model.encoder(pixel_values=pixels)
                ids = self.model.generate(pixel_values=pixels, encoder_outputs=encoded,
                    max_new_tokens=min(160, max(12, len(expected)*3+8)), num_beams=1, do_sample=False)
                logits = self.model(encoder_outputs=encoded, decoder_input_ids=decoder).logits
                # Score content only. Otherwise an omitted EOS before a visible
                # period can dominate the metric without measuring the letters.
                content = max(1,len(token_ids))
                nll = float(F.cross_entropy(logits[:,:content].reshape(-1, logits.shape[-1]), labels[:,:content].reshape(-1)))
            observed = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            return {"recognized_text": observed, "nll": nll}
        finally:
            image.close()


@lru_cache(maxsize=1)
def get_recognizer() -> HandwritingRecognizer:
    return HandwritingRecognizer()
