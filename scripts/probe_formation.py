"""Controlled decoder experiments; no glyph replacements or font substitutions."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw
import torch

from inkform.generator import GravesGenerator
from inkform.recognition import get_recognizer, recognition_image, compare_readback
from inkform.schema import StyleReference

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/formation"


@dataclass(frozen=True)
class Variant:
    name: str
    focus: float = 1
    speed: float = 1
    finish_last_space: bool = True


VARIANTS = [Variant("legacy", finish_last_space=False), Variant("finish"),
            Variant("focus", focus=1.5), Variant("slow-focus", focus=1.5, speed=.9),
            Variant("gentle", focus=1.2, finish_last_space=False),
            Variant("balanced", focus=1.25, speed=.95, finish_last_space=False),
            Variant("steady", focus=1.5, speed=.9, finish_last_space=False)]


def decode(g, text, bias, seed, reference, variant):
    m = g.model
    m.mixture.bias = bias
    prefix = reference.text + "  " if reference else ""
    context = g._context(prefix + text + "  ")
    x = m.get_initial_input().reshape(1, 1, 3)
    w = m.get_initial_window(1)
    k = torch.zeros(1, m.gaussian_components)
    h1s, h2s, h3s = m.get_all_initial_states(1)
    points, labels = [], []

    def step(current):
        nonlocal w, k, h1s, h2s, h3s
        h1, h1s = m.lstm1(torch.cat((current, w), -1), h1s)
        if variant.focus == 1 and variant.speed == 1:
            phi, k = m.window(h1, context, k)
        else:
            h = h1[:, 0]
            alpha = torch.exp(m.window.alpha(h))
            beta = torch.exp(m.window.beta(h))
            k = k + torch.exp(m.window.k(h)) * variant.speed
            original = m.window.compute_attention_weights(alpha, beta, k, context.shape[1])
            phi = m.window.compute_attention_weights(alpha, beta * variant.focus, k, context.shape[1])
            phi *= original.sum(-1, keepdim=True) / phi.sum(-1, keepdim=True).clamp_min(1e-9)
        w = torch.bmm(phi, context)
        mixture, h2s, h3s = m.compute_mixture(current, h1, w, h2s, h3s)
        return mixture, phi

    stop = len(prefix) + len(text) + int(variant.finish_last_space)
    complete = False
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        if reference:
            step(x)
            for p in reference.normalized_offsets[:-1]:
                step(torch.tensor(p).reshape(1, 1, 3))
            x = torch.tensor(reference.normalized_offsets[-1]).reshape(1, 1, 3)
        for _ in range(min(8000, max(1200, len(text) * 85))):
            mixture, phi = step(x)
            x = g._sample(mixture)
            if not torch.isfinite(x).all() or x[0, 0, :2].abs().max() > 30:
                break
            points.append(x[0, 0].clone())
            labels.append(int(phi[0, 0].argmax())-len(prefix))
            if labels[-1] >= stop-len(prefix) and x[0, 0, 2] > .5:
                complete = True
                break
    if not points:
        return None
    normalized = torch.stack(points)
    normalized[-1, 2] = 1
    offsets = normalized*g.std+g.mu
    return {"offsets": offsets.tolist(), "complete": complete, "points": len(points),
            "character_dwell": {str(i): labels.count(i) for i in range(-2, len(text)+2)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default="52")
    parser.add_argument("--variants", nargs="*", default=["legacy","finish","focus","slow-focus"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    g = GravesGenerator(ROOT / f"models/graves/Epoch_{args.epoch}")
    ref = StyleReference.model_validate_json((ROOT / "inkform/assets/default_style.json").read_text())
    reader = get_recognizer()
    cases = [("student here,", 817, .8), ("student here,", 817, 1.0),
             ("Hey Eric! Former SAT masterclass student here,", 104771, .8),
             ("on the sat :)", 817, .8)]
    variants = [v for v in VARIANTS if args.variants is None or v.name in args.variants]
    results = []
    for case_index, (text, seed, bias) in enumerate(cases):
        rows = []
        for variant in variants:
            start = time.perf_counter()
            result = decode(g, text, bias, seed, ref, variant)
            if result is None:
                continue
            readback = reader.read(result["offsets"])
            result.update(text=text, seed=seed, bias=bias, variant=variant.name,
                          seconds=time.perf_counter()-start, readback=compare_readback(text, readback).model_dump())
            results.append(result)
            im = recognition_image(result["offsets"])
            row = Image.new("RGB", (1600, 190), "white")
            draw = ImageDraw.Draw(row)
            draw.text((12, 6), f"Epoch {args.epoch} / {variant.name} / bias={bias} / completed={result['complete']} / {readback}", fill="black")
            im.thumbnail((1550, 140))
            row.paste(im, (15, 35))
            rows.append(row)
            print(json.dumps({k: result[k] for k in ("text", "variant", "complete", "points", "readback")}), flush=True)
            (OUT / f"epoch{args.epoch}-probe.json").write_text(json.dumps(results))
        sheet = Image.new("RGB", (1600, len(rows)*190), "white")
        for i, row in enumerate(rows):
            sheet.paste(row, (0, i*190))
        sheet.save(OUT / f"epoch{args.epoch}-case{case_index}.png")


if __name__ == "__main__":
    main()
