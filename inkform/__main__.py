import argparse
from pathlib import Path

from .schema import Document, GenerationSettings, InkSettings, PageSettings, StyleReference
from .variation_presets import VARIATION_PRESETS


def main():
    parser = argparse.ArgumentParser(prog="inkform", description="Neural handwriting and independent ink rendering")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Generate handwriting and export A4 pages")
    text = generate.add_mutually_exclusive_group(required=True)
    text.add_argument("--text")
    text.add_argument("--input", type=Path)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--bias", type=float, default=0.85)
    generate.add_argument("--form-diversity", type=float, default=0.5,
                          help="Choice of learned pen movements, 0 to 1; zero uses the original sampler")
    generate.add_argument("--t-refinement",action=argparse.BooleanOptionalAction,default=True,
                          help="Check t words and regenerate closed words at safe boundaries; requires readback")
    generate.add_argument("--cursive-r",action=argparse.BooleanOptionalAction,default=False,
                          help=argparse.SUPPRESS)
    generate.add_argument("--prefer-cursive-r", action="store_true",
                          help="Fit the reference cursive r between existing joins in checked words")
    generate.add_argument("--reference", type=Path)
    generate.add_argument("--unprimed", action="store_true")
    generate.add_argument("--fast", action="store_true", help="Skip independent readback and candidate ranking")
    render = sub.add_parser("render", help="Render saved trajectories without inference")
    render.add_argument("document", type=Path)
    for command in (generate, render):
        command.add_argument("--out", type=Path, default=Path("output/handwriting"))
        command.add_argument("--color", default="#203f78")
        command.add_argument("--width", type=float, default=0.25)
        command.add_argument("--roughness", type=float, default=0.4)
        command.add_argument("--opacity", type=float, default=0.95)
        command.add_argument("--plain", action="store_true")
        command.add_argument("--height", type=float, default=7.5)
        command.add_argument("--margin", type=float, default=20)
        command.add_argument("--spacing", type=float, default=11.5)
        command.add_argument("--variation-preset", choices=tuple(VARIATION_PRESETS))
        command.add_argument("--naturalness", type=float, help="Relaxed line layout and movement, 0 to 1")
        command.add_argument("--letter-variation", type=float, help="Local letter proportion and slant variation, 0 to 1")
        command.add_argument("--word-gap-min-mm", type=float, help="Widen detected blank word gaps before line fitting, 0 to 2.5 mm")
        command.add_argument("--slant-adjustment-deg", type=float, help="Relative letter lean adjustment, -50 to 50 degrees; negative reduces rightward lean")
        command.add_argument("--fit-lines-individually", action=argparse.BooleanOptionalAction, default=None,
                             help="Fit crowded lines separately to keep other lines at the requested size")
        for field in ("width_variation", "height_variation", "slant_variation", "rhythm_variation",
                      "word_spacing_variation", "line_start_variation", "line_end_variation", "line_spacing_variation",
                      "line_tilt_variation", "baseline_variation", "density_variation", "pressure_variation"):
            command.add_argument("--"+field.replace("_","-"),type=float,help="Variation strength, 0 to 1")
        command.add_argument("--variation-seed", type=int, default=29)
        command.add_argument("--dpi", type=int, choices=(96, 120, 150, 300), default=300)
    serve = sub.add_parser("serve", help="Open the local studio server")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run("inkform.server:app", host="127.0.0.1", port=args.port)
        return
    try:
        preset = VARIATION_PRESETS.get(args.variation_preset,{"page":{},"ink":{}})
        ink_options = {**preset["ink"],"color":args.color,"width_mm":args.width,
                       "roughness":args.roughness,"opacity":args.opacity}
        page_options = {**preset["page"],"paper":"plain" if args.plain else "ruled",
                        "writing_height_mm":args.height,"margin_mm":args.margin,
                        "line_spacing_mm":args.spacing,"layout_seed":args.variation_seed}
        for field in ("naturalness","letter_variation","width_variation","height_variation",
                      "slant_variation","rhythm_variation","word_spacing_variation","line_start_variation","line_end_variation",
                      "line_spacing_variation","line_tilt_variation","baseline_variation","word_gap_min_mm","slant_adjustment_deg"):
            if getattr(args,field) is not None:page_options[field] = getattr(args,field)
        if args.fit_lines_individually is not None:page_options['fit_lines_individually'] = args.fit_lines_individually
        for field in ("density_variation","pressure_variation"):
            if getattr(args,field) is not None:ink_options[field] = getattr(args,field)
        ink, page = InkSettings(**ink_options), PageSettings(**page_options)
        if args.command == "generate":
            from .generator import get_generator
            reference = StyleReference.model_validate_json(args.reference.read_text()) if args.reference else None
            settings = GenerationSettings(text=args.text if args.text is not None else args.input.read_text(),
                                          seed=args.seed, bias=args.bias, reference=reference,
                                          form_diversity=args.form_diversity,
                                          t_refinement=args.t_refinement,
                                          cursive_r=args.cursive_r,
                                          prefer_cursive_r=args.prefer_cursive_r,
                                          consistent_style=not args.unprimed, quality_checks=not args.fast, ink=ink, page=page)
            document = get_generator().generate(settings, lambda i, n, text: print(f"Writing {i + 1}/{n}: {text}", flush=True))
        else:
            document = Document.model_validate_json(args.document.read_text())
        from .render import export_document
        args.out.mkdir(parents=True, exist_ok=True)
        manifest = export_document(document, page, ink, args.out, args.dpi)
        (args.out / "document.json").write_text(document.model_dump_json())
        (args.out / "reference.json").write_text(document.reference.model_dump_json())
        print(f"Saved {manifest['pages']} page(s) to {args.out.resolve()}")
        print(f"Generation: {document.generation_seconds:.2f}s; pressure: {document.pressure_source}")
        for warning in document.warnings:
            print(f"Note: {warning}")
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"InkForm: {exc}\n")


if __name__ == "__main__":
    main()
