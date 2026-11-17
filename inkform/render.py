"""Physical page layout and simulated optical ink deposition, independent of ML."""
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from .schema import Document, InkSettings, PageSettings


@dataclass
class PlacedLine:
    text: str
    strokes: list[np.ndarray]
    baseline_mm: float


@dataclass
class Layout:
    pages: list[list[PlacedLine]]
    scale: float
    fitted_to_width: bool


def strokes_from_offsets(offsets) -> list[np.ndarray]:
    if not offsets:
        return []
    data = np.asarray(offsets, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3 or not np.isfinite(data).all():
        raise ValueError("Invalid trajectory data.")
    xy = np.cumsum(data[:, :2], axis=0)
    strokes, start = [], 0
    for i, end in enumerate(data[:, 2]):
        if end >= 0.5:
            strokes.append(xy[start:i + 1])
            start = i + 1
    if start < len(xy):
        strokes.append(xy[start:])
    return strokes


def layout_document(document: Document, settings: PageSettings) -> Layout:
    prepared, heights, widths = [], [], []
    for line in document.lines:
        strokes = strokes_from_offsets(line.offsets)
        if not strokes:
            prepared.append((line.text, [], np.zeros(2), np.zeros(2), 0.0))
            continue
        all_points = np.concatenate(strokes)
        # Correct most global line tilt while retaining local learned movement.
        centered_x = all_points[:, 0] - all_points[:, 0].mean()
        variance = float(centered_x @ centered_x)
        if len(all_points) > 50 and variance > 0:
            slope = float(centered_x @ (all_points[:, 1] - all_points[:, 1].mean()) / variance)
            for stroke in strokes:
                stroke[:, 1] -= 0.9 * slope * (stroke[:, 0] - all_points[0, 0])
            all_points = np.concatenate(strokes)
        if settings.slant_adjustment_deg:
            strokes = _adjust_slant(strokes, settings.slant_adjustment_deg)
            all_points = np.concatenate(strokes)
        low, high = all_points.min(axis=0), all_points.max(axis=0)
        baseline = float(np.quantile(all_points[:, 1], 0.8)) - low[1]
        heights.append(high[1] - low[1])
        widths.append(high[0] - low[0])
        prepared.append((line.text, strokes, low, high, baseline))
    if not heights or min(heights) <= 0:
        raise ValueError("Document has no drawable handwriting.")
    # One scale for the whole document preserves relative letter proportions.
    scale = settings.writing_height_mm / float(np.median(heights))
    usable_width = 210 - 2 * settings.margin_mm
    # Reserve a little room for relaxed line starts and width variation.
    width_scale = (usable_width - 8 * settings.naturalness) / max(widths)
    fitted = bool(scale > width_scale)
    if not settings.fit_lines_individually:
        scale = min(scale, width_scale, (settings.line_spacing_mm - 1.5) / max(heights))
    per_page = max(1, int((297 - 2 * settings.margin_mm) // settings.line_spacing_mm))
    pages: list[list[PlacedLine]] = [[]]
    rng = np.random.default_rng(np.random.SeedSequence([settings.layout_seed, document.seed]))
    for index, (text, strokes, low, high, relative_baseline) in enumerate(prepared):
        if index and index % per_page == 0:
            pages.append([])
        y_top = settings.margin_mm + (index % per_page) * settings.line_spacing_mm
        baseline = y_top + settings.line_spacing_mm * 0.78
        y = baseline - relative_baseline * scale
        # Approximate baselines can be unreliable for a very short line. Shift
        # that line inside its box rather than shrinking the entire document.
        y = float(np.clip(y, y_top + 0.5, y_top + settings.line_spacing_mm - 0.5 - (high[1] - low[1]) * scale))
        origin = np.array([settings.margin_mm, y])
        line_scale = scale
        if strokes and settings.fit_lines_individually:
            # Use a common preferred size, fitting only the crowded/tall line
            # that needs it. One wide line no longer shrinks all other lines.
            line_scale = min(scale,
                (usable_width-8*settings.naturalness)/max(float(high[0]-low[0]),.01),
                (settings.line_spacing_mm-1.5)/max(float(high[1]-low[1]),.01))
            y = float(np.clip(baseline-relative_baseline*line_scale,y_top+.5,
                y_top+settings.line_spacing_mm-.5-(high[1]-low[1])*line_scale))
            origin[1] = y
        placed = [(s - low) * line_scale + origin for s in strokes]
        if placed and (settings.word_spacing_variation or settings.word_gap_min_mm):
            gap_rng = np.random.default_rng(np.random.SeedSequence(
                [settings.layout_seed, document.seed, index, 171]))
            placed = _vary_word_gaps(placed, text, settings.word_spacing_variation, gap_rng,
                                    settings.word_gap_min_mm)
        if placed and settings.letter_variation:
            # Independent stream: changing letter variation does not pick new
            # line starts or depend on the ink/DPI used for export.
            letter_rng = np.random.default_rng(np.random.SeedSequence(
                [settings.layout_seed, document.seed, index, 173]))
            placed = _vary_letters(placed, baseline, text, settings.letter_variation, letter_rng,
                settings.width_variation, settings.height_variation,
                settings.slant_variation, settings.rhythm_variation)
        if placed and settings.naturalness:
            end_rng = np.random.default_rng(np.random.SeedSequence(
                [settings.layout_seed, document.seed, index, 179])) if settings.line_end_variation else None
            placed, baseline = _relax_line(placed, baseline, y_top, settings, rng, end_rng)
        elif placed and (settings.letter_variation or settings.word_spacing_variation or settings.word_gap_min_mm):
            placed, baseline = _fit_line(placed, baseline, y_top, settings)
        pages[-1].append(PlacedLine(text, placed, baseline))
    return Layout(pages, scale, fitted)


def _adjust_slant(strokes, degrees):
    """Apply a common horizontal shear around the line's estimated baseline.

    Negative adjustments move upper strokes left, reducing a rightward lean.
    This is relative to the original handwriting, not a measured final angle.
    The same map preserves coincident joins, vertical positions and pen lifts.
    """
    baseline = float(np.quantile(np.concatenate(strokes)[:,1], .8))
    shear = math.tan(math.radians(degrees))
    adjusted = []
    for stroke in strokes:
        moved = stroke.copy()
        moved[:,0] += shear*(baseline-moved[:,1])
        adjusted.append(moved)
    return adjusted


def _vary_word_gaps(strokes, text, amount, rng, min_gap_mm=0):
    """Shift whole ink groups across detected blank gaps, preserving their shape.

    Gap detection is geometric, not text-to-character alignment. Only clearly
    blank projected intervals are used; joined strokes are never split.
    """
    word_breaks = max(0, len(text.split())-1)
    if not word_breaks:
        return strokes
    intervals = sorted((float(s[:,0].min()),float(s[:,0].max())) for s in strokes if len(s))
    merged = []
    for low,high in intervals:
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low,high])
    pitch = (merged[-1][1]-merged[0][0])/max(len(text),1)
    gaps = [(a[1],b[0]) for a,b in zip(merged,merged[1:]) if b[0]-a[1] >= max(.3,pitch*.5)]
    gaps = sorted(sorted(gaps,key=lambda g:g[1]-g[0],reverse=True)[:word_breaks])
    # Widen only already detected blank intervals. This cannot recover spaces
    # inside a joined stroke or provide character/word alignment. The final
    # line fit can uniformly reduce these gaps when space is tight.
    shifts = [(end,max(min_gap_mm-(end-start),rng.uniform(-.45,.45)*amount*(end-start)))
              for start,end in gaps]
    result = []
    for stroke in strokes:
        moved = stroke.copy()
        for end,delta in shifts:
            moved[:,0] += (stroke[:,0] >= end)*delta
        result.append(moved)
    return result


def _vary_letters(strokes, baseline, text, amount, rng,
                  width_variation=.5, height_variation=.5, slant_variation=.5, rhythm_variation=.5):
    """Correlated local width, height and slant changes over ~1–2 letters.

    This is bounded augmentation of the learned writer's movements, not a
    character generator or a model of measured human motor variation. A shared
    smooth spatial map keeps coincident joins/dots aligned; no point jitter.
    """
    points = np.concatenate(strokes)
    left, right = points[:, 0].min(), points[:, 0].max()
    width = max(float(right-left), .01)
    pitch = width/max(len(text), 1)
    count = max(2, int(np.ceil(width/max(1.8, pitch*(1.85-rhythm_variation)))))
    step = width/count
    # Neighbouring movements share a tendency without repeating a sine wave.
    fields = rng.uniform(-1, 1, (4, count+1))
    for i in range(1, count+1):
        fields[:, i] = .3*fields[:, i-1] + .7*fields[:, i]
    speed = 1 + .36*amount*width_variation*fields[0]
    integral = np.r_[0., np.cumsum(step*(speed[:-1]+speed[1:])/2)]
    height = max(float(np.abs(points[:, 1]-baseline).max()), .1)
    shear = min(.26*amount*slant_variation, .12*step/height)
    varied = []
    for stroke in strokes:
        x = np.clip(stroke[:, 0]-left, 0, width)
        slot = np.minimum((x/step).astype(int), count-1)
        u = np.clip((x-slot*step)/step, 0, 1)
        blend = (1-np.cos(np.pi*u))/2
        local = fields[:, slot]*(1-blend) + fields[:, slot+1]*blend
        # Integrate positive writing pace to prevent reversing letter order.
        forward = integral[slot] + step*(speed[slot]*u +
            (speed[slot+1]-speed[slot])*(u/2-np.sin(np.pi*u)/(2*np.pi)))
        forward *= width/integral[-1]
        y = stroke[:, 1]-baseline
        varied.append(np.column_stack((left+forward+shear*local[2]*y,
            baseline+y*(1+.26*amount*height_variation*local[1])+.32*amount*rhythm_variation*local[3])))
    return varied


def _relax_line(strokes, baseline, y_top, settings, rng, end_rng=None):
    """Small, smooth writer movements in millimetres; never per-point noise.

    A shared spatial map keeps joins and pen lifts intact. Geometry is derived
    afresh from saved trajectories, independently of ink or rendering DPI.
    """
    amount = settings.naturalness
    points = np.concatenate(strokes)
    left, right = points[:, 0].min(), points[:, 0].max()
    width = max(float(right-left), 1.)
    start = settings.margin_mm + amount * 2*settings.line_start_variation * rng.uniform(0, 5.6)
    # Separate deterministic stream keeps other movement choices unchanged
    # when only the right-edge control changes. Only whole lines are fitted.
    end_inset = 0.
    if settings.line_end_variation:
        if end_rng is None:end_rng = np.random.default_rng(settings.layout_seed)
        end_inset = amount*2*settings.line_end_variation*end_rng.uniform(0,12)
    stretch = 1 + amount * 2*settings.rhythm_variation * rng.uniform(-.04, .04)
    shift = amount * 2*settings.line_spacing_variation * rng.uniform(-.6, .6)
    drift = amount * 2*settings.line_tilt_variation * rng.uniform(-.9, .9)
    slant = amount * 2*settings.slant_variation * rng.uniform(-.055, .055)
    phase, fine_phase = rng.uniform(0, 2*np.pi, 2)
    period = rng.uniform(10, 18)
    relaxed = []
    for stroke in strokes:
        x, y = stroke[:, 0]-left, stroke[:, 1]-baseline
        # Pace varies gradually along a line; this changes spacing by only a
        # few percent and keeps x monotonic for a fixed baseline height.
        pace = .035*amount*period/(2*np.pi) * (np.sin(x*2*np.pi/period+phase)-np.sin(phase))
        sway = amount*2*settings.baseline_variation*(.18*np.sin(x*2*np.pi/period+phase)+.055*np.sin(x*2*np.pi/3.8+fine_phase))
        relaxed.append(np.column_stack((start+x*stretch+pace+slant*y,
                                        baseline+y+shift+drift*x/width+sway)))
    return _fit_line(relaxed, baseline+shift, y_top, settings, end_inset)


def _fit_line(strokes, baseline, y_top, settings, end_inset=0):
    # Move the entire line inside its printable cell, without clipping strokes
    # or independently moving dots/crossbars. Very tall lines are fit as a unit.
    points = np.concatenate(strokes)
    low, high = points.min(0), points.max(0)
    bounds_low = np.array([settings.margin_mm, y_top+.25])
    bounds_high = np.array([210-settings.margin_mm-end_inset, y_top+settings.line_spacing_mm-.25])
    span = bounds_high-bounds_low
    if end_inset or settings.fit_lines_individually:
        # Preserve the chosen start, reserving space at the right edge. A
        # crowded line is scaled about its own start rather than shifted flush
        # left, which would erase the requested start-position variation.
        span[0] = bounds_high[0]-float(np.clip(low[0],bounds_low[0],bounds_high[0]-.1))
    fit = min(1., float(np.min(span/np.maximum(high-low, 1e-6))))
    anchor = np.array([low[0], baseline])
    relaxed = [(s-anchor)*fit+anchor for s in strokes]
    points = np.concatenate(relaxed)
    correction = np.maximum(bounds_low-points.min(0), 0) + np.minimum(bounds_high-points.max(0), 0)
    return [s+correction for s in relaxed], baseline+correction[1]


def _deposit(density, points, ink, rng, px_per_mm):
    if len(points) == 0:
        return
    tip = ink.width_mm * px_per_mm
    padding = math.ceil(tip * 2 + 3)
    low = np.floor(points.min(axis=0)).astype(int) - padding
    high = np.ceil(points.max(axis=0)).astype(int) + padding
    low = np.maximum(low, 0)
    high = np.minimum(high, [density.shape[1] - 1, density.shape[0] - 1])
    size = high - low + 1
    if np.any(size <= 0):
        return
    # Small supersampled masks retain subpixel widths, including 0.15 mm tips.
    ss = 2
    mask = Image.new("L", tuple((size * ss).tolist()), 0)
    draw = ImageDraw.Draw(mask)
    local = (points - low) * ss
    if len(points) > 1:
        speed = np.linalg.norm(np.diff(points, axis=0), axis=1)
        relative_speed = speed / max(float(np.median(speed)), 0.1)
    else:
        relative_speed = np.ones(1)
    phase = rng.uniform(0, 2 * np.pi)
    for i in range(max(1, len(points) - 1)):
        t = i / max(1, len(points) - 1)
        pressure = 1 + ink.pressure_variation * (0.18 * math.sin(t * 6 + phase) +
                                                0.12 * (1 - min(relative_speed[i], 2)))
        taper = 0.78 + 0.22 * min(1, t * 12 + 0.15, (1 - t) * 12 + 0.15)
        width = max(1, round(tip * pressure * taper * ss))
        amount = int(220 * (0.86 + 0.24 * ink.density_variation * math.sin(t * 11 + phase)))
        a = tuple(local[i])
        b = tuple(local[min(i + 1, len(points) - 1)])
        draw.line((a, b), fill=amount, width=width)
        radius = width / 2
        draw.ellipse((a[0] - radius, a[1] - radius, a[0] + radius, a[1] + radius), fill=amount)
    mask = mask.resize(tuple(size.tolist()), Image.Resampling.LANCZOS)
    coverage = np.asarray(mask, dtype=np.float32) / 255
    rough = ink.roughness
    if rough:
        grain = rng.uniform(1 - rough * 0.48, 1, coverage.shape).astype(np.float32)
        coarse = Image.fromarray(rng.integers(70, 256, (max(2, size[1] // 20), max(2, size[0] // 20)), dtype=np.uint8))
        coarse = np.asarray(coarse.resize(tuple(size.tolist()), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        coverage *= grain * (1 - rough * 0.3 + rough * 0.3 * coarse)
        # Rare dry-ink holes are stronger at low flow, not arbitrary broken geometry.
        dry = rng.random(coverage.shape) < rough * (1 - ink.flow) * 0.09
        coverage[dry] *= 0.2
    deposit = coverage * ink.flow * ink.opacity * 2.5
    density[low[1]:high[1] + 1, low[0]:high[0] + 1] += deposit


def render_page(lines: list[PlacedLine], page: PageSettings, ink: InkSettings,
                dpi: int = 120, page_index: int = 0) -> Image.Image:
    if dpi not in (96, 120, 150, 300):
        raise ValueError("Use 96, 120, 150, or 300 dpi.")
    px = dpi / 25.4
    width, height = round(210 * px), round(297 * px)
    density = np.zeros((height, width), dtype=np.float32)
    rng = np.random.default_rng(ink.seed + page_index)
    for line in lines:
        for stroke in line.strokes:
            _deposit(density, stroke * px, ink, rng, px)
    background = Image.new("RGB", (width, height), "white")
    if page.paper == "ruled":
        draw = ImageDraw.Draw(background)
        count = int((297 - 2 * page.margin_mm) // page.line_spacing_mm)
        for i in range(count):
            y = round((page.margin_mm + i * page.line_spacing_mm + page.line_spacing_mm * 0.78) * px)
            draw.line([(round(page.margin_mm * px), y), (round((210 - page.margin_mm) * px), y)],
                      fill=(207, 221, 233), width=max(1, round(px * 0.12)))
    # Optical density adds at overlaps before conversion to reflected light.
    alpha = 1 - np.exp(-density)
    color = np.array(ImageColor.getrgb(ink.color), dtype=np.float32)
    paper = np.asarray(background, dtype=np.float32)
    pixels = paper * (1 - alpha[..., None]) + color * alpha[..., None]
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def export_document(document: Document, page: PageSettings, ink: InkSettings, folder: Path,
                    dpi: int = 300) -> dict:
    """Render one page at a time to keep memory bounded on shared-RAM laptops."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    folder.mkdir(parents=True, exist_ok=True)
    layout = layout_document(document, page)
    pdf_path = folder / "inkform.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(210 * 72 / 25.4, 297 * 72 / 25.4))
    pdf.setTitle("InkForm handwriting")
    pdf.setAuthor("InkForm")
    pngs = []
    for index, lines in enumerate(layout.pages):
        image = render_page(lines, page, ink, dpi, index)
        path = folder / f"page-{index + 1:03d}.png"
        image.save(path, dpi=(dpi, dpi))
        pdf.drawImage(ImageReader(image), 0, 0, width=210 * 72 / 25.4, height=297 * 72 / 25.4)
        pdf.showPage()
        image.close()
        pngs.append(path.name)
    pdf.save()
    return {"pages": len(layout.pages), "dpi": dpi, "pdf": pdf_path.name,
            "png": pngs, "fitted_to_width": layout.fitted_to_width}
