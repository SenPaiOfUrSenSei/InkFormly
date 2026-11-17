"""Fit the reference r between the writer's own entry and exit strokes."""

import numpy as np


def _unit(vector):
    return vector / max(np.linalg.norm(vector), 1e-8)


def _curve(knots, entry_direction, exit_direction, count):
    """Cubic interpolation with the original directions at both joins."""
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(knots, axis=0), axis=1))]
    tangent = np.empty_like(knots)
    tangent[1:-1] = (knots[2:] - knots[:-2]) / (distance[2:] - distance[:-2])[:, None]
    tangent[0] = entry_direction
    tangent[-1] = exit_direction
    # Tight turns define the top and the angled shoulder.
    tangent[2] *= .20
    tangent[3] *= .32
    samples = np.linspace(0, distance[-1], count)
    span = np.clip(np.searchsorted(distance, samples, side="right") - 1, 0, len(knots) - 2)
    step = (distance[span + 1] - distance[span])[:, None]
    u = (samples[:, None] - distance[span, None]) / step
    return ((2*u**3 - 3*u**2 + 1) * knots[span]
            + (u**3 - 2*u**2 + u) * step * tangent[span]
            + (-2*u**3 + 3*u**2) * knots[span + 1]
            + (u**3 - u**2) * step * tangent[span + 1])


def _crosses_ink(points, lifts, start, stop):
    """Test altered segments for crossings with other visible ink."""
    existing = np.flatnonzero(lifts[:-1] < .5)
    other_start, other_end = points[existing], points[existing + 1]

    def side(direction, difference):
        return direction[..., 0] * difference[..., 1] - direction[..., 1] * difference[..., 0]

    other_direction = other_end - other_start
    for i in range(start, stop):
        left, right = points[i:i + 2]
        direction = right - left
        first = side(direction, other_start - left)
        second = side(direction, other_end - left)
        third = side(other_direction, left - other_start)
        fourth = side(other_direction, right - other_start)
        if np.any((np.abs(existing - i) > 1) & (first * second < -1e-10) & (third * fourth < -1e-10)):
            return True
    return False


def fit_joined_r(offsets, labels, word, contours):
    """Fit a peaked shoulder and rounded foot, keeping the outer joins fixed."""
    data = np.asarray(offsets, dtype=float)
    points = data[:, :2].cumsum(axis=0)
    labels = np.asarray(labels)
    lifts = data[:, 2]
    fitted = points.copy()
    changes = []
    first_unused = 0

    for contour in contours:
        if contour is None:
            continue
        character = contour.character
        crest, foot = contour.crest, contour.foot
        if foot - crest < 6 or not np.all(labels[crest:foot + 1] == character):
            continue
        depth = points[foot, 1] - points[crest, 1]
        top, baseline = points[crest, 1], points[foot, 1]
        start, stop = contour.entry, contour.exit

        while (start > 0 and not lifts[start - 1] and labels[start - 1] >= character - 1
               and points[start - 1, 1] > points[start, 1]
               and points[start - 1, 0] < points[start, 0]
               and points[start - 1, 1] - top < .95 * depth):
            start -= 1
        while (stop < len(points) - 1 and not lifts[stop] and labels[stop + 1] <= character + 1
               and points[stop + 1, 1] < points[stop, 1]
               and points[stop + 1, 0] > points[stop, 0]
               and points[stop + 1, 1] - top > .07 * depth):
            stop += 1
        if start < first_unused or np.any(lifts[start:stop] > .5):
            continue

        first, last = points[start], points[stop]
        width = (last[0] - first[0]) / depth
        if not 1.05 <= width <= 2.8 or last[1] > baseline - .40 * depth:
            continue
        # A word-internal r needs continuous ink to both adjoining letters.
        # Word edges can have an open entry or a terminal flourish.
        before_lift = np.flatnonzero(lifts[:start])
        after_lift = np.flatnonzero(lifts[stop:])
        stroke_start = int(before_lift[-1]) + 1 if len(before_lift) else 0
        stroke_end = stop + int(after_lift[0]) + 1 if len(after_lift) else len(points)
        if (character > 0 and word[character - 1].isalpha()
                and character - 1 not in labels[stroke_start:crest]):
            continue
        if (character + 1 < len(word) and word[character + 1].isalpha()
                and character + 1 not in labels[foot:stroke_end]):
            continue

        peak_x = first[0] + .49 * (last[0] - first[0])
        shoulder = min(.21, width * .15) * depth
        knots = np.array([
            first, [peak_x - .20 * depth, top + .34 * depth], [peak_x, top],
            [peak_x + shoulder, top + .24 * depth],
            [peak_x - .012 * depth, baseline - .19 * depth], [peak_x + .11 * depth, baseline],
            [peak_x + .34 * depth, baseline - .11 * depth], last,
        ])
        if last[0] <= knots[-2, 0] or last[1] >= knots[-2, 1]:
            continue
        approach = points[start] - points[start - 1] if start > stroke_start else points[start + 1] - first
        departure = points[stop + 1] - last if stop + 1 < stroke_end else last - points[stop - 1]
        # A backwards tangent at either cut would wrap into adjacent lettering.
        if min(approach[0], departure[0]) <= 0:
            continue
        trial = fitted.copy()
        trial[start:stop + 1] = _curve(knots, _unit(approach), _unit(departure), stop - start + 1)
        if _crosses_ink(trial, lifts, start, stop):
            continue
        fitted = trial
        first_unused = stop + 1
        changes.append(dict(character=character, start=start, stop=stop))

    if not changes:
        return offsets, []
    result = data.copy()
    for change in changes:
        start, stop = change["start"], change["stop"]
        # The boundary coordinates stay exact. Only interior step offsets
        # change, and any points outside the selected joins stay in place.
        result[start + 1:stop + 1, :2] = np.diff(fitted[start:stop + 1], axis=0)
    return result.tolist(), changes
