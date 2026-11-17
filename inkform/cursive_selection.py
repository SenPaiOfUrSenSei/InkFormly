"""Select reference-shaped cursive r joins in checked neural words."""

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from .joined_r import fit_joined_r
from .recognition import compare_readback
from .word_clarity import lexical_word, place_complete_word, word_regions


def _crossing_x(points, y, *, descending):
    """Horizontal position where a path crosses a vertical height."""
    if descending:
        crossings = (points[:-1, 1] < y) & (points[1:, 1] >= y)
    else:
        crossings = (points[:-1, 1] > y) & (points[1:, 1] <= y)
    indices = np.flatnonzero(crossings)
    if not len(indices):
        return None
    index = indices[0] if descending else indices[-1]
    start, end = points[index:index + 2]
    return start[0] + (y - start[1]) * (end[0] - start[0]) / (end[1] - start[1])


@dataclass(frozen=True)
class RContour:
    character: int
    entry: int
    crest: int
    foot: int
    exit: int
    score: float


def _cursive_r_contours(offsets, labels, word):
    """Score visible entry, crest, returning leg and distinct rising exit.

    Attention identifies possible crests. The geometry is read from the
    continuous pen stroke so it can include joins across adjacent letters.
    This is a form preference, independent of text recognition.
    """
    data = np.asarray(offsets, dtype=float)
    labels = np.asarray(labels)
    points = data[:, :2].cumsum(axis=0)
    stops = np.flatnonzero(data[:, 2] > .5)
    contours = []

    for index, letter in enumerate(word):
        if letter != "r":
            continue
        owned = np.flatnonzero(labels == index)
        best = None
        if len(owned) < 4 or np.any(np.diff(owned) != 1):
            contours.append(None)
            continue
        height = np.ptp(points[owned, 1])
        if height < 3:
            contours.append(None)
            continue
        for crest in owned[len(owned) // 3:]:
            if crest < 2 or crest >= len(points) - 2:
                continue
            x, y = points[crest]
            if y > min(points[crest - 1, 1], points[crest + 1, 1]):
                continue
            # Limit the search to the local pen stroke, bounded by real lifts.
            stroke = np.searchsorted(stops, crest)
            start = max(0, crest - 24, stops[stroke - 1] + 1 if stroke else 0)
            end = min(len(points), crest + 28, stops[stroke] + 1 if stroke < len(stops) else len(points))
            entry = np.flatnonzero(points[start:crest, 1] >= y + .53 * height)
            if not len(entry) or end <= crest + 2:
                continue
            entry = int(start + entry[-1])
            following = points[crest:end, 1]
            valleys = np.flatnonzero(
                (following[1:-1] > following[:-2])
                & (following[1:-1] >= following[2:])
                & (following[1:-1] >= y + .45 * height)
            )
            if not len(valleys):
                continue
            bottom = int(crest + 1 + valleys[0])
            depth = points[bottom, 1] - y
            if depth > 1.25 * height or labels[bottom] > index + 1:
                continue
            level = points[bottom, 1] - .28 * height
            rising_exit = np.flatnonzero(points[bottom + 1:end, 1] <= level)
            if not len(rising_exit):
                continue
            exit_index = int(bottom + 1 + rising_exit[0])
            rise = (x - points[entry, 0]) / depth
            back = (x - points[bottom, 0]) / depth
            advance = (points[exit_index, 0] - points[bottom, 0]) / depth
            if not (.22 < rise < 2.3 and -.02 < back < .75 and .12 < advance < 1.2):
                continue

            leg = points[crest:bottom + 1]
            length = np.linalg.norm(np.diff(leg, axis=0), axis=1).sum()
            efficiency = np.linalg.norm(leg[-1] - leg[0]) / length
            entry_x = _crossing_x(points[start:crest + 1], level, descending=False)
            leg_x = _crossing_x(leg, level, descending=True)
            exit_x = _crossing_x(points[bottom:exit_index + 1], level, descending=False)
            if entry_x is None or leg_x is None or exit_x is None or efficiency < .7:
                continue
            separation = (exit_x - leg_x) / depth
            if separation < 0 or leg_x < entry_x - .09 * depth:
                continue
            slope_penalty = .8 * abs(back - .25) + .23 * abs(rise - .95)
            score = .65 - slope_penalty + 1.45 * min(separation, .5) + .15 * (1 - efficiency)
            if score > 0 and (best is None or score > best.score):
                best = RContour(index, entry, int(crest), bottom, exit_index, float(score))
        contours.append(best)
    return contours


def cursive_r_scores(offsets, labels, word):
    return np.asarray([item.score if item is not None else 0.0
                       for item in _cursive_r_contours(offsets, labels, word)])


def _keep_word_in_place(line, region, shaped):
    """Apply a shape within a continuous word block without moving the word."""
    indices = region["source_indices"]
    start, stop = indices[0], indices[-1]
    if (indices != list(range(start, stop + 1)) or len(shaped) != len(indices)
            or (start and line[start - 1][2] < .5) or line[stop][2] < .5):
        return None
    result = np.asarray(line, dtype=float).copy()
    result[start:stop + 1] += np.asarray(shaped) - np.asarray(region["offsets"])
    return result.tolist(), {"start": start, "stop": stop, "new_points": len(shaped),
                             "outside_points_preserved": True, "in_place": True}


def select_cursive_words(generator, baseline, reference, bias, diversity, seed):
    """Try the reference join in the original word, then in intact new takes."""
    original, native = baseline
    if original.readback.status != "match":
        return baseline
    text = original.text
    _, labels = generator.t_evaluator.character_trace(text, original.offsets, reference)
    regions = word_regions(original.offsets, labels, text)
    if regions is None:
        return baseline

    word_info = original.t_formation or {}
    t_replacements = {item["word_index"] for item in word_info.get("word_placements", [])}
    reader = generator.recognizer_factory()
    current = original.offsets
    evidence = deepcopy(word_info.get("selected_words"))
    readback = original.readback
    changes = []
    attempts = 0
    for index in reversed(range(len(regions))):
        region = regions[index]
        word = region["text"]
        if "r" not in word or not region["ownership_verified"] or index in t_replacements:
            continue
        local_labels = [labels[i] - region["start"] for i in region["source_indices"]]
        original_proof = None
        caption = word.strip('.,;:!?"()[]')
        row = evidence["words"][index] if evidence is not None else None
        margin = 0.0 if row and row["target"] else .16
        # The original word is preferred because this preserves neighboring
        # letters exactly. New samples can offer clearer ownership and room.
        for number, delta in enumerate((0, 0, -.12, .10, -.05, 0, -.12)):
            sample_seed = original.seed
            offsets, candidate_labels = region["offsets"], local_labels
            if number:
                if attempts >= 18:
                    break
                attempts += 1
                sample_seed = (seed + number * 104729 + (index + 1) * 15485863) % (2**31 - 1)
                sample_bias = min(1.2, max(.5, bias + delta))
                sample_diversity = min(1., diversity + (.20 if number % 2 == 0 else 0.))
                take = generator._trajectory(word, sample_bias, sample_seed, reference, sample_diversity)
                if take is None:
                    continue
                offsets = take[0]
                _, candidate_labels = generator.t_evaluator.character_trace(word, offsets, reference)
            joined, contours = fit_joined_r(
                offsets, candidate_labels, word, _cursive_r_contours(offsets, candidate_labels, word))
            # Resampling a whole word can change its other r forms too. Only
            # take that route when every r in the new word has a clean fit.
            if not contours or (number and len(contours) != word.count("r")):
                continue
            placed = (place_complete_word(current, region, joined) if number
                      else _keep_word_in_place(current, region, joined))
            if placed is None:
                continue
            offsets, placement = placed
            if original_proof is None:
                original_proof = reader.evidence(region["offsets"], caption)
            start = placement["start"]
            proof = reader.evidence(offsets[start:start + placement["new_points"]], caption)
            if lexical_word(proof["recognized_text"]) != lexical_word(caption):
                continue
            if (not math.isfinite(proof["nll"]) or not math.isfinite(original_proof["nll"])
                    or proof["nll"] > original_proof["nll"] + margin):
                continue
            candidate_readback = compare_readback(text, reader.read(offsets))
            if candidate_readback.status != "match":
                continue
            current, readback = offsets, candidate_readback
            changes.append({
                "word": word, "word_index": index, "seed": sample_seed,
                "contours": contours, "placement": placement,
            })
            if row is not None:
                row.update(**proof, exact=True, boundary_verified=True)
            break

    if not changes:
        return baseline
    result = original.model_copy(deep=True)
    result.offsets = [(float(x), float(y), int(end)) for x, y, end in current]
    result.readback = readback
    result.r_selection = {"method": "reference-joined-r-v1", "words": changes}
    result.attempts += attempts
    if evidence is not None:
        targets = [row for row in evidence["words"] if row["target"]]
        neighbors = [row for row in evidence["words"] if not row["target"]]
        evidence.update(
            exact_words=sum(row["exact"] for row in targets),
            mean_nll=sum(row["nll"] for row in targets) / max(1, len(targets)),
            context_nll=sum(row["nll"] for row in neighbors) / max(1, len(neighbors)),
        )
        result.t_formation["selected_words"] = evidence

    native = np.asarray(result.offsets, dtype=float)
    native[:, :2] /= 20
    native[:, 1] *= -1
    return result, native.tolist()
