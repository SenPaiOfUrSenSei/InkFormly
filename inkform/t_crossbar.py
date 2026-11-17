"""Raise confidently identified existing t crossbars without adding strokes."""
import numpy as np


def raise_t_crossbars(native, stem_audits, strength=.85):
    values = np.asarray(native, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError('Expected finite pen offsets.')
    if not 0 <= strength <= 1:
        raise ValueError('Crossbar strength must be between zero and one.')
    xy = np.cumsum(values[:, :2], axis=0)
    result = xy.copy()
    occupied = set()
    audits = []
    for stem in stem_audits:
        peak, end = stem['stem_peak'], stem['downstroke_end']
        descending = xy[peak:end+1]
        order = np.argsort(descending[:, 1])
        sy, sx = descending[order, 1], descending[order, 0]
        keep = np.r_[True, np.diff(sy) > 1e-8]
        sy, sx = sy[keep], sx[keep]
        low, high = float(sy[0]), float(sy[-1])
        height = high-low
        if height <= .5:
            continue
        candidates = []
        # A crossing must be separate from the stem and cross both sides.
        for width in (5, 9, 13):
            for start in range(end+1, min(len(xy)-width, end+110)):
                stop = start+width
                if np.any(values[start:stop, 2] > .5):
                    continue
                segment = xy[start:stop+1]
                dx, dy = segment[-1]-segment[0]
                if dx < .32*height or abs(dy) > .35*dx:
                    continue
                y = float(np.median(segment[:, 1]))
                relative = (y-low)/height
                if not .15 < relative < .68 or np.ptp(segment[:, 1]) > .22*height:
                    continue
                x = float(np.interp(y, sy, sx))
                if not segment[:, 0].min() < x-.06*height < x+.06*height < segment[:, 0].max():
                    continue
                crossing = start+int(np.abs(segment[:, 0]-x).argmin())
                candidates.append((abs(dy)/dx+.002*(start-end), start, stop, crossing, relative))
        if not candidates:
            continue
        _, start, stop, crossing, relative = min(candidates)
        # Extend along the existing near-horizontal motion, stopping before a
        # turn into a following letter. Never cross a pen lift or the t stem.
        while start > end+1 and values[start-1, 2] <= .5:
            dx, dy = xy[start]-xy[start-1]
            if dx <= 0 or abs(dy) > .65*dx or xy[crossing, 0]-xy[start-1, 0] > height:
                break
            start -= 1
        while stop+1 < len(xy) and values[stop, 2] <= .5:
            dx, dy = xy[stop+1]-xy[stop]
            if dx <= 0 or abs(dy) > .65*dx or xy[stop+1, 0]-xy[crossing, 0] > height:
                break
            stop += 1
        indices = np.arange(start, stop+1)
        if occupied.intersection(indices) or crossing-start < 2 or stop-crossing < 2:
            continue
        amount = strength*min(.22, .68-relative)*height
        if amount < .025*height:
            continue
        # Keep endpoint joins fixed. A flat central region retains crossbar
        # shape, with smooth transitions confined to its two outer ends.
        u = np.arange(len(indices), dtype=float)
        edge = max(2., min(crossing-start, stop-crossing)*.65)
        envelope = .5-.5*np.cos(np.pi*np.clip(np.minimum(u,u[-1]-u)/edge,0,1))
        shift = amount*envelope
        # Follow the t's own lean so the raised dash still intersects its stem.
        lean = float(np.interp(low+(relative+amount/height)*height,sy,sx)-np.interp(low+relative*height,sy,sx))
        result[indices, 1] += shift
        result[indices, 0] += lean*envelope
        occupied.update(indices)
        audits.append(dict(character_index=stem['character_index'],start=int(start),stop=int(stop),
            crossing=int(crossing),height_before=relative,height_after=relative+amount/height,
            maximum_rise=float(amount),changed_points=[int(i) for i,s in zip(indices,shift) if s>1e-9]))
    out = values.copy()
    if audits:
        out[:, :2] = np.diff(np.vstack((np.zeros((1,2)),result)),axis=0)
    return out.tolist(), audits
