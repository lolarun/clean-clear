"""Temporal smoothing of inpainted areas.

ProPainter's fill of an erased area changes slightly from frame to frame even where the scene does not (measured
1.5-2.5x the source's own frame-to-frame change). At normal speed this is hardly visible, at 2x it reads as a
shimmer where the subtitle was. Each filled pixel is blended with the previous output frame, but only as much as
the untouched ring around the area is static: during camera or subject motion the blend fades out, so it does
not leave trails, and it restarts at every shot cut and every change of mask."""
import cv2
import numpy as np

from .common import log


def _geometry(mask, margin=24):
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + 1 + margin)
    x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + 1 + margin)
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    ell = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fill = cv2.dilate(m, ell(9)).astype(bool)  # what the backends paste: the mask grown by 4 px
    ring = cv2.dilate(m, ell(31)).astype(bool) & ~cv2.dilate(m, ell(13)).astype(bool)  # untouched surroundings
    return y0, y1, x0, x1, fill, ring


def stabilize(frames, frame_key, masks, cuts=(), strength=0.6, motion=6.0, label="fill"):
    """Yields `frames` with the filled pixels of every frame in `frame_key` blended towards the previous output.

    strength: weight of the previous frame on a perfectly static background (0 = off).
    motion: mean grey-level change of the ring between two frames at which the blend reaches zero."""
    if strength <= 0 or not frame_key:
        yield from frames
        return
    cuts = set(cuts)
    geo = {}
    prev = None  # (mask key, previous output crop as float32)
    used = 0
    for i, f in enumerate(frames):
        k = frame_key.get(i)
        if k is None:
            prev = None
            yield f
            continue
        if k not in geo:
            geo[k] = _geometry(masks[k]) if masks[k].any() else None
        g = geo[k]
        if g is None:
            yield f
            continue
        y0, y1, x0, x1, fill, ring = g
        cur = f[y0:y1, x0:x1].astype(np.float32)
        if prev is not None and prev[0] == k and i not in cuts and ring.any():
            change = float(np.abs(cur[ring] - prev[1][ring]).mean())
            w = strength * max(0.0, 1.0 - change / motion)
            if w > 0.01:
                cur[fill] = (1.0 - w) * cur[fill] + w * prev[1][fill]
                if not f.flags.writeable:
                    f = f.copy()
                f[y0:y1, x0:x1][fill] = np.clip(cur[fill] + 0.5, 0, 255).astype(np.uint8)
                used += 1
        prev = (k, cur)
        yield f
    log(f"  stabilize ({label}): smoothed {used} of {len(frame_key)} frames")
