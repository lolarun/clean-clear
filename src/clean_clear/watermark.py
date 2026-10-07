"""Static watermark detection: pixels that keep the same colour in (almost) every sampled frame."""
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .common import log
from .video import _exe


def guard_fill(frames, mask, thresh=25.0, margin=20, dark=60.0):
    """Safety net for the watermark erase step: yields the frames, repairing implausible fills.

    In dark scenes the inpainting model was seen to fill the logo area with a bright blob for a few frames
    (a 'flash' in the corner). A correct fill continues its surroundings, so its mean brightness is close
    to that of a ring of untouched pixels around the mask. When the ring is dark (mean below `dark`) and
    the fill is brighter than it by more than `thresh` grey levels, the area is re-filled from the ring with
    a classic spatial inpaint (stable on the flat dark backgrounds where the model fails). Bright or textured
    backgrounds are left alone: there the model's fill is good and a spatial inpaint only smears it."""
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + 1 + margin)
    x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + 1 + margin)
    m = mask[y0:y1, x0:x1]
    m8 = m.astype(np.uint8)
    ell = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ring = cv2.dilate(m8, ell(31)).astype(bool) & ~cv2.dilate(m8, ell(9)).astype(bool)
    fix = cv2.dilate(m8, ell(5))
    if not ring.any():
        yield from frames
        return
    fixed = 0
    for f in frames:
        crop = f[y0:y1, x0:x1]
        grey = crop.mean(2)
        inside, around = float(grey[m].mean()), float(grey[ring].mean())
        # only the observed failure: a bright fill on a dark background. On bright or textured backgrounds the
        # model's fill is good and legitimately differs from the ring average (a symmetric test smeared it)
        if around < dark and inside - around > thresh:
            if not f.flags.writeable:
                f = f.copy()
            f[y0:y1, x0:x1] = cv2.inpaint(np.ascontiguousarray(f[y0:y1, x0:x1]), fix, 5, cv2.INPAINT_TELEA)
            fixed += 1
        yield f
    if fixed:
        log(f"  watermark guard repaired {fixed} frame(s) where the fill did not match its surroundings")


def spatial_fill(frames, mask, radius=3, margin=12):
    """Yields the frames with the (small) `mask` re-filled by a spatial inpaint. For pieces too small to be worth a
    video-model pass, e.g. a few logo pixels the first pass missed."""
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + 1 + margin)
    x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + 1 + margin)
    fix = cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    for f in frames:
        if not f.flags.writeable:
            f = f.copy()
        f[y0:y1, x0:x1] = cv2.inpaint(np.ascontiguousarray(f[y0:y1, x0:x1]), fix, radius, cv2.INPAINT_TELEA)
        yield f


def leftover_accents(src, W, H, duration, mask, samples=200, ring=11, frac=0.05, dark_frac=0.3, dark=60):
    """-> HxW bool mask of small saturated bright pieces of the logo (e.g. a green tick) lying just outside `mask`.

    They are semi-transparent, so their colour changes with the background and the 'same colour in 85 % of the
    frames' test of `detect` misses them. They stand out on dark backgrounds: in a ring around the logo, a pixel
    that is much brighter than the background in `dark_frac` of the sampled frames whose ring is dark (median grey
    < `dark`), or saturated and bright in
    `frac` of all sampled frames, is part of the logo, not of the scene. (Counting all frames only missed the tip of
    the tick in a film with mostly bright scenes: it was saturated in under 5 % of them.)"""
    ys, xs = np.nonzero(mask)
    y0, y1 = max(0, ys.min() - ring - 8), min(H, ys.max() + 1 + ring + 8)
    x0, x1 = max(0, xs.min() - ring - 8), min(W, xs.max() + 1 + ring + 8)
    m = mask[y0:y1, x0:x1]
    ring_m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))).astype(bool) & ~m
    times = np.linspace(duration * 0.03, duration * 0.97, samples)
    with ThreadPoolExecutor(4) as ex:
        frames = [f for f in ex.map(lambda t: _grab(src, t, W, H), times) if f is not None]
    if len(frames) < 20:
        return None
    hits = np.zeros(m.shape, np.int32)
    dark_hits, n_dark = np.zeros(m.shape, np.int32), 0
    for f in frames:
        crop = np.ascontiguousarray(f[y0:y1, x0:x1])
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hits += (hsv[:, :, 1] > 128) & (hsv[:, :, 2] > 100)
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bg = float(np.median(grey[ring_m]))
        if bg < dark:  # on a dark background any logo piece is simply much brighter than its surroundings
            dark_hits += grey > max(80.0, bg + 50.0)
            n_dark += 1
    acc = ring_m & (hits >= frac * len(frames))
    if n_dark >= 8:
        acc |= ring_m & (dark_hits >= dark_frac * n_dark)
    if not acc.any():
        return None
    out = np.zeros((H, W), bool)
    out[y0:y1, x0:x1] = cv2.dilate(acc.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~m
    return out


def _grab(src, t, W, H):
    r = subprocess.run([_exe("ffmpeg"), "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
    if len(r.stdout) != W * H * 3:
        return None
    return np.frombuffer(r.stdout, np.uint8).reshape(H, W, 3)


def detect(src, W, H, duration, samples=150, tol=14, consistency=0.85, min_area=20, grow=15, near_small=True):
    """-> HxW bool mask of the watermark(s) found in the four corners, or None.

    Real footage changes from frame to frame while an overlaid logo does not. Frames are sampled across
    the whole video; a pixel belongs to the watermark if it is within `tol` of its median colour in at
    least `consistency` of the samples and sits on an edge of the median image (which excludes flat
    black bars). `grow` is the dilation kernel size covering the anti-aliased outline; 15 (7 px) rather than
    9 because parts of a logo can show only now and then (the test film's green tick lit up in 4 of 136 dark samples,
    reaching 2-3 px past a 4 px margin) and no 'what stays the same' test can find them."""
    cw, ch = min(W // 3, 480), min(H // 4, 200)
    corners = [(0, 0), (W - cw, 0), (0, H - ch), (W - cw, H - ch)]  # top-left, top-right, bottom-left, bottom-right
    times = np.linspace(duration * 0.03, duration * 0.97, samples)
    with ThreadPoolExecutor(4) as ex:
        frames = list(ex.map(lambda t: _grab(src, t, W, H), times))
    frames = [f for f in frames if f is not None]
    if len(frames) < 20:
        log("  watermark: too few frames could be sampled")
        return None

    mask = np.zeros((H, W), np.uint8)
    for x, y in corners:
        stack = np.stack([f[y:y + ch, x:x + cw] for f in frames])
        k = len(stack) // 2
        med = np.partition(stack, k, axis=0)[k]
        near = (np.abs(stack.astype(np.int16) - med.astype(np.int16)).max(-1) < tol).mean(0)
        gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
        edge = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        cand = (near > consistency) & (cv2.dilate(edge, np.ones((3, 3), np.uint8)) > 25)
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        big = [i for i in range(1, n) if st[i, 4] >= min_area]
        near = cv2.dilate(np.isin(lab, big).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0             if big and near_small else None
        for i in range(1, n):
            # small static pieces right next to the logo (e.g. the green tick under it) belong to it
            if st[i, 4] >= min_area or (near is not None and st[i, 4] >= 3 and near[lab == i].any()):
                mask[y:y + ch, x:x + cw][lab == i] = 1
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    log(f"  watermark found: x={xs.min()}-{xs.max()} y={ys.min()}-{ys.max()}")
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))).astype(bool)
    if mask.mean() > 0.05:
        log("  watermark: detected area is more than 5% of the frame, ignoring it (probably not a logo)")
        return None
    return mask
