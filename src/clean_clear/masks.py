"""Erase masks: one fixed mask per subtitle, either the text boxes or the glyph strokes."""
import cv2
import numpy as np

from .common import Progress, log
from .video import read_frames


def white_pixels(img):
    """Bright, low-saturation pixels (white subtitle glyphs)"""
    img = img.astype(np.int16)
    mn, mx = img.min(2), img.max(2)
    return (mn > 170) & (mx - mn < 50)


def box_mask(seg, H, W, dilate):
    """Union of all text boxes in a segment, expanded by `dilate` pixels"""
    m = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in seg.boxes:
        m[max(0, y0 - dilate):y1 + dilate, max(0, x0 - dilate):x1 + dilate] = True
    return m


def box_masks(segs, H, W, dilate):
    return {k: box_mask(s, H, W, dilate) for k, s in enumerate(segs)}


def _same_glyphs(w, ref, ref_near):
    """Is the white-pixel image `w` (cropped to a segment's rectangle) the segment's text again? Most of the
    segment's glyph pixels must be white, and there must be little white elsewhere (a different sentence
    in the same place would add as much white outside the old strokes as it covers inside them)."""
    g = np.count_nonzero(ref)
    return np.count_nonzero(w & ref) >= 0.6 * g and np.count_nonzero(w & ~ref_near) <= 0.5 * g


def glyph_masks(src, W, H, band, segs, dilate, grow, shift, total=0, extend=0, gap=2):
    """Glyph masks: within a segment, pixels that are white in more than half of the frames are text;
    they are then dilated to cover the outline and shadow. Inpainting only the strokes instead of the
    whole text box keeps far more original pixels, so the result is sharper, and the mask is fixed
    within a segment so it does not flicker. Segments whose text is not white fall back to the box
    mask. Returns {segment index: full-frame bool mask}.

    extend > 0: OCR sometimes loses a subtitle for part of its time (low confidence, busy background),
    which leaves the rest of the sentence on screen. Up to `extend` frames before/after each segment are
    checked against the segment's own glyph pixels; frames where the same glyphs are still visible
    (tolerating `gap` bad frames) widen the segment's erase range (Seg.ext_lo / Seg.ext_hi). This
    happens inside the same pass over the video, so it costs no extra decoding."""
    by0, by1 = band
    boxm = [box_mask(s, H, W, dilate)[by0:by1] for s in segs]
    # only count pixels inside each segment's bounding rectangle
    rects = []
    for m in boxm:
        ys, xs = np.nonzero(m)
        rects.append((ys.min(), ys.max() + 1, xs.min(), xs.max() + 1))
    seg_of = {i: k for k, s in enumerate(segs) for i in range(s.start, s.end + 1)}
    counts = {}

    # frame -> [(segment, "head"|"tail")] windows to check; neighbouring segments bound each window
    windows = {}
    if extend > 0:
        for k, s in enumerate(segs):
            lo = max(segs[k - 1].end + 1 if k else 0, s.start - extend)
            hi = min(segs[k + 1].start - 1 if k + 1 < len(segs) else s.end + extend, s.end + extend)
            for i in range(lo, s.start):
                windows.setdefault(i, []).append((k, "head"))
            for i in range(s.end + 1, hi + 1):
                windows.setdefault(i, []).append((k, "tail"))
    head = {}  # k -> [(frame, white rect)] of the head window, judged once the segment's glyphs are known
    tail = {}  # k -> [glyph reference, dilated reference, bad frames in a row]
    small = np.ones((5, 5), np.uint8)

    def finish(k):
        """The segment's last frame was read: its glyph reference is final; judge the head window, arm the tail"""
        s = segs[k]
        ry0, ry1, rx0, rx1 = rects[k]
        c = counts[k][ry0:ry1, rx0:rx1]
        ref = (c >= 0.5 * (s.end - s.start + 1)) & boxm[k][ry0:ry1, rx0:rx1]
        if ref.sum() < 0.03 * boxm[k].sum():  # no white glyphs: the segment uses a box mask, nothing to match against
            head.pop(k, None)
            return
        near = cv2.dilate(ref.astype(np.uint8), small).astype(bool)
        bad = 0
        for f, w in reversed(head.pop(k, [])):
            if _same_glyphs(w, ref, near):
                s.ext_lo, bad = f, 0
            else:
                bad += 1
                if bad > gap:
                    break
        tail[k] = [ref, near, 0]

    prog = Progress("masks", total or len(seg_of))
    for i, strip in enumerate(read_frames(src, W, H, band)):
        prog.update(i + 1)
        k = seg_of.get(i)
        if k is not None:
            ry0, ry1, rx0, rx1 = rects[k]
            c = counts.setdefault(k, np.zeros(boxm[k].shape, np.uint16))
            c[ry0:ry1, rx0:rx1] += white_pixels(strip[ry0:ry1, rx0:rx1]) & boxm[k][ry0:ry1, rx0:rx1]
            if i == segs[k].end and extend > 0:
                finish(k)
        for kw, kind in windows.get(i, ()):
            ry0, ry1, rx0, rx1 = rects[kw]
            w = white_pixels(strip[ry0:ry1, rx0:rx1])
            if kind == "head":
                head.setdefault(kw, []).append((i, w))
            elif kw in tail and tail[kw][2] <= gap:
                ref, near, _ = tail[kw]
                if _same_glyphs(w, ref, near):
                    segs[kw].ext_hi, tail[kw][2] = i, 0
                else:
                    tail[kw][2] += 1
    if extend > 0:
        n_lo = sum(s.ext_lo is not None for s in segs)
        n_hi = sum(s.ext_hi is not None for s in segs)
        frames = sum((s.start - s.span[0]) + (s.span[1] - s.end) for s in segs)
        log(f"  glyph matching widened {n_lo} subtitle starts and {n_hi} ends by {frames} frames in total")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    out = {}
    fallback = 0
    for k, s in enumerate(segs):
        m = np.zeros((H, W), bool)
        c = counts.get(k)
        white = (c >= 0.5 * (s.end - s.start + 1)).astype(np.uint8) if c is not None else None
        if white is None or white.sum() < 0.03 * boxm[k].sum():
            m[by0:by1] = boxm[k]
            fallback += 1
        else:
            g = cv2.dilate(white, kernel)
            if shift:  # subtitle shadows usually fall to the lower right
                g |= np.roll(np.roll(g, shift, 0), shift, 1)
            m[by0:by1] = (g > 0) & cv2.dilate(boxm[k].astype(np.uint8), kernel).astype(bool)
        out[k] = m
    if fallback:
        log(f"  {fallback}/{len(segs)} subtitles have no white glyphs; using box masks")
    return out


def frame_masks(segs, masks, pad):
    """Map every frame to be erased onto a mask, including `pad` extra frames before/after each segment.

    A frame covered by several segments (back-to-back subtitles, where one segment's padding
    overlaps the next segment) gets the union of their masks - otherwise the new subtitle would be
    left unmasked on that frame, and ProPainter would even propagate it into neighbouring frames.
    Returns ({frame index: mask key}, {mask key: HxW bool mask}); a key is a tuple of segment indices."""
    cover = {}
    for k, s in enumerate(segs):
        lo, hi = s.span
        for i in range(max(0, lo - pad), hi + pad + 1):
            cover.setdefault(i, []).append(k)
    frame_key = {i: tuple(ks) for i, ks in cover.items()}
    out = {}
    for key in set(frame_key.values()):
        out[key] = masks[key[0]] if len(key) == 1 else np.logical_or.reduce([masks[k] for k in key])
    return frame_key, out
