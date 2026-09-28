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


def glyph_masks(src, W, H, band, segs, dilate, grow, shift, total=0):
    """Glyph masks: within a segment, pixels that are white in more than half of the frames are text;
    they are then dilated to cover the outline and shadow. Inpainting only the strokes instead of the
    whole text box keeps far more original pixels, so the result is sharper, and the mask is fixed
    within a segment so it does not flicker. Segments whose text is not white fall back to the box
    mask. Returns {segment index: full-frame bool mask}."""
    by0, by1 = band
    boxm = [box_mask(s, H, W, dilate)[by0:by1] for s in segs]
    # only count pixels inside each segment's bounding rectangle
    rects = []
    for m in boxm:
        ys, xs = np.nonzero(m)
        rects.append((ys.min(), ys.max() + 1, xs.min(), xs.max() + 1))
    seg_of = {i: k for k, s in enumerate(segs) for i in range(s.start, s.end + 1)}
    counts = {}
    prog = Progress("masks", total or len(seg_of))
    for i, strip in enumerate(read_frames(src, W, H, band)):
        prog.update(i + 1)
        k = seg_of.get(i)
        if k is not None:
            ry0, ry1, rx0, rx1 = rects[k]
            c = counts.setdefault(k, np.zeros(boxm[k].shape, np.uint16))
            c[ry0:ry1, rx0:rx1] += white_pixels(strip[ry0:ry1, rx0:rx1]) & boxm[k][ry0:ry1, rx0:rx1]
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
        for i in range(max(0, s.start - pad), s.end + pad + 1):
            cover.setdefault(i, []).append(k)
    frame_key = {i: tuple(ks) for i, ks in cover.items()}
    out = {}
    for key in set(frame_key.values()):
        out[key] = masks[key[0]] if len(key) == 1 else np.logical_or.reduce([masks[k] for k in key])
    return frame_key, out
