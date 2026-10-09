"""Erase masks: one fixed mask per subtitle, either the text boxes or the glyph strokes."""
from collections import OrderedDict
from collections.abc import Mapping

import cv2
import numpy as np

from .common import Progress, log
from .video import read_frames


def white_pixels(img):
    """Bright, low-saturation pixels (white subtitle glyphs)"""
    img = img.astype(np.int16)
    mn, mx = img.min(2), img.max(2)
    return (mn > 170) & (mx - mn < 50)


class MaskSet(Mapping):
    """{key: HxW bool mask}, stored as the crop around each mask's pixels.

    A feature film has ~1,500 subtitles; full-frame masks for all of them (plus the per-pixel counts used to build
    them) took ~8 GB at 1080p. Crops take a few hundred KB each at most. Indexing builds the full-frame array on demand
    and keeps the most recently used ones, so a backend that asks for the same key frame after frame gets the same
    (read-only) array object back."""

    def __init__(self, shape, cache=16):
        self.shape = tuple(shape)
        self._crops = {}
        self._full = OrderedDict()
        self._cache = cache

    @classmethod
    def from_full(cls, masks):
        out = None
        for k, m in masks.items():
            if out is None:
                out = cls(m.shape)
            out.add(k, 0, 0, m)
        return out if out is not None else cls((0, 0))

    def add(self, key, y0, x0, crop):
        """Store `crop` (bool) placed at row y0, column x0; trimmed to its pixels"""
        crop = np.asarray(crop, bool)
        ys, xs = np.nonzero(crop.any(1))[0], np.nonzero(crop.any(0))[0]
        if len(ys):
            crop = crop[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1].copy()
            y0, x0 = y0 + int(ys[0]), x0 + int(xs[0])
        else:
            crop, y0, x0 = np.zeros((0, 0), bool), 0, 0
        self._crops[key] = (y0, x0, crop)
        self._full.pop(key, None)

    def crop(self, key):
        """-> (y0, x0, bool crop) of the mask"""
        return self._crops[key]

    def union(self, keys):
        """-> (y0, x0, bool crop) of the union of the masks `keys`"""
        parts = [self._crops[k] for k in keys if self._crops[k][2].size]
        if not parts:
            return 0, 0, np.zeros((0, 0), bool)
        y0 = min(p[0] for p in parts)
        x0 = min(p[1] for p in parts)
        y1 = max(p[0] + p[2].shape[0] for p in parts)
        x1 = max(p[1] + p[2].shape[1] for p in parts)
        out = np.zeros((y1 - y0, x1 - x0), bool)
        for py, px, c in parts:
            out[py - y0:py - y0 + c.shape[0], px - x0:px - x0 + c.shape[1]] |= c
        return y0, x0, out

    def __getitem__(self, key):
        m = self._full.get(key)
        if m is not None:
            self._full.move_to_end(key)
            return m
        y0, x0, c = self._crops[key]
        m = np.zeros(self.shape, bool)
        m[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
        m.flags.writeable = False
        self._full[key] = m
        if len(self._full) > self._cache:
            self._full.popitem(last=False)
        return m

    def __iter__(self):
        return iter(self._crops)

    def __len__(self):
        return len(self._crops)


def _box_rect(seg, H, W, dilate, y_lo=0, y_hi=None):
    """-> (y0, y1, x0, x1, bool crop) of the union of a segment's text boxes expanded by `dilate`, rows limited to
    y_lo..y_hi; None if nothing is left"""
    y_hi = H if y_hi is None else y_hi
    rs = [(max(y_lo, y0 - dilate), min(y_hi, y1 + dilate), max(0, x0 - dilate), min(W, x1 + dilate))
          for x0, y0, x1, y1 in seg.boxes]
    rs = [r for r in rs if r[0] < r[1] and r[2] < r[3]]
    if not rs:
        return None
    ry0, ry1 = min(r[0] for r in rs), max(r[1] for r in rs)
    rx0, rx1 = min(r[2] for r in rs), max(r[3] for r in rs)
    c = np.zeros((ry1 - ry0, rx1 - rx0), bool)
    for y0, y1, x0, x1 in rs:
        c[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = True
    return ry0, ry1, rx0, rx1, c


def box_mask(seg, H, W, dilate):
    """Union of all text boxes in a segment, expanded by `dilate` pixels"""
    m = np.zeros((H, W), bool)
    r = _box_rect(seg, H, W, dilate)
    if r:
        m[r[0]:r[1], r[2]:r[3]] = r[4]
    return m


def box_masks(segs, H, W, dilate):
    out = MaskSet((H, W))
    for k, s in enumerate(segs):
        r = _box_rect(s, H, W, dilate)
        if r:
            out.add(k, r[0], r[2], r[4])
        else:
            out.add(k, 0, 0, np.zeros((0, 0), bool))
    return out


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
    mask. Masks are keyed by segment index.

    extend > 0: OCR sometimes loses a subtitle for part of its time (low confidence, busy background),
    which leaves the rest of the sentence on screen. Up to `extend` frames before/after each segment are
    checked against the segment's own glyph pixels; frames where the same glyphs are still visible
    (tolerating `gap` bad frames) widen the segment's erase range (Seg.ext_lo / Seg.ext_hi). This
    happens inside the same pass over the video, so it costs no extra decoding. Returns a MaskSet."""
    by0, by1 = band
    # each segment's box rectangle in band coordinates and its box mask cropped to it; pixels are only counted there
    rects, boxc = [], []
    for s in segs:
        r = _box_rect(s, H, W, dilate, by0, by1)
        if r is None:
            rects.append(None)
            boxc.append(None)
        else:
            rects.append((r[0] - by0, r[1] - by0, r[2], r[3]))
            boxc.append(r[4])
    seg_of = {i: k for k, s in enumerate(segs) if rects[k] for i in range(s.start, s.end + 1)}
    counts = {}

    # frame -> [(segment, "head"|"tail")] windows to check; neighbouring segments bound each window
    windows = {}
    if extend > 0:
        for k, s in enumerate(segs):
            if rects[k] is None:
                continue
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
        ref = (counts[k] >= 0.5 * (s.end - s.start + 1)) & boxc[k]
        if ref.sum() < 0.03 * boxc[k].sum():  # no white glyphs: the segment uses a box mask, nothing to match against
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
            c = counts.setdefault(k, np.zeros(boxc[k].shape, np.uint16))
            c += white_pixels(strip[ry0:ry1, rx0:rx1]) & boxc[k]
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
    out = MaskSet((H, W))
    fallback = 0
    e = grow + max(0, shift)  # how far the mask can reach beyond the box rectangle
    for k, s in enumerate(segs):
        if rects[k] is None:
            out.add(k, 0, 0, np.zeros((0, 0), bool))
            continue
        ry0, ry1, rx0, rx1 = rects[k]
        c = counts.get(k)
        white = (c >= 0.5 * (s.end - s.start + 1)).astype(np.uint8) if c is not None else None
        if white is None or white.sum() < 0.03 * boxc[k].sum():
            out.add(k, by0 + ry0, rx0, boxc[k])
            fallback += 1
            continue
        ey0, ey1, ex0, ex1 = max(0, ry0 - e), min(by1 - by0, ry1 + e), max(0, rx0 - e), min(W, rx1 + e)
        sl = (slice(ry0 - ey0, ry1 - ey0), slice(rx0 - ex0, rx1 - ex0))
        wexp = np.zeros((ey1 - ey0, ex1 - ex0), np.uint8)
        bexp = np.zeros_like(wexp)
        wexp[sl], bexp[sl] = white, boxc[k]
        g = cv2.dilate(wexp, kernel)
        if shift:  # subtitle shadows usually fall to the lower right
            g[shift:, shift:] |= g[:-shift, :-shift].copy()
        out.add(k, by0 + ey0, ex0, (g > 0) & cv2.dilate(bexp, kernel).astype(bool))
    if fallback:
        log(f"  {fallback}/{len(segs)} subtitles have no white glyphs; using box masks")
    return out


def frame_masks(segs, masks, pad):
    """Map every frame to be erased onto a mask, including `pad` extra frames before/after each segment.

    A frame covered by several segments (back-to-back subtitles, where one segment's padding
    overlaps the next segment) gets the union of their masks - otherwise the new subtitle would be
    left unmasked on that frame, and ProPainter would even propagate it into neighbouring frames.
    Returns ({frame index: mask key}, MaskSet {mask key: HxW bool mask}); a key is a tuple of segment indices."""
    cover = {}
    for k, s in enumerate(segs):
        lo, hi = s.span
        for i in range(max(0, lo - pad), hi + pad + 1):
            cover.setdefault(i, []).append(k)
    frame_key = {i: tuple(ks) for i, ks in cover.items()}
    if not isinstance(masks, MaskSet):
        masks = MaskSet.from_full(masks)
    out = MaskSet(masks.shape)
    for key in set(frame_key.values()):
        out.add(key, *(masks.crop(key[0]) if len(key) == 1 else masks.union(key)))
    return frame_key, out
