"""Erase masks: one fixed mask per subtitle, either the text boxes or the glyph strokes."""
from collections import OrderedDict
from collections.abc import Mapping

import cv2
import numpy as np

from .common import Progress, log
from .subtitles import STYLE_DROP
from .video import read_frames


def white_pixels(img):
    """Bright, low-saturation pixels (white subtitle glyphs)"""
    img = img.astype(np.int16)
    mn, mx = img.min(2), img.max(2)
    return (mn > 170) & (mx - mn < 50)


def dark_pixels(img):
    """Dark pixels: the outline around white subtitle glyphs"""
    return img.max(2) < 80


_RING5 = np.ones((5, 5), np.uint8)


def outline_score(img):
    """How much the text in `img` (an OCR box) looks like a subtitle: the share of the surroundings of its white
    pixels that is dark. 1304: subtitles (white with a black outline) 0.43-0.9, also over a white shirt or paper;
    the title in red, opening credits, a sign, print on a box and a T-shirt 0-0.09."""
    w = white_pixels(img)
    if np.count_nonzero(w) < 20:
        return 0.0
    ring = cv2.dilate(w.astype(np.uint8), _RING5).astype(bool) & ~w
    return float(np.count_nonzero(dark_pixels(img) & ring) / max(1, np.count_nonzero(ring)))


def _text_height(seg):
    return float(np.median([b[3] - b[1] for b in seg.boxes]))


def thin_strokes(white, text_h):
    """`white` (bool crop) without its thick blobs. Glyph strokes are thin (half a stroke width is about 0.05-0.1 of
    the text height); a white sheet of paper, a white shirt or a bright wall behind the subtitle is white in every
    frame as well and was erased as "text", which regenerated the paper and left feathery textures (1304 at 06:14
    and 06:59). Pixels within the radius of a blob whose inscribed radius exceeds 0.15 text heights are dropped."""
    if not white.any():
        return white
    r = max(3, int(round(0.15 * text_h)))
    dist = cv2.distanceTransform(white.astype(np.uint8), cv2.DIST_L2, 3)
    core = dist > r
    if not core.any():
        return white
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 3, 2 * r + 3))
    return white & ~cv2.dilate(core.astype(np.uint8), k).astype(bool)


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


def _same_glyphs(w, ref, ref_near, dark=None, outline=None):
    """Is the white-pixel image `w` (cropped to a segment's rectangle) the segment's text again? Most of the
    segment's glyph pixels must be white, and there must be little white elsewhere (a different sentence
    in the same place would add as much white outside the old strokes as it covers inside them)."""
    g = np.count_nonzero(ref)
    if outline is not None and dark is not None and np.count_nonzero(dark & outline) < 0.5 * np.count_nonzero(outline):
        return False  # the strokes' dark outline is gone: white paper or a shirt under the old glyphs, not the text
    return np.count_nonzero(w & ref) >= 0.6 * g and np.count_nonzero(w & ~ref_near) <= 0.5 * g


def glyph_masks(src, W, H, band, segs, dilate, grow, shift, total=0, extend=0, gap=2, outlined=False):
    """Glyph masks: within a segment, pixels that are white in more than half of the frames are text;
    they are then dilated to cover the outline and shadow. Inpainting only the strokes instead of the
    whole text box keeps far more original pixels, so the result is sharper, and the mask is fixed
    within a segment so it does not flicker. Segments whose text is not white fall back to the box
    mask. Masks are keyed by segment index.

    extend > 0: OCR sometimes loses a subtitle for part of its time (low confidence, busy background),
    which leaves the rest of the sentence on screen. Up to `extend` frames before/after each segment are
    checked against the segment's own glyph pixels; frames where the same glyphs are still visible
    (tolerating `gap` bad frames) widen the segment's erase range (Seg.ext_lo / Seg.ext_hi). This
    happens inside the same pass over the video, so it costs no extra decoding. Returns a MaskSet.

    Seg.gaps: frames where OCR lost the text between two reads of the same subtitle (subtitles.build_segments joined
    them). They are neither counted for the glyphs nor erased unless they show a subtitle: with outlined=True (white
    subtitles with a dark outline) a gap frame whose box area scores below STYLE_DROP (outline_score) goes to Seg.skip
    (1304 at 12:29: text behind a sliding sign stayed on screen, but "卖什么油桃" ... "还卖什么油桃" at 2:04 have 0.8 s
    of plain picture between them)."""
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
    gap_of = {i: k for k, s in enumerate(segs) if rects[k] for a, b in s.gaps for i in range(a, b + 1)}
    seg_of = {i: k for k, s in enumerate(segs) if rects[k] for i in range(s.start, s.end + 1) if i not in gap_of}
    # frames that vote for a segment's glyphs: its own, without the gaps (more than half of THOSE must show a stroke)
    n_vote = {k: sum(1 for i in range(s.start, s.end + 1) if i not in gap_of) for k, s in enumerate(segs)}
    gap_score = {}  # k -> {frame: outline score of the box area}
    own_score = {}  # k -> [the same on the frames of the segment itself], for the segments that have gaps
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
    head = {}  # k -> [(frame, white rect, dark rect)] of the head window, judged once the segment's glyphs are known
    tail = {}  # k -> [glyph reference, dilated reference, bad frames in a row, outline reference]
    small = np.ones((5, 5), np.uint8)
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dcounts = {}  # k -> per-pixel count of dark frames (the outline of the glyphs)
    refs = {}

    def glyph_ref(k):
        """Glyph pixels of segment k: white in more than half of its frames, thin strokes only"""
        if k not in refs:
            s = segs[k]
            refs[k] = thin_strokes((counts[k] >= 0.5 * n_vote[k]) & boxc[k], _text_height(s))
        return refs[k]

    def finish(k):
        """The segment's last frame was read: its glyph reference is final; judge the head window, arm the tail"""
        s = segs[k]
        ref = glyph_ref(k)
        if ref.sum() < 0.03 * boxc[k].sum():  # no white glyphs: the segment uses a box mask, nothing to match against
            head.pop(k, None)
            return
        near = cv2.dilate(ref.astype(np.uint8), small).astype(bool)
        ring = cv2.dilate(ref.astype(np.uint8), ring_k).astype(bool) & ~ref & boxc[k]
        outline = None
        if ring.any() and k in dcounts:
            dark_ref = (dcounts[k] >= 0.5 * n_vote[k]) & ring
            if dark_ref.sum() >= 0.3 * ring.sum():  # the subtitle has a dark outline
                outline = dark_ref
        bad = 0
        for f, w, d in reversed(head.pop(k, [])):
            if _same_glyphs(w, ref, near, d, outline):
                s.ext_lo, bad = f, 0
            else:
                bad += 1
                if bad > gap:
                    break
        tail[k] = [ref, near, 0, outline]

    prog = Progress("masks", total or len(seg_of))
    for i, strip in enumerate(read_frames(src, W, H, band)):
        prog.update(i + 1)
        kg = gap_of.get(i)
        if kg is not None and outlined:
            ry0, ry1, rx0, rx1 = rects[kg]
            gap_score.setdefault(kg, {})[i] = outline_score(strip[ry0:ry1, rx0:rx1])
        k = seg_of.get(i)
        if k is not None:
            ry0, ry1, rx0, rx1 = rects[k]
            crop = strip[ry0:ry1, rx0:rx1]
            if outlined and segs[k].gaps and (i - segs[k].start) % 3 == 0:
                own_score.setdefault(k, []).append(outline_score(crop))
            c = counts.setdefault(k, np.zeros(boxc[k].shape, np.uint16))
            c += white_pixels(crop) & boxc[k]
            dcounts.setdefault(k, np.zeros(boxc[k].shape, np.uint16))
            dcounts[k] += dark_pixels(crop) & boxc[k]
            if i == segs[k].end and extend > 0:
                finish(k)
        for kw, kind in windows.get(i, ()):
            ry0, ry1, rx0, rx1 = rects[kw]
            crop = strip[ry0:ry1, rx0:rx1]
            w = white_pixels(crop)
            d = dark_pixels(crop)
            if kind == "head":
                head.setdefault(kw, []).append((i, w, d))
            elif kw in tail and tail[kw][2] <= gap:
                ref, near, _, outline = tail[kw]
                if _same_glyphs(w, ref, near, d, outline):
                    segs[kw].ext_hi, tail[kw][2] = i, 0
                else:
                    tail[kw][2] += 1
    for k, d in gap_score.items():
        # a gap frame without the outline is a frame without the subtitle only when the subtitle shows it elsewhere: one that
        # scores low itself (a bright picture) cannot be told from plain picture by the score, and is erased in the gap too
        own = float(np.median(own_score[k])) if own_score.get(k) else 0.0
        segs[k].skip = {i for i, sc in d.items() if sc < STYLE_DROP and sc < 0.5 * own}
    if any(s.gaps for s in segs):
        n = sum(b - a + 1 for s in segs for a, b in s.gaps)
        log(f"  OCR lost the text for {n} frames inside subtitles; {sum(len(s.skip) for s in segs)} of them show no subtitle")
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
        white = glyph_ref(k).astype(np.uint8) if c is not None else None
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
            if i not in s.skip:
                cover.setdefault(i, []).append(k)
    frame_key = {i: tuple(ks) for i, ks in cover.items()}
    if not isinstance(masks, MaskSet):
        masks = MaskSet.from_full(masks)
    out = MaskSet(masks.shape)
    for key in set(frame_key.values()):
        out.add(key, *(masks.crop(key[0]) if len(key) == 1 else masks.union(key)))
    return frame_key, out
