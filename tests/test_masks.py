import numpy as np

from clean_clear import masks as M
from clean_clear.masks import MaskSet, box_masks, frame_masks, glyph_masks
from clean_clear.subtitles import Seg

H, W = 360, 640
BAND = (240, 360)


def seg(start, end, box=(250, 300, 390, 320)):
    return Seg(start, end, boxes=[box] * 3)


def strips(n, lit, colour=(255, 255, 255), rect=(60, 80, 250, 390)):
    """band strips; frames in `lit` show a bar at band rows/cols `rect`"""
    y0, y1, x0, x1 = rect
    for i in range(n):
        s = np.full((BAND[1] - BAND[0], W, 3), (160, 96, 48), np.uint8)
        if i in lit:
            s[y0:y1, x0:x1] = colour
        yield s


def test_maskset_crop_union_and_cache():
    ms = MaskSet((H, W))
    a = np.zeros((H, W), bool); a[10:20, 30:40] = True
    b = np.zeros((H, W), bool); b[15:25, 35:60] = True
    ms.add("a", 0, 0, a)
    ms.add("b", 0, 0, b)
    assert ms.crop("a")[2].shape == (10, 10)
    assert np.array_equal(ms["a"], a)
    assert ms["a"] is ms["a"] and not ms["a"].flags.writeable
    y0, x0, u = ms.union(["a", "b"])
    full = np.zeros((H, W), bool); full[y0:y0 + u.shape[0], x0:x0 + u.shape[1]] = u
    assert np.array_equal(full, a | b)
    assert set(ms) == {"a", "b"} and len(ms) == 2


def test_frame_masks_union_on_shared_frames():
    segs = [seg(0, 9, (100, 300, 200, 320)), seg(10, 19, (300, 300, 400, 320))]
    fs, ms = frame_masks(segs, box_masks(segs, H, W, 0), pad=1)
    assert fs[9] == (0, 1) and fs[10] == (0, 1)  # each segment's padding covers the other's first/last frame
    assert fs[5] == (0,) and fs[20] == (1,)
    assert ms[(0, 1)][310, 150] and ms[(0, 1)][310, 350]
    assert isinstance(ms, MaskSet)


def test_glyph_mask_covers_white_strokes_and_shadow(monkeypatch):
    monkeypatch.setattr(M, "read_frames", lambda *a, **k: strips(40, set(range(10, 30))))
    s = seg(10, 29, (245, 295, 395, 325))
    out = glyph_masks("x", W, H, BAND, [s], dilate=4, grow=3, shift=2)
    m = out[0]
    assert m[300:320, 250:390].all()  # strokes
    # outline (grow 3) on every side, plus the shadow (shift 2) towards the lower right only
    assert m[324, 320] and m[310, 394] and not m[296, 320] and not m[310, 246]
    assert not m[:BAND[0]].any() and not m[200:290].any()
    y0, x0, c = out.crop(0)
    assert c.size < 0.05 * H * W  # stored as a crop, not as a full frame


def test_glyph_mask_falls_back_to_box_for_coloured_text(monkeypatch):
    monkeypatch.setattr(M, "read_frames", lambda *a, **k: strips(40, set(range(10, 30)), colour=(0, 220, 255)))
    s = seg(10, 29, (250, 300, 390, 320))
    m = glyph_masks("x", W, H, BAND, [s], dilate=5, grow=3, shift=2)[0]
    assert m[295:325, 245:395].all() and m.sum() == 30 * 150


def test_glyph_matching_extends_lost_frames(monkeypatch):
    # the bar is visible 5-34, but OCR only found it on 10-29
    monkeypatch.setattr(M, "read_frames", lambda *a, **k: strips(45, set(range(5, 35))))
    s = seg(10, 29)
    glyph_masks("x", W, H, BAND, [s], dilate=4, grow=3, shift=2, total=45, extend=10)
    assert s.span == (5, 34)
