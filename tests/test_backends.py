"""The backends' streaming logic with fake models (no PyTorch / ONNX needed)."""
import cv2
import numpy as np
import pytest

from clean_clear.backends.lama import LamaBackend
from clean_clear.backends.propainter import ProPainterBackend
from clean_clear.masks import box_masks, frame_masks
from clean_clear.subtitles import Seg

H, W = 360, 640


def frames(n):
    for i in range(n):
        f = np.full((H, W, 3), i % 200, np.uint8)
        f.flags.writeable = False  # like read_frames
        yield f


def subtitle_masks():
    segs = [Seg(10, 59, boxes=[(250, 300, 390, 320)]), Seg(60, 99, boxes=[(200, 300, 300, 320)]),
            Seg(150, 152, boxes=[(250, 300, 390, 320)])]
    return frame_masks(segs, box_masks(segs, H, W, 4), pad=1)


class FakeEngine:
    def __init__(self):
        self.calls = []

    def __call__(self, fr, ms):
        self.calls.append((len(fr), fr[0].shape))
        assert all(f.shape[0] % 8 == 0 and f.shape[1] % 8 == 0 for f in fr)
        return [np.full_like(f, 255) for f in fr]


def propainter(chunk=40):
    b = ProPainterBackend.__new__(ProPainterBackend)
    b.chunk, b.CTX, b.PAD, b.margin = chunk, 5, 4, 32
    b.engine = FakeEngine()
    return b


@pytest.mark.parametrize("make", [propainter, lambda: LamaBackend.__new__(LamaBackend)])
def test_backends_stream_every_frame_in_order_and_touch_only_masks(make):
    b = make()
    if isinstance(b, LamaBackend):
        class FakeLama:
            SIZE = 512

            def __call__(self, img, m):
                return np.full_like(img, 255)
        b.lama = FakeLama()
    frame_seg, masks = subtitle_masks()
    cuts = [30, 151]
    out = list(b.erase(frames(200), frame_seg, masks, cuts))
    assert len(out) == 200
    for i, f in enumerate(out):
        k = frame_seg.get(i)
        m = masks[k] if k is not None else np.zeros((H, W), bool)
        assert (f[m] == 255).all(), i
        near = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)  # ProPainter pastes mask + 4 px
        assert (f[~near] == i % 200).all(), i  # elsewhere the frame is untouched


def test_propainter_chunks_never_cross_a_cut():
    b = propainter()
    frame_seg, _ = subtitle_masks()
    cuts = [30, 151]
    for s, e, cs, ce in b._chunks(frame_seg, cuts):
        shot = lambda i: sum(c <= i for c in cuts)
        assert shot(cs) == shot(ce) and cs <= s <= e <= ce
    covered = {i for s, e, _, _ in b._chunks(frame_seg, cuts) for i in range(s, e + 1)}
    assert set(frame_seg) <= covered


def test_propainter_tolerates_a_frame_map_past_the_end():
    b = propainter()
    m = np.zeros((H, W), bool)
    m[10:40, 10:60] = True
    out = list(b.erase(frames(50), dict.fromkeys(range(300), 0), {0: m}))
    assert len(out) == 50 and all((f[m] == 255).all() for f in out)
