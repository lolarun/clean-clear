"""End to end through pipeline.process with a fake OCR and a fake inpainting model, on a real encode."""
import numpy as np
import pytest

from clean_clear import pipeline
from clean_clear.cli import build_argparser
from clean_clear.masks import white_pixels
from clean_clear.video import read_frames
from conftest import probe_streams


def fake_ocr(strip):
    ys, xs = np.nonzero(white_pixels(strip))
    if len(ys) < 50:
        return []
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return [(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), "字幕", 0.95)]


class BlackFill:
    name = "fake"

    def __init__(self, fail_at=None):
        self.fail_at = fail_at

    def erase(self, frames, frame_seg, masks, cuts=()):
        for i, f in enumerate(frames):
            if i == self.fail_at:
                raise RuntimeError("model crashed")
            k = frame_seg.get(i)
            if k is not None:
                f = f.copy()
                f[masks[k]] = (160, 96, 48)
            yield f


def args(*extra):
    return build_argparser().parse_args(["--model", "lama", "--watermark", "off", "--stabilize", "0", *extra])


def test_subtitle_is_erased_and_extracted(subtitle_video, tmp_path):
    out = tmp_path / "out"
    pipeline.process(subtitle_video, out, args(), fake_ocr, BlackFill(), "libx264")
    dst = out / "film_clean.mp4"
    assert sorted(p.name for p in out.iterdir()) == [".cache", "film.srt", "film_clean.mp4"]
    assert (out / "film.srt").read_text(encoding="utf-8") == "1\n00:00:01,000 --> 00:00:03,000\n字幕\n\n"
    frames = list(read_frames(dst, 640, 360))
    assert len(frames) == 100
    assert not white_pixels(frames[50]).any()
    streams = probe_streams(dst)
    assert {s["codec_type"] for s in streams} == {"video", "audio"}
    # the cache is reused
    pipeline.process(subtitle_video, out, args("--verify", "off"), lambda s: pytest.fail("OCR ran again"), BlackFill(),
                     "libx264")


def test_a_failed_run_leaves_no_output_file(subtitle_video, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="model crashed"):
        pipeline.process(subtitle_video, out, args(), fake_ocr, BlackFill(fail_at=60), "libx264")
    assert not (out / "film_clean.mp4").exists()
    assert not [p for p in out.iterdir() if p.suffix == ".mp4"]


def test_verify_erases_what_the_first_pass_missed(subtitle_video, tmp_path):
    class MissFirst(BlackFill):
        calls = 0

        def erase(self, frames, frame_seg, masks, cuts=()):
            MissFirst.calls += 1
            if MissFirst.calls == 1:
                return iter(frames)  # first pass leaves everything
            return super().erase(frames, frame_seg, masks, cuts)

    out = tmp_path / "out"
    pipeline.process(subtitle_video, out, args(), fake_ocr, MissFirst(), "libx264")
    frames = list(read_frames(out / "film_clean.mp4", 640, 360))
    assert len(frames) == 100 and not white_pixels(frames[50]).any()
