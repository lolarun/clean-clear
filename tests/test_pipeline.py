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


def bar_ocr(strip):
    """Reads only the white bar drawn over the SMPTE bars (280 x 36 px), not the white blocks of the bars themselves: a
    "subtitle" 218 px high is no subtitle line (subtitles.drop_phantoms)"""
    import cv2
    count, _, stats, _ = cv2.connectedComponentsWithStats(white_pixels(strip).astype(np.uint8))
    return [(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), "字幕", 0.95)
            for x, y, w, h, _ in stats[1:] if 250 <= w <= 310 and 30 <= h <= 42]


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


def test_the_verify_pass_reads_with_the_smaller_detector_and_restores_it(subtitle_video, tmp_path):
    sides = []

    class SizedOCR:
        def set_det_side(self, side):
            sides.append(side)

        def __call__(self, strip):
            return fake_ocr(strip)

    pipeline.process(subtitle_video, tmp_path / "o", args(), SizedOCR(), BlackFill(), "libx264")
    assert sides == [pipeline.VERIFY_DET_SIDE, pipeline.DET_SIDE]


def test_keep_ranges_are_parsed_and_leave_frames_alone():
    assert pipeline.parse_keep("0-95,1:30:00-end", 6000) == [(0.0, 95.0), (5400.0, 6000)]
    assert pipeline.parse_keep("", 100) == []
    assert pipeline.keep_frames("1-2", 25.0, 100) == set(range(25, 51))
    with pytest.raises(ValueError):
        pipeline.parse_keep("20-10", 100)
    with pytest.raises(ValueError):
        pipeline.parse_keep("20", 100)


def test_a_kept_range_is_not_read_not_erased_and_not_in_the_srt(subtitle_video, tmp_path):
    # the subtitle bar is on frames 25-74 (1.0-3.0 s): keeping 0-2.0 s leaves its first half on screen
    calls = {"all": 0, "kept": 0}

    def spy(key):
        def f(strip):
            calls[key] += 1
            return fake_ocr(strip)
        return f

    pipeline.process(subtitle_video, tmp_path / "all", args("--verify", "off", "--ocr-interval", "1"), spy("all"), BlackFill(),
                     "libx264")
    out = tmp_path / "out"
    pipeline.process(subtitle_video, out, args("--keep-ranges", "0-2.0", "--verify", "off", "--ocr-interval", "1"),
                     spy("kept"), BlackFill(), "libx264")
    assert calls["all"] == 100 and calls["kept"] == 100 - 51   # frames 0-50 are not read
    res = out / "film_clean.mp4"
    srt = (out / "film.srt").read_text(encoding="utf-8")
    assert "00:00:01" not in srt and "00:00:02" in srt   # the cue starts where the kept range ends (frame 51 = 2.04 s)
    frames = list(read_frames(res, 640, 360, count=100))
    assert white_pixels(frames[40]).sum() > 0        # frame 40 (1.6 s): the bar is still there
    assert white_pixels(frames[60]).sum() == 0       # frame 60 (2.4 s): erased

def test_untouched_pixels_stay_as_they_were_not_converted_through_rgb(tmp_path):
    # SMPTE HD colour bars hold colours that RGB cannot: converting every pixel through RGB moves them. The bars (top 60 %)
    # are not erased, so with --compose auto they must stay as the decoder gave them; --compose off converts them.
    from clean_clear.subtitle_qc import finish_psnr, start_psnr
    from conftest import ffmpeg
    src = tmp_path / "bars.mp4"
    ffmpeg("-f", "lavfi", "-i", "smptehdbars=size=1280x720:rate=25:duration=3", "-vf",
           "drawbox=x=500:y=650:w=280:h=36:color=white:t=fill:enable='between(n,25,60)'", "-c:v", "libx264", "-qp", "8",
           "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", src)
    mean = {}
    for mode in ("auto", "off"):
        out = tmp_path / mode
        pipeline.process(src, out, args("--compose", mode, "--verify", "off"), bar_ocr, BlackFill(), "libx264")
        mean[mode] = finish_psnr(start_psnr(src, out / "bars_clean.mp4", tmp_path, mode, top=0.6), 25.0)["mean"]
    assert mean["auto"] > 45 and mean["auto"] > mean["off"] + 3, mean


def test_the_verify_windows_keep_untouched_pixels_too(tmp_path):
    # the first pass leaves the subtitle; the verify pass re-encodes the window around it, composed in YUV as well
    from clean_clear.subtitle_qc import finish_psnr, start_psnr
    from conftest import ffmpeg

    class MissFirst(BlackFill):
        calls = 0

        def erase(self, frames, frame_seg, masks, cuts=()):
            MissFirst.calls += 1
            if MissFirst.calls == 1:
                return iter(frames)
            return super().erase(frames, frame_seg, masks, cuts)

    src = tmp_path / "bars.mp4"
    ffmpeg("-f", "lavfi", "-i", "smptehdbars=size=1280x720:rate=25:duration=3", "-vf",
           "drawbox=x=500:y=650:w=280:h=36:color=white:t=fill:enable='between(n,25,60)'", "-c:v", "libx264", "-qp", "8",
           "-g", "25", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", src)
    out = tmp_path / "o"
    pipeline.process(src, out, args("--compose", "auto"), bar_ocr, MissFirst(), "libx264")
    psnr = finish_psnr(start_psnr(src, out / "bars_clean.mp4", tmp_path, "v", top=0.6), 25.0)
    assert psnr["n"] == 75 and psnr["min"] > 40, psnr
