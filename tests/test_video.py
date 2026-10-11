import threading

import numpy as np
import pytest

from clean_clear import video
from clean_clear.video import ThreadedWriter, duration_check, open_writer, prefetch, probe, read_frames
from conftest import ffmpeg, probe_streams, yuv_mean


def round_trip(src, dst, encoder="libx264"):
    W, H, fps, n = probe(src)
    wr = open_writer(src, dst, W, H, fps, encoder, 18)
    tw = ThreadedWriter(wr)
    k = 0
    try:
        for f in read_frames(src, W, H):
            tw.write(f.tobytes())
            k += 1
    finally:
        tw.close()
    return k


def test_writer_raises_instead_of_hanging_when_the_encoder_dies(tmp_path):
    src = tmp_path / "s.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=s=64x64:d=0.2", src)
    result = {}

    def run():
        tw = ThreadedWriter(open_writer(src, tmp_path / "o.mp4", 640, 360, 25.0, "no_such_encoder", 18))
        frame = np.zeros((360, 640, 3), np.uint8).tobytes()
        try:
            try:
                for _ in range(200):
                    tw.write(frame)
            finally:
                tw.close()
        except RuntimeError as e:
            result["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(30)
    assert not t.is_alive(), "writer blocked after the encoder exited"
    assert "encoder stopped" in str(result.get("error"))


UNTAGGED = ["-bsf:v", "h264_metadata=matrix_coefficients=2:colour_primaries=2:transfer_characteristics=2"]


# HD bars are defined in BT.709, SD bars in BT.601 (other combinations clip outside the RGB gamut)
@pytest.mark.parametrize("bars,size,tagged", [("smptehdbars", "1280x720", True), ("smptehdbars", "1280x720", False),
                                              ("smptebars", "640x360", False)])
def test_colours_survive_a_round_trip(tmp_path, bars, size, tagged):
    src, dst = tmp_path / "bars.mp4", tmp_path / "out.mp4"
    ffmpeg("-f", "lavfi", "-i", f"{bars}=s={size}:r=25:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           *([] if tagged else UNTAGGED), src)
    round_trip(src, dst)
    w, h = map(int, size.split("x"))
    for x in (w // 2 + w // 16, w // 4, w // 9):  # saturated areas
        a, b = yuv_mean(src, (16, 16, x, h // 6)), yuv_mean(dst, (16, 16, x, h // 6))
        assert np.abs(a - b).max() < 2.0, (x, a, b)
    src_tag = probe_streams(src)[0].get("color_space")
    assert src_tag == ("bt709" if tagged else None)
    assert probe_streams(dst)[0]["color_space"] == src_tag or ("bt709" if h >= 720 else "smpte170m")


def test_grey_is_not_darkened(tmp_path):
    src, dst = tmp_path / "grey.mp4", tmp_path / "out.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=0x808080:s=640x360:r=25:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", src)
    round_trip(src, dst)
    assert abs(yuv_mean(src, (64, 64, 100, 100))[0] - yuv_mean(dst, (64, 64, 100, 100))[0]) < 0.6


def test_variable_frame_rate_keeps_audio_in_sync(tmp_path):
    src, dst = tmp_path / "vfr.mp4", tmp_path / "out.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=6", "-f", "lavfi", "-i", "sine=d=6",
           "-vf", "select='lt(n\\,90)*not(mod(n\\,3))+gte(n\\,90)'", "-fps_mode", "vfr", "-c:v", "libx264",
           "-c:a", "aac", src)
    W, H, fps, n = probe(src)
    k = round_trip(src, dst)
    assert abs(k - n) <= 2
    assert duration_check(src, dst) is None
    assert abs(float(probe_streams(dst)[0]["duration"]) - 6.0) < 0.25


def test_duration_check_reports_a_short_output(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=s=64x64:r=25:d=4", a)
    ffmpeg("-f", "lavfi", "-i", "color=s=64x64:r=25:d=3", b)
    assert "drift" in duration_check(a, b)


def test_closing_a_reader_stops_its_ffmpeg(subtitle_video, monkeypatch):
    procs = []
    real = video.subprocess.Popen

    def spy(*a, **k):
        procs.append(real(*a, **k))
        return procs[-1]

    monkeypatch.setattr(video.subprocess, "Popen", spy)
    g = prefetch(read_frames(subtitle_video, 640, 360))
    next(g)
    g.close()
    assert procs and procs[0].wait(timeout=10) is not None


def test_prefetch_passes_errors_and_order():
    def gen():
        yield from range(50)
        raise ValueError("boom")

    out = []
    with pytest.raises(ValueError):
        for x in prefetch(gen(), maxsize=4):
            out.append(x)
    assert out == list(range(50))


def test_read_frames_window_is_frame_exact(subtitle_video):
    frames = list(read_frames(subtitle_video, 640, 360))
    part = list(read_frames(subtitle_video, 640, 360, start=30, count=5))
    assert len(part) == 5
    assert all(np.array_equal(a, b) for a, b in zip(frames[30:35], part))
