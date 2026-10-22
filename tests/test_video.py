import threading

import numpy as np
import pytest

from clean_clear import video
from clean_clear.video import (ThreadedWriter, duration_check, keyframes, open_writer, packet_count, prefetch, probe,
                               read_frames, restore_start, sar, video_offset)
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
    # tagged: say BT.709 on the command line. The bars source marks its frames itself, and ffmpeg 7.1.1 marks the HD bars
    # BT.470BG (601): the file then says 601 for colours that only fit RGB as 709, the decode (rightly) follows the file,
    # the saturated bars clip (Y 63 -> 79) and the test failed with a defect that was in the test film, not in the tool
    ffmpeg("-f", "lavfi", "-i", f"{bars}=s={size}:r=25:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           *(["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"] if tagged else UNTAGGED), src)
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


def test_packet_count_is_the_exact_frame_count_of_a_part_cut_with_copy(subtitle_video, tmp_path):
    part = tmp_path / "part.mkv"
    ffmpeg("-i", subtitle_video, "-map", "0:v:0", "-c", "copy", part)
    assert packet_count(part) == 100
    assert packet_count(tmp_path / "missing.mkv") == 0


@pytest.fixture
def late_video(tmp_path):
    """2 s of moving test pattern (every frame differs) with audio; the picture starts 0.04 s after the audio, like
    sources whose video stream starts at 0.045 s. Returns (plain, late)."""
    plain, late = tmp_path / "plain.mp4", tmp_path / "late.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:d=2", "-f", "lavfi", "-i", "sine=d=2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", plain)
    ffmpeg("-itsoffset", "0.04", "-i", plain, "-i", plain, "-map", "0:v:0", "-map", "1:a", "-c", "copy", late)
    return plain, late


def test_a_late_first_frame_is_neither_repeated_nor_dropped(late_video):
    plain, late = late_video
    assert video_offset(late) == pytest.approx(0.04, abs=0.002) and video_offset(plain) == 0.0
    a = list(read_frames(plain, 160, 120))
    b = list(read_frames(late, 160, 120, count=probe(late)[3]))
    assert len(a) == len(b) == 50
    assert all(np.array_equal(x, y) for x, y in zip(a, b))  # cfr decoding gave frame 0 twice and lost frame 49


def test_restore_start_keeps_the_picture_after_the_audio(late_video, tmp_path):
    _, late = late_video
    dst = tmp_path / "out.mp4"
    assert round_trip(late, dst) == 50
    assert video_offset(dst) == 0.0  # the encoder writes the picture from time 0
    assert restore_start(late, dst) == pytest.approx(0.04, abs=0.002)
    assert video_offset(dst) == pytest.approx(0.04, abs=0.002)
    assert probe(dst)[3] == 50 and min(keyframes(dst)) == 0  # frame numbering still starts at the first frame
    assert len(list(read_frames(dst, 160, 120))) == 50


def test_writer_keeps_the_pixel_aspect_ratio(tmp_path):
    src, dst = tmp_path / "anamorphic.mp4", tmp_path / "out.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:d=1", "-vf", "setsar=4/3", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", src)
    assert sar(src) == "4/3"
    round_trip(src, dst)
    assert sar(dst) == "4/3"  # the encoder used to write square pixels: the picture was shown 3/4 as wide
    plain = tmp_path / "plain.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", plain)
    assert sar(plain) is None


def test_ffmpeg_threads_follow_the_process_cpu_share(monkeypatch):
    monkeypatch.setattr(video, "FFMPEG_THREADS", "auto")
    monkeypatch.setenv("CLEAN_CLEAR_CPUS", "6")      # 25 cores shared by 4 part processes
    assert video.ffmpeg_threads() == 3
    assert video._threads_opts() == ["-filter_threads", "3", "-threads", "3"]
    monkeypatch.setenv("CLEAN_CLEAR_CPUS", "25")     # one process alone
    assert video.ffmpeg_threads() == 12
    monkeypatch.setenv("CLEAN_CLEAR_CPUS", "1")
    assert video.ffmpeg_threads() == 2
    monkeypatch.setattr(video, "FFMPEG_THREADS", "0")  # ffmpeg's own default: no options at all
    assert video.ffmpeg_threads() == 0 and video._threads_opts() == []


@pytest.mark.parametrize("rate", ["24000/1001", "30000/1001", "60000/1001", "25/1", "24/1"])
def test_the_result_states_the_frame_rate_of_the_source_exactly(tmp_path, rate):
    # NTSC rates came out as 991001/41333, 979001/32666 and 979001/16333: right picture, another stated rate
    src, dst = tmp_path / "s.mp4", tmp_path / "d.mp4"
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=size=128x96:rate={rate}:duration=2", "-f", "lavfi", "-i", "sine=duration=2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", src)
    w, h, fps, n = probe(src)
    tw = ThreadedWriter(open_writer(src, dst, w, h, fps, "libx264", 23))
    for f in read_frames(src, w, h, count=n):
        tw.write(f)
    tw.close()
    entries = "stream=r_frame_rate,avg_frame_rate,nb_frames"
    a, b = probe_streams(src, entries)[0], probe_streams(dst, entries)[0]
    assert (b["r_frame_rate"], b["avg_frame_rate"], b["nb_frames"]) == (a["r_frame_rate"], a["avg_frame_rate"], a["nb_frames"])
