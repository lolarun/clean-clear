import subprocess

import numpy as np
import pytest

from clean_clear import yuvcomp
from clean_clear.video import can_compose_yuv, read_frames, read_yuv
from conftest import ffmpeg


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "c.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=128x96:rate=25:duration=1", "-c:v", "libx264", "-qp", "10", "-pix_fmt", "yuv420p", p)
    return p


def test_the_yuv_frames_are_those_of_the_bgr_frames(clip):
    assert can_compose_yuv(clip) == "bt601"          # untagged, under 720 lines: what players assume
    yuv = list(read_yuv(clip, 128, 96))
    bgr = list(read_frames(clip, 128, 96))
    assert len(yuv) == len(bgr) == 25 and yuv[0].size == 128 * 96 * 3 // 2
    # our conversion of the BGR frame agrees with the decoder's YUV where RGB can hold the colour (testsrc2 mostly can)
    y, u, v = yuvcomp.bgr_to_yuv420(bgr[3], "bt601")
    raw = yuv[3]
    ry = raw[:128 * 96].reshape(96, 128)
    assert np.abs(y.astype(int) - ry).mean() < 1.5


def test_pixels_outside_the_regions_are_the_decoders_bytes(clip):
    raw = next(iter(read_yuv(clip, 128, 96)))
    black = np.zeros((96, 128, 3), np.uint8)
    crop = np.zeros((20, 40), bool)
    crop[4:16, 6:34] = True
    region = (60, 30, crop)                              # rows 60-79, columns 30-69
    out = np.frombuffer(yuvcomp.compose(raw, black, [region], "bt601", 128, 96), np.uint8)
    Y, Yo = raw[:128 * 96].reshape(96, 128), out[:128 * 96].reshape(96, 128)
    inside = np.zeros((96, 128), bool)
    inside[64:76, 36:64] = True
    assert (Yo[~inside] == Y[~inside]).all()             # not one byte outside the region changed
    assert (Yo[inside] == 16).all()                      # black in limited range
    U, Uo = raw[128 * 96:128 * 96 + 128 * 24].reshape(48, 64), out[128 * 96:128 * 96 + 128 * 24].reshape(48, 64)
    blocks = np.zeros((48, 64), bool)
    blocks[32:38, 18:32] = True
    assert (Uo[~blocks] == U[~blocks]).all() and (Uo[blocks] == 128).all()
    assert yuvcomp.compose(raw, black, [], "bt601", 128, 96) == raw.tobytes()      # nothing to change: as decoded
    assert yuvcomp.compose(raw, black, [(0, 0, np.zeros((4, 4), bool))], "bt601", 128, 96) == raw.tobytes()


def test_a_region_at_the_edge_and_an_odd_box_are_handled(clip):
    raw = next(iter(read_yuv(clip, 128, 96)))
    white = np.full((96, 128, 3), 255, np.uint8)
    crop = np.ones((9, 9), bool)
    out = np.frombuffer(yuvcomp.compose(raw, white, [(87, 123, crop)], "bt601", 128, 96), np.uint8)   # sticks out at the corner
    Y = out[:128 * 96].reshape(96, 128)
    assert (Y[87:96, 123:128] == 235).all() and Y[86, 123] == raw[:128 * 96].reshape(96, 128)[86, 123]


def test_not_every_video_can_be_composed(tmp_path):
    p = tmp_path / "ten.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=128x96:rate=25:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p10le", p)
    q = tmp_path / "odd.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=126x94:rate=25:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", q)
    r = tmp_path / "full.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=128x96:rate=25:duration=1", "-vf", "scale=out_range=pc", "-c:v", "libx264",
           "-pix_fmt", "yuvj420p", r)
    assert can_compose_yuv(p) is None and can_compose_yuv(r) is None
    assert can_compose_yuv(q) == "bt601"
