"""Shared helpers: small synthetic videos made with ffmpeg (no models or GPU needed)."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg is required for the tests", allow_module_level=True)


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def probe_streams(path, entries="stream=codec_type,color_space,duration,nb_frames"):
    import json
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json", str(path)],
                         capture_output=True, check=True).stdout
    return json.loads(out)["streams"]


def yuv_mean(path, crop, frame=0):
    """Mean Y, Cb, Cr of a crop (w, h, x, y) of one frame, read straight from the YUV planes (no conversion)"""
    w, h, x, y = crop
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{frame}),crop={w}:{h}:{x}:{y}",
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv444p", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(3, -1).mean(1)


@pytest.fixture
def subtitle_video(tmp_path):
    """640x360, 25 fps, 4 s, flat blue background with a white 'subtitle' bar on frames 25-74, plus audio"""
    p = tmp_path / "film.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=0x3060a0:s=640x360:r=25:d=4", "-f", "lavfi", "-i", "sine=d=4",
           "-vf", "drawbox=x=250:y=300:w=140:h=20:color=white:t=fill:enable='between(n,25,74)'",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", p)
    return p
