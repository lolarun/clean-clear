"""ffmpeg helpers: probing, frame decoding and encoding."""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading

import numpy as np

from common import ROOT

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".ts", ".m4v", ".wmv", ".webm"}


def _exe(name):
    local = ROOT / "ffmpeg" / (name + (".exe" if os.name == "nt" else ""))
    if local.exists():
        return str(local)
    p = shutil.which(name)
    if not p:
        sys.exit(f"{name} not found: install ffmpeg and add it to PATH, or put it in {ROOT / 'ffmpeg'}")
    return p


def probe(path):
    """-> (width, height, fps, approximate frame count)"""
    out = subprocess.run(
        [_exe("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
         "-of", "json", str(path)],
        capture_output=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = (s.get("avg_frame_rate") or s["r_frame_rate"]).split("/")
    if float(den) == 0 or float(num) == 0:
        num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    dur = float(j["format"].get("duration", 0) or 0)
    n = int(s.get("nb_frames") or 0) or int(round(dur * fps))
    return int(s["width"]), int(s["height"]), fps, n


def read_frames(path, w, h, crop=None):
    """Yield frames as read-only BGR ndarrays. With crop=(y0, y1) only that horizontal band is decoded."""
    cmd = [_exe("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0"]
    if crop:
        y0, y1 = crop
        cmd += ["-vf", f"crop={w}:{y1 - y0}:0:{y0}"]
        h = y1 - y0
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=int(w * h * 3 * 4))
    size = w * h * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


def detect_cuts(path, threshold=12.0, ratio=3.0, window=5):
    """Shot cut detection: frame indices that start a new shot.

    Consecutive frames are compared as 320x180 grayscale (mean absolute difference, 0-255 scale).
    A cut is a spike: the difference exceeds `threshold` AND is `ratio` times the local level
    (median of the `window` differences on each side). Normal motion stays below ~5 and hard cuts
    are usually 15-30+; the spike rule rejects sustained fast motion, fire or flashes, which would
    otherwise split chunks needlessly."""
    w, h = 320, 180
    cmd = [_exe("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", f"scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 16)
    diffs, prev = [], None
    try:
        while True:
            buf = p.stdout.read(w * h)
            if len(buf) < w * h:
                break
            cur = np.frombuffer(buf, np.uint8).astype(np.int16)
            if prev is not None:
                diffs.append(float(np.abs(cur - prev).mean()))  # diffs[j]: frame j -> j + 1
            prev = cur
    finally:
        p.stdout.close()
        p.wait()
    d = np.array(diffs)
    cuts = []
    for j in np.nonzero(d > threshold)[0]:
        around = np.r_[d[max(0, j - window):j], d[j + 1:j + 1 + window]]
        if len(around) == 0 or d[j] > ratio * max(float(np.median(around)), 2.0):
            cuts.append(int(j) + 1)
    return cuts


def prefetch(gen, maxsize=8):
    """Run `gen` (e.g. read_frames) on a background thread so decoding overlaps with the consumer
    (GPU inference in the main thread) instead of alternating with it."""
    q = queue.Queue(maxsize)
    done = object()

    def worker():
        try:
            for item in gen:
                q.put(item)
            q.put(done)
        except Exception as e:
            q.put(e)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is done:
            return
        if isinstance(item, Exception):
            raise item
        yield item


class ThreadedWriter:
    """Wraps an encoder process (from open_writer) so stdin writes happen on a background thread,
    overlapping the encoder with the next frame's inference instead of blocking on it."""

    def __init__(self, proc, maxsize=8):
        self.proc = proc
        self._q = queue.Queue(maxsize)
        self._err = None
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _worker(self):
        try:
            while True:
                buf = self._q.get()
                if buf is None:
                    return
                self.proc.stdin.write(buf)
        except Exception as e:
            self._err = e

    def write(self, buf):
        if self._err:
            raise self._err
        self._q.put(buf)

    def close(self):
        self._q.put(None)
        self._t.join()
        self.proc.stdin.close()
        self.proc.wait()
        if self._err:
            raise self._err


def pick_encoder(want):
    if want != "auto":
        return want
    try:
        r = subprocess.run(
            [_exe("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.2",
             "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=30)
        if r.returncode == 0:
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"


def open_writer(src, dst, w, h, fps, encoder, crf):
    """ffmpeg process that encodes raw BGR frames from stdin and copies the audio of `src`."""
    if encoder in ("h264_nvenc", "hevc_nvenc"):
        venc = ["-c:v", encoder, "-preset", "p5", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    else:
        venc = ["-c:v", encoder, "-preset", "medium", "-crf", str(crf)]
    cmd = [_exe("ffmpeg"), "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
           "-i", str(src), "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy",
           *venc, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
