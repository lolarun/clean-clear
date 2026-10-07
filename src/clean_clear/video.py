"""ffmpeg helpers: probing, frame decoding and encoding."""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading

import numpy as np

from .common import ROOT, Progress

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


# Exact YUV<->RGB rounding. With the default flags every decode/encode round trip made the picture ~1.5 luma
# levels darker, so a film that went through three passes ended up ~4.4 levels darker than its source.
# The flags must follow the input: placed before -i, ffmpeg 6.1 silently ignores them.
SWS = ["-sws_flags", "accurate_rnd+full_chroma_int+bitexact"]


def read_frames(path, w, h, crop=None, start=0, count=0):
    """Yield frames as read-only BGR ndarrays. With crop=(y0, y1) only that horizontal band is decoded.
    start/count: decode only frames start .. start+count-1 (frame-exact for the constant-rate files we write)."""
    cmd = [_exe("ffmpeg"), "-v", "error"]
    if start:
        cmd += ["-ss", f"{(start - 0.25) / _fps(path):.6f}"]  # a quarter frame early: output starts exactly at frame `start`
    cmd += ["-i", str(path), *SWS, "-map", "0:v:0"]  # -sws_flags only takes effect after the input (ffmpeg 6.1)
    if count:
        cmd += ["-frames:v", str(count)]
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


def detect_cuts(path, threshold=12.0, ratio=3.0, window=5, total=0, hist_min=0.2):
    """Shot cut detection: frame indices that start a new shot.

    Consecutive frames are compared as 320x180 images. A cut is a spike of the mean absolute grey difference
    (above `threshold`, 0-255 scale) that is also either `ratio` times the local level of that difference
    (median of the `window` values on each side), or a spike of the colour-histogram distance (Bhattacharyya,
    above `hist_min` and `ratio` times its local level). The grey test alone missed a hard cut right after fast
    motion (54:41 of the test film: 39.4 against 15-23 around it); the colour distribution barely changes during
    motion (0.06-0.09 there) but jumps at the cut (0.33). Sustained fast motion, fire or flashes do not split chunks."""
    import cv2
    w, h = 320, 180
    cmd = [_exe("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", f"scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 16)
    diffs, hdiffs, prev, prev_h = [], [], None, None
    prog = Progress("cuts", total) if total else None
    try:
        while True:
            buf = p.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                break
            img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            cur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
            hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if prev is not None:
                diffs.append(float(np.abs(cur - prev).mean()))  # diffs[j]: frame j -> j + 1
                hdiffs.append(float(cv2.compareHist(prev_h, hist, cv2.HISTCMP_BHATTACHARYYA)))
                if prog:
                    prog.update(len(diffs) + 1)
            prev, prev_h = cur, hist
    finally:
        p.stdout.close()
        p.wait()
    d, hd = np.array(diffs), np.array(hdiffs)
    cuts = []
    for j in np.nonzero(d > threshold)[0]:
        around = np.r_[d[max(0, j - window):j], d[j + 1:j + 1 + window]]
        h_around = np.r_[hd[max(0, j - window):j], hd[j + 1:j + 1 + window]]
        grey_spike = len(around) == 0 or d[j] > ratio * max(float(np.median(around)), 2.0)
        colour_spike = hd[j] > hist_min and (len(h_around) == 0 or hd[j] > ratio * max(float(np.median(h_around)), 0.03))
        if grey_spike or colour_spike:
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


def _fps(path):
    return probe(path)[2]


def keyframes(path):
    """{frame index: exact pts_time string} of the keyframes of a constant-frame-rate video"""
    fps = _fps(path)
    out = subprocess.run([_exe("ffprobe"), "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
                          "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return {round(float(x.strip(",")) * fps): x.strip(",") for x in out.split() if x.strip(",")}


def open_writer(src, dst, w, h, fps, encoder, crf, audio=True):
    """ffmpeg process that encodes raw BGR frames from stdin and copies the audio of `src` (audio=False: video only)."""
    if encoder in ("h264_nvenc", "hevc_nvenc"):
        venc = ["-c:v", encoder, "-preset", "p5", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    else:
        venc = ["-c:v", encoder, "-preset", "medium", "-crf", str(crf)]
    cmd = [_exe("ffmpeg"), "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-"]
    cmd += ["-i", str(src), *SWS, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy"] if audio else [*SWS, "-map", "0:v:0"]
    cmd += [*venc, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
