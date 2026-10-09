"""--find-ads: find advertisement pages that were spliced into a film.

Two signals, both cheap:
- Letterbox intrusion: a film is usually shot with black bars above and below the picture; an inserted ad is
  full-frame, so its picture (or its corner logo) shows up in rows that are black everywhere else. The bars are
  measured on the whole video (rows that are black in almost every sampled frame), then runs of frames with
  content in the bars are candidates.
- URL text: a few frames of each candidate are read with OCR; `www.`, `.com`, `http` ... confirm an ad.
Candidates with a URL are "confirmed" (safe to cut), the others are only "suspect" (listed, never cut)."""
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from .common import log
from .video import SWS, _exe, probe

SW, SH = 320, 180  # analysis resolution
URL = re.compile(r"(www\.|https?:|\.com\b|\.cn\b|\.net\b|\.cc\b|\.org\b|\.vip\b)", re.I)


def _gray_frames(video, fps_expr, start=None, end=None):
    """Yield (time offset index, SHxSW uint8 gray frame) at the rate `fps_expr` ('1' = one per second, or 'source')"""
    cmd = [_exe("ffmpeg"), "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    vf = (f"fps={fps_expr}," if fps_expr != "source" else "") + f"scale={SW}:{SH},format=gray"
    cmd += ["-i", str(video), *SWS, "-map", "0:v:0", "-vf", vf, "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=SW * SH * 8)
    i = 0
    try:
        while True:
            buf = p.stdout.read(SW * SH)
            if len(buf) < SW * SH:
                break
            yield i, np.frombuffer(buf, np.uint8).reshape(SH, SW)
            i += 1
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def find_bars(rows_dark, min_frac=0.03):
    """rows_dark: (H,) fraction of sampled frames in which each row is black -> (top rows, bottom rows) of the
    letterbox, or None when the film has no (or very thin) bars"""
    black = rows_dark > 0.9
    top = 0
    while top < len(black) and black[top]:
        top += 1
    bot = 0
    while bot < len(black) - top and black[-1 - bot]:
        bot += 1
    if top + bot < min_frac * len(black):
        return None
    return top, bot


def _intrusion(frame, top, bot, margin):
    """Fraction of pixels in the bar rows (minus a margin next to the picture) that are not black"""
    parts = []
    if top > margin:
        parts.append(frame[:top - margin])
    if bot > margin:
        parts.append(frame[SH - bot + margin:])
    if not parts:
        return 0.0
    return float(np.mean([np.mean(p > 40) for p in parts]))


def _start_time(video):
    """Presentation time of the first video frame (mkv files often start at a few ms, not at 0)"""
    out = subprocess.run([_exe("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=start_time",
                          "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def _frame_at(video, t, W, H):
    r = subprocess.run([_exe("ffmpeg"), "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", *SWS,
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
    if len(r.stdout) != W * H * 3:
        return None
    return np.frombuffer(r.stdout, np.uint8).reshape(H, W, 3)


def find_ads(video, ocr=None, min_len=10.0, gap=3.0, thresh=0.02):
    """-> [{'start': s, 'end': s, 'status': 'confirmed'|'suspect', 'text': str}] sorted by time.
    start/end are frame-exact seconds (end exclusive)."""
    video = Path(video)
    W, H, fps, n = probe(video)
    duration = n / fps
    off = _start_time(video)
    t0 = time.time()
    frames = [f.copy() for _, f in _gray_frames(video, "1")]
    if len(frames) < 30:
        return []
    stack = np.stack(frames)
    rows_dark = (stack.max(axis=2) < 40).mean(axis=0)  # per row: fraction of frames in which it is black
    bars = find_bars(rows_dark)
    if bars is None:
        log("  ads: no letterbox bars in this film, cannot tell ad pages from the picture")
        return []
    top, bot = bars
    margin = max(2, round(0.02 * SH))
    log(f"  ads: letterbox bars {top}+{bot} of {SH} rows (scaled); {len(frames)} frames sampled ({time.time() - t0:.0f}s)")
    inside = np.array([_intrusion(f, top, bot, margin) > thresh for f in frames])
    # runs of intrusion frames, short gaps joined (dark moments of an ad), then long enough
    runs, i = [], 0
    while i < len(inside):
        if inside[i]:
            j = i
            while j + 1 < len(inside) and inside[j + 1 : j + 2 + int(gap)].any():
                j += 1 + int(np.argmax(inside[j + 1 : j + 2 + int(gap)]))
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    runs = [(a, b) for a, b in runs if b - a + 1 >= min_len]
    ads = []
    for a, b in runs:
        s, e = _refine(video, fps, off, bars, margin, thresh, max(0.0, a - 2.0), a + 2.0, "first"), \
               _refine(video, fps, off, bars, margin, thresh, b - 1.0, min(duration, b + 3.0), "last")
        ad = {"start": s, "end": e, "status": "suspect", "text": ""}
        if ocr is not None:
            for t in np.linspace(s + 1, e - 1, 8):
                img = _frame_at(video, t + off, W, H)
                if img is None:
                    continue
                hit = [txt for _, txt, sc in ocr(img) if sc > 0.5 and URL.search(txt)]
                if hit:
                    ad["status"], ad["text"] = "confirmed", hit[0]
                    break
        ads.append(ad)
    for ad in ads:
        log(f"  ad {ad['status']}: {ts(ad['start'])} - {ts(ad['end'])}  ({ad['end'] - ad['start']:.1f}s)"
            + (f"  \"{ad['text']}\"" if ad["text"] else ""))
    if not ads:
        log("  ads: none found")
    return ads


def _refine(video, fps, off, bars, margin, thresh, a, b, which):
    """Frame-exact edge of an ad between times a and b: the first (which='first') or last+1 ('last') intrusion frame"""
    top, bot = bars
    flags = []
    for _, f in _gray_frames(video, "source", a + off, b + off):
        flags.append(_intrusion(f, top, bot, margin) > thresh)
    idx = [i for i, v in enumerate(flags) if v]
    if not idx:
        return a if which == "first" else b
    k = idx[0] if which == "first" else idx[-1] + 1
    # frame numbers count from the first frame of the stream; the decode starts at the first frame at or after `a`
    return (math.ceil(a * fps - 1e-3) + k) / fps


def ts(sec):
    """Seconds -> H:MM:SS.mmm (what --cut parses)"""
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    return f"{h}:{m:02d}:{ms // 1000:02d}.{ms % 1000:03d}"


def write_report(video, out_dir, ads):
    """NAME.ads.txt: one line per ad: START-END, status, URL text; START-END can be pasted into --cut"""
    path = Path(out_dir) / (Path(video).stem + ".ads.txt")
    path.write_text("".join(f"{ts(a['start'])}-{ts(a['end'])}\t{a['status']}\t{a['text']}\n" for a in ads),
                    encoding="utf-8")
    return path
