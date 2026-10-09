"""--cut START-END: remove time ranges (e.g. an advertisement page) from a video, frame-exact.

Only the keyframe windows around the cuts are decoded and re-encoded; everything else is stream-copied (see
rewrite.py for why). The audio of the cut ranges is removed with a sample-exact filter, and the SRT next to the
video (if any) loses the cues inside the ranges and is shifted accordingly."""
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .common import fmt_time, log
from .rewrite import NotPlainTimeline, _copy, plain_timeline, windows_for
from .video import ThreadedWriter, _exe, keyframes, open_writer, probe, read_frames

_CUE = re.compile(r"(\d+):(\d\d):(\d\d),(\d{3}) --> (\d+):(\d\d):(\d\d),(\d{3})")


def parse_time(s):
    """'2401.667', '40:01.667' or '0:40:01.667' -> seconds"""
    sec = 0.0
    for part in s.strip().split(":"):
        sec = sec * 60 + float(part)
    return sec


def parse_ranges(specs):
    """['0:40:01.667-0:40:46.958', ...] -> [(start s, end s)] sorted, validated"""
    out = []
    for spec in specs:
        a, sep, b = spec.partition("-")
        if not sep:
            raise ValueError(f"--cut expects START-END, got {spec!r}")
        a, b = parse_time(a), parse_time(b)
        if b <= a:
            raise ValueError(f"--cut range ends before it starts: {spec!r}")
        out.append((a, b))
    return sorted(out)


def _audio_bitrate(video):
    out = subprocess.run([_exe("ffprobe"), "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=bit_rate",
                          "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 192000


def _cut_srt(srt, dst, ranges):
    """Copy `srt` to `dst` without the cues inside `ranges` [(start s, end s)], later cues shifted earlier"""
    def ms(h, m, s, f):
        return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(f)

    def fmt(v):
        h, v = divmod(int(round(v)), 3600000)
        m, v = divmod(v, 60000)
        s, v = divmod(v, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{v:03d}"

    def shift(t):  # milliseconds removed before time t (ms)
        return sum(min(t, b * 1000) - min(t, a * 1000) for a, b in ranges)

    cues = []
    for block in re.split(r"\n\s*\n", srt.read_text(encoding="utf-8").strip()):
        lines = block.splitlines()
        m = _CUE.match(lines[1]) if len(lines) >= 3 else None
        if not m:
            continue
        a, b = ms(*m.groups()[:4]), ms(*m.groups()[4:])
        if any(a < rb * 1000 and b > ra * 1000 for ra, rb in ranges):
            continue
        cues.append((a - shift(a), b - shift(b), "\n".join(lines[2:])))
    with open(dst, "w", encoding="utf-8") as fh:
        for i, (a, b, t) in enumerate(cues, 1):
            fh.write(f"{i}\n{fmt(a)} --> {fmt(b)}\n{t}\n\n")
    return len(cues)


def cut(video, out_dir, ranges, encoder, crf):
    """Write NAME_cut.mp4 (and NAME_cut.srt if NAME.srt exists) without the `ranges` [(start s, end s)]"""
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (video.stem + "_cut.mp4")
    if not plain_timeline(video):
        raise NotPlainTimeline(f"{video.name} has hidden or out-of-range frames (e.g. a stream-copy cut); "
                               "frame counts cannot be trusted for a partial re-encode")
    from .ads import _start_time
    if abs(_start_time(video)) > 0.001:
        raise NotPlainTimeline(f"{video.name} does not start at time 0 ({_start_time(video):.3f} s, typical for mkv); "
                               "cut the *_clean.mp4 result instead (its timeline starts at 0)")
    W, H, fps, n = probe(video)
    drop, spans = set(), []
    for a, b in ranges:
        fa, fb = round(a * fps), min(n, round(b * fps))  # frames fa .. fb-1
        if fb > fa:
            spans.append((fa, fb))
            drop.update(range(fa, fb))
    if not spans:
        raise ValueError("--cut: nothing to remove")
    removed = len(drop)
    log(f"\n==> cut {video.name}: {len(spans)} range(s), {removed} frames = {fmt_time(removed / fps)}: "
        + ", ".join(f"{fmt_time(a / fps)}-{fmt_time(b / fps)}" for a, b in spans))
    t0 = time.time()
    kft = keyframes(video)
    wins = windows_for([(a, b - 1) for a, b in spans], sorted(kft), n, 0)
    work = out_dir / f".cut_{video.stem}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    pieces, pos = [], 0
    for j, (a, b) in enumerate(wins):
        if a > pos:
            pieces.append((_copy(video, work / f"c{j:03d}.mp4", kft.get(pos), a - pos), a - pos))
        keep = (b - a) - sum(1 for i in range(a, b) if i in drop)
        chunk = work / f"w{j:03d}.mp4"
        wr = open_writer(video, chunk, W, H, fps, encoder, crf, audio=False)
        tw = ThreadedWriter(wr)
        written = 0
        try:
            for i, f in enumerate(read_frames(video, W, H, start=a, count=b - a), a):
                if i not in drop:
                    tw.write(np.ascontiguousarray(f).tobytes())
                    written += 1
        finally:
            tw.close()
        if wr.returncode != 0 or written != keep:
            raise RuntimeError(f"re-encode of frames {a}-{b} failed (encoder code {wr.returncode}, {written}/{keep} frames)")
        if keep:
            pieces.append((chunk, keep))
        pos = b
    if pos < n:
        pieces.append((_copy(video, work / "tail.mp4", kft.get(pos), 0), n - pos))
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\nduration {cnt / fps:.6f}\n" for p, cnt in pieces),
                   encoding="utf-8")
    vid = work / "video.mp4"
    subprocess.run([_exe("ffmpeg"), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-map", "0:v:0",
                    "-c", "copy", str(vid)], check=True)
    got = probe(vid)[3]
    if got != n - removed:
        raise RuntimeError(f"cut video has {got} frames instead of {n - removed}; kept the parts in {work}")
    # audio: drop the same time ranges (sample-exact), keep the original bit rate
    expr = "+".join(f"between(t,{a / fps:.6f},{b / fps:.6f})" for a, b in spans)
    subprocess.run([_exe("ffmpeg"), "-v", "error", "-y", "-i", str(vid), "-i", str(video), "-map", "0:v:0", "-map", "1:a?",
                    "-c:v", "copy", "-af", f"aselect='not({expr})',asetpts=N/SR/TB", "-c:a", "aac",
                    "-b:a", str(_audio_bitrate(video)), "-movflags", "+faststart", str(dst)], check=True)
    shutil.rmtree(work, ignore_errors=True)
    log(f"  video -> {dst}  ({time.time() - t0:.0f}s, {sum(b - a for a, b in wins)} of {n} frames re-encoded)")
    srt = video.with_suffix(".srt")
    if not srt.exists():
        srt = video.with_name(video.stem.removesuffix("_clean") + ".srt")
    if srt.exists():
        dst_srt = out_dir / (video.stem + "_cut.srt")
        k = _cut_srt(srt, dst_srt, [(a / fps, b / fps) for a, b in spans])
        log(f"  {k} subtitles -> {dst_srt}")
    return dst
