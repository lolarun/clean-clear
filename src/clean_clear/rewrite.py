"""Re-encode only the parts of a finished video that change; stream-copy everything else.

Every decode/encode round trip costs picture quality, and a correction that touches a few subtitles should not
re-encode a 90-minute film. The changed frames are widened to the surrounding keyframes, those windows are decoded,
processed and encoded with the same settings, and the film is reassembled with the untouched pieces copied
bit for bit. Pieces are cut by frame count, not by time: with B-frames a time cut at a keyframe carries a few
packets of the next GOP along (seen as 3 extra frames and a broken seam)."""
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .common import log
from .video import ThreadedWriter, _exe, keyframes, open_writer, probe, read_frames


def windows_for(ranges, kfs, n, margin):
    """Frame ranges [(first, last)] -> merged keyframe-aligned windows [(a, b)], b exclusive"""
    out = []
    for first, last in sorted(ranges):
        a = max([k for k in kfs if k <= max(0, first - margin)], default=0)
        b = min([k for k in kfs if k > min(n - 1, last + margin)], default=n)
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def rewrite(video, audio_src, dst, ranges, process, encoder, crf, margin=40):
    """Write `dst` = `video` with the frames around `ranges` replaced by process(frames, a, b) (an iterator of the
    frames a..b-1, already processed). The audio is copied from `audio_src`. Returns the windows rewritten."""
    video, dst = Path(video), Path(dst)
    W, H, fps, n = probe(video)
    kfs = keyframes(video)
    wins = windows_for(ranges, kfs, n, margin)
    work = dst.parent / (dst.stem + ".rewrite")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    t0 = time.time()
    pieces, pos = [], 0
    for j, (a, b) in enumerate(wins):
        if a > pos:
            pieces.append(_copy(video, work / f"c{j:03d}.mp4", pos, a - pos, fps))
        chunk = work / f"w{j:03d}.mp4"
        wr = open_writer(video, chunk, W, H, fps, encoder, crf, audio=False)
        tw = ThreadedWriter(wr)
        written = 0
        try:
            for f in process(read_frames(video, W, H, start=a, count=b - a), a, b):
                tw.write(np.ascontiguousarray(f).tobytes())
                written += 1
        finally:
            tw.close()
        if wr.returncode != 0 or written != b - a:
            raise RuntimeError(f"rewrite of frames {a}-{b} failed (encoder code {wr.returncode}, {written} frames)")
        pieces.append(chunk)
        pos = b
    if pos < n:
        pieces.append(_copy(video, work / "tail.mp4", pos, 0, fps))
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in pieces), encoding="utf-8")
    subprocess.run([_exe("ffmpeg"), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(audio_src),
                    "-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-movflags", "+faststart", str(dst)], check=True)
    got = probe(dst)[3]
    if got != n:
        raise RuntimeError(f"rewritten video has {got} frames instead of {n}; kept the parts in {work}")
    shutil.rmtree(work, ignore_errors=True)
    log(f"  re-encoded {sum(b - a for a, b in wins)} of {n} frames in {len(wins)} window(s), the rest copied ({time.time() - t0:.0f}s)")
    return wins


def _copy(video, out, start, count, fps):
    cmd = [_exe("ffmpeg"), "-v", "error", "-y"]
    if start:
        cmd += ["-ss", f"{(start + 0.25) / fps:.6f}"]  # stream copy starts at the keyframe at/before this time: `start` itself
    cmd += ["-i", str(video), "-map", "0:v:0", "-c", "copy"]
    if count:
        cmd += ["-frames:v", str(count)]
    subprocess.run(cmd + [str(out)], check=True)
    return out
