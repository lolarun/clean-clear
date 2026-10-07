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
    if not plain_timeline(video):
        raise NotPlainTimeline(f"{video.name} has hidden or out-of-range frames (e.g. a stream-copy cut); "
                               "frame counts cannot be trusted for a partial re-encode")
    W, H, fps, n = probe(video)
    kft = keyframes(video)
    wins = windows_for(ranges, sorted(kft), n, margin)
    work = dst.parent / (dst.stem + ".rewrite")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    t0 = time.time()
    pieces, pos = [], 0
    for j, (a, b) in enumerate(wins):
        if a > pos:
            pieces.append((_copy(video, work / f"c{j:03d}.mp4", kft.get(pos), a - pos), a - pos))
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
        pieces.append((chunk, b - a))
        pos = b
    if pos < n:
        pieces.append((_copy(video, work / "tail.mp4", kft.get(pos), 0), n - pos))
    lst = work / "list.txt"
    # exact durations: an encoded window's last frame can be stored with a 1-tick duration, and the concat demuxer
    # would then start the next piece one frame early (seen as a non-monotonic DTS at the join)
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\nduration {cnt / fps:.6f}\n" for p, cnt in pieces),
                   encoding="utf-8")
    subprocess.run([_exe("ffmpeg"), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(audio_src),
                    "-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-movflags", "+faststart", str(dst)], check=True)
    got = probe(dst)[3]
    if got != n:
        raise RuntimeError(f"rewritten video has {got} frames instead of {n}; kept the parts in {work}")
    shutil.rmtree(work, ignore_errors=True)
    log(f"  re-encoded {sum(b - a for a, b in wins)} of {n} frames in {len(wins)} window(s), the rest copied ({time.time() - t0:.0f}s)")
    return wins


class NotPlainTimeline(ValueError):
    pass


def plain_timeline(video, packets=60):
    """True if the first packets have no negative timestamps and no discard flag. Stream-copy cuts start with
    hidden pre-roll frames that the container counts but playback skips, which would shift every window."""
    out = subprocess.run([_exe("ffprobe"), "-v", "error", "-select_streams", "v:0", "-read_intervals", f"%+#{packets}",
                          "-show_entries", "packet=pts,flags", "-of", "csv=p=0", str(video)],
                         capture_output=True, text=True, check=True).stdout
    for line in out.split():
        pts, _, flags = line.partition(",")
        if (pts.lstrip("-").isdigit() and int(pts) < 0) or "D" in flags:
            return False
    return True


def _copy(video, out, start_time, count):
    """Stream-copy `count` frames (0 = to the end) starting at the keyframe whose exact pts_time is `start_time`
    (None = the beginning). Seeking past the keyframe, even by a fraction of a frame, shifts the piece by a frame."""
    cmd = [_exe("ffmpeg"), "-v", "error", "-y"]
    if start_time and float(start_time) > 0:
        cmd += ["-ss", start_time]
    cmd += ["-i", str(video), "-map", "0:v:0", "-c", "copy"]
    if count:
        cmd += ["-frames:v", str(count)]
    subprocess.run(cmd + [str(out)], check=True)
    return out
