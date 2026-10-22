"""ffmpeg helpers: probing, frame decoding and encoding."""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from .common import ROOT, Progress, cpu_share

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".ts", ".m4v", ".wmv", ".webm"}
FFMPEG_THREADS = "auto"  # --ffmpeg-threads


def ffmpeg_threads():
    """Threads for one ffmpeg decoder or encoder (and its filters); 0 = ffmpeg's own default.
    ffmpeg sizes its pools from the host's core count: 104 decoding and 76 encoding threads per process on a 208-core
    host under a 25-core container quota. Four part processes, each decoding and encoding, then used up the quota:
    throttled in 30 % of the scheduler periods while the GPU was busy 38 % of the time (1304, RTX 5090). Decoding
    1500 frames took 4.1 s with the default and 4.4 s with 4 threads, but 67 s of CPU against 15. auto = half of the
    process's CPU share (common.cpu_share), at least 2: 3 per ffmpeg for 4 processes on 25 cores (libx264 1080p
    still encodes 24 fps, more than the ~10 fps erasing feeds it), 12 for one process (libx264 ~55 fps, default 65)."""
    if FFMPEG_THREADS != "auto":
        return max(0, int(FFMPEG_THREADS))
    return max(2, cpu_share() // 2)


def _threads_opts():
    """ffmpeg options limiting the next input's decoder and all filters to ffmpeg_threads()"""
    n = ffmpeg_threads()
    return ["-filter_threads", str(n), "-threads", str(n)] if n else []


def _exe(name):
    local = ROOT / "ffmpeg" / (name + (".exe" if os.name == "nt" else ""))
    if local.exists():
        return str(local)
    p = shutil.which(name)
    if not p:
        sys.exit(f"{name} not found: install ffmpeg and add it to PATH, or put it in {ROOT / 'ffmpeg'}")
    return p


def _stream_info(path):
    out = subprocess.run(
        [_exe("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,color_space,color_primaries,color_transfer,"
         "color_range,sample_aspect_ratio,pix_fmt:format=duration", "-of", "json", str(path)],
        capture_output=True, check=True).stdout
    j = json.loads(out)
    return j["streams"][0], float(j["format"].get("duration", 0) or 0)


def _rate(s):
    """'num/den' -> (float, the string) or (0.0, None)"""
    try:
        num, den = (float(v) for v in str(s).split("/"))
        return (num / den, s) if num > 0 and den > 0 else (0.0, None)
    except ValueError:
        return 0.0, None


def _timing(s, dur):
    """-> (fps, exact rate string, variable frame rate?, approximate frame count)"""
    avg, avg_s = _rate(s.get("avg_frame_rate"))
    r, r_s = _rate(s.get("r_frame_rate"))
    fps, rate = (avg, avg_s) if avg else (r, r_s)
    # A variable-frame-rate source (phone or screen recordings) decodes to more or fewer frames than its average rate
    # implies; everything downstream counts frames at a constant rate, so such sources are resampled to `fps`
    vfr = bool(avg and r) and abs(avg - r) > 0.005 * r
    n = (0 if vfr else int(s.get("nb_frames") or 0)) or int(round(dur * fps))
    return fps, rate, vfr, n


def probe(path):
    """-> (width, height, fps, approximate frame count). For a variable-frame-rate video, fps is the average rate and
    the count is that of the constant-rate stream read_frames() produces."""
    s, dur = _stream_info(path)
    fps, _, _, n = _timing(s, dur)
    return int(s["width"]), int(s["height"]), fps, n


def packet_count(path):
    """Exact number of video frames of a file without decoding it (one packet per frame), or 0 if unknown. A part cut
    with `-c copy` has no frame count in its header, and its container duration overshoots by up to 3 frames."""
    r = subprocess.run([_exe("ffprobe"), "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return int(r.stdout.strip().split()[0])
    except (IndexError, ValueError):
        return 0


# ffprobe colour space -> scale filter matrix name
_MATRIX = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601", "fcc": "fcc", "smpte240m": "smpte240m",
           "bt2020nc": "bt2020", "bt2020c": "bt2020"}
_TAGS = {"bt709": ("bt709", "bt709", "bt709"), "bt601": ("smpte170m", "smpte170m", "smpte170m"),
         "bt2020": ("bt2020nc", "bt2020", "bt709")}


def colour(path):
    """-> (matrix for the scale filter, range 'tv'/'pc', [ffmpeg output tag options]) of the video's YUV<->RGB
    conversion. Untagged videos get what players assume: BT.709 from 720 lines up, BT.601 below.

    Decoding with the source's matrix and encoding with the same one (and tagging the output) keeps colours unchanged;
    ffmpeg's defaults decoded a tagged BT.709 film with BT.709 but encoded with BT.601 and no tag, which shifted every
    saturated colour (a test bar moved from YCbCr 84/154/158 to 93/150/156)."""
    s, _ = _stream_info(path)
    s = {k: v for k, v in s.items() if v not in ("unknown", "reserved", "unspecified")}
    m = _MATRIX.get(s.get("color_space")) or ("bt709" if int(s["height"]) >= 720 else "bt601")
    rng = "pc" if s.get("color_range") in ("pc", "jpeg") else "tv"
    cs, pri, trc = _TAGS.get(m, ("bt709",) * 3)
    tags = ["-colorspace", s.get("color_space") if s.get("color_space") in _MATRIX else cs,
            "-color_primaries", s.get("color_primaries") or pri,
            "-color_trc", s.get("color_transfer") or trc,
            "-color_range", rng]
    return m, rng, tags


def _cfr(path):
    """Output options for decoding `path` frame by frame: a variable-frame-rate video is resampled to its average
    rate; any other video yields every decoded frame exactly once (passthrough). Without passthrough the rawvideo
    output runs in cfr mode and fills the time between the start of the file and a later first video frame (audio
    starting 0.045 s before the picture) with a copy of the first frame; with the frame count capped at n, the last
    real frame was then dropped."""
    s, dur = _stream_info(path)
    _, rate, vfr, _ = _timing(s, dur)
    return ["-fps_mode", "cfr", "-r", rate] if vfr else ["-fps_mode", "passthrough"]


def sar(path):
    """Sample (pixel) aspect ratio of the video of `path` as 'num/den', or None for square or unknown pixels"""
    s, _ = _stream_info(path)
    num, _, den = str(s.get("sample_aspect_ratio") or "").partition(":")
    if num.isdigit() and den.isdigit() and int(num) and int(den) and int(num) != int(den):
        return f"{num}/{den}"
    return None


def video_offset(path):
    """Seconds between the start of `path` (its earliest stream) and its first video frame; 0.0 if they coincide.
    The encoder writes the picture from time 0; restore_start() puts it back at this offset."""
    out = subprocess.run([_exe("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=start_time:format=start_time", "-of", "json", str(path)],
                         capture_output=True, check=True).stdout
    j = json.loads(out)
    try:
        d = float(j["streams"][0].get("start_time") or 0) - float(j.get("format", {}).get("start_time") or 0)
    except (IndexError, ValueError):
        return 0.0
    return d if d > 0.0005 else 0.0


def restore_start(src, path):
    """Shift the video of the finished `path` so it starts as late after the audio as in `src` (the audio of `path`
    is a copy of the source audio and kept its place). Without this the picture played 45 ms early (seen on a
    source whose video starts at 0.045 s). Stream copy, no re-encode; returns the shift in seconds."""
    d = video_offset(src)
    if not d:
        return 0.0
    path = Path(path)
    tmp = path.with_name(path.stem + ".start" + path.suffix)
    try:
        subprocess.run([_exe("ffmpeg"), "-v", "error", "-y", "-itsoffset", f"{d:.6f}", "-i", str(path), "-i", str(path),
                        "-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-movflags", "+faststart", str(tmp)], check=True)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return d


# Exact YUV<->RGB rounding. With the default flags every decode/encode round trip made the picture ~1.5 luma
# levels darker, so a film that went through three passes ended up ~4.4 levels darker than its source.
# The flags must follow the input: placed before -i, ffmpeg 6.1 silently ignores them.
SWS_FLAGS = "accurate_rnd+full_chroma_int+bitexact"
SWS = ["-sws_flags", SWS_FLAGS]


def _to_bgr(path):
    """scale filter converting `path`'s YUV to BGR with its own matrix and range"""
    m, rng, _ = colour(path)
    return f"scale=flags={SWS_FLAGS}:in_color_matrix={m}:in_range={rng}:out_range=pc,format=bgr24"


def read_frames(path, w, h, crop=None, start=0, count=0):
    """Yield frames as read-only BGR ndarrays. With crop=(y0, y1) only that horizontal band is decoded.
    start/count: decode only frames start .. start+count-1 (frame-exact for the constant-rate files we write)."""
    cmd = [_exe("ffmpeg"), "-v", "error", *_threads_opts()]
    if start:
        cmd += ["-ss", f"{(start - 0.25) / _fps(path):.6f}"]  # a quarter frame early: output starts exactly at frame `start`
    cmd += ["-i", str(path), *SWS, "-map", "0:v:0", *_cfr(path)]  # -sws_flags only takes effect after the input
    if count:
        cmd += ["-frames:v", str(count)]
    vf = _to_bgr(path)
    if crop:
        y0, y1 = crop
        vf = f"crop={w}:{y1 - y0}:0:{y0}," + vf
        h = y1 - y0
    cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=int(w * h * 3 * 4))
    size = w * h * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        _stop(p)


def can_compose_yuv(path):
    """Can a result be composed in the source's own YUV (yuvcomp)? -> the matrix name, or None (8-bit 4:2:0, limited range,
    a known matrix, even size: otherwise the erased BGR frames are encoded as a whole, as before)"""
    from . import yuvcomp
    s, _ = _stream_info(path)
    m, rng, _ = colour(path)
    ok = yuvcomp.supported(m, s.get("pix_fmt"), rng, int(s["width"]), int(s["height"]))
    return m if ok else None


def read_yuv(path, w, h, count=0, start=0):
    """Yield the decoder's frames as flat uint8 arrays of yuv420p (w*h*3/2 values), no conversion; frame for frame what
    read_frames() yields as BGR (same options, also start/count)"""
    cmd = [_exe("ffmpeg"), "-v", "error", *_threads_opts()]
    if start:
        cmd += ["-ss", f"{(start - 0.25) / _fps(path):.6f}"]
    cmd += ["-i", str(path), "-map", "0:v:0", *_cfr(path)]
    if count:
        cmd += ["-frames:v", str(count)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 2)
    size = w * h * 3 // 2
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8)
    finally:
        _stop(p)


def _stop(p):
    """End a reader process; kill it if the consumer stopped early, so no decoder is left behind"""
    if p.poll() is None:
        p.kill()
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
    cmd = [_exe("ffmpeg"), "-v", "error", *_threads_opts(), "-i", str(path), "-map", "0:v:0", *_cfr(path), "-vf", f"scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]  # same frame numbering as read_frames
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
        _stop(p)
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
    (GPU inference in the main thread) instead of alternating with it. When the consumer stops early (an error
    further down the chain), the thread stops too and closes `gen`, which ends its ffmpeg process."""
    q = queue.Queue(maxsize)
    done = object()
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                pass
        return False

    def worker():
        try:
            for item in gen:
                if not put(item):
                    return
            put(done)
        except Exception as e:
            put(e)
        finally:
            if hasattr(gen, "close"):
                gen.close()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
        t.join(timeout=10)


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
        while True:
            buf = self._q.get()
            if buf is None:
                return
            if self._err is None:
                try:
                    self.proc.stdin.write(buf)
                except Exception as e:  # e.g. the encoder exited: keep draining so that write()/close() never block
                    self._err = e

    def _raise(self):
        e = self._err
        raise RuntimeError(f"ffmpeg encoder stopped (code {self.proc.poll()}): {e!r}") from e

    def write(self, buf):
        if self._err:
            self._raise()
        self._q.put(buf)

    def close(self):
        self._q.put(None)
        self._t.join()
        try:
            self.proc.stdin.close()
        except OSError as e:  # broken pipe: the encoder already exited
            self._err = self._err or e
        self.proc.wait()
        if self._err:
            self._raise()


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


def duration_check(src, dst, tol=0.5):
    """Warning text if the video of `dst` is not as long as `src` (audio would drift), else None"""
    a, b = _video_duration(src), _video_duration(dst)
    if a and b and abs(a - b) > tol:
        return f"output video lasts {b:.2f}s but the source {a:.2f}s; audio and subtitles may drift"
    return None


def _video_duration(path):
    _, dur = _stream_info(path)
    try:
        return float(subprocess.run([_exe("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=duration", "-of", "csv=p=0", str(path)],
                                    capture_output=True, text=True, check=True).stdout.strip().strip(",")) or dur
    except ValueError:
        return dur


def keyframes(path):
    """{frame index: exact pts_time string} of the keyframes of a constant-frame-rate video. Indices count from the
    first video frame, which in our results can come a little after the start of the file (restore_start)."""
    fps = _fps(path)
    out = subprocess.run([_exe("ffprobe"), "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
                          "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    times = [x.strip(",") for x in out.split() if x.strip(",")]
    v0 = float(times[0]) if times else 0.0  # the first frame of a video is a keyframe
    return {round((float(t) - v0) * fps): t for t in times}


def _exact_rate(src, fps):
    """The frame rate to tell the encoder: the source's own rate string ("24000/1001") when `fps` is that rate, else
    `fps` with six decimals. The decimal 23.976024 of an NTSC rate was turned into 991001/41333 by ffmpeg (29.97 into
    979001/32666, 59.94 into 979001/16333): the picture was right, but the result stated another rate than the source."""
    try:
        s, dur = _stream_info(src)
        value, rate = _timing(s, dur)[:2]
        if rate and abs(value - fps) <= 1e-4 * fps:
            return rate
    except Exception:
        pass
    return f"{fps:.6f}"


def open_writer(src, dst, w, h, fps, encoder, crf, audio=True, yuv=False):
    """ffmpeg process that encodes raw BGR frames from stdin and copies the audio of `src` (audio=False: video only).
    yuv=True: the frames are yuv420p already (yuvcomp.compose) in the source's matrix and range: no conversion."""
    if encoder in ("h264_nvenc", "hevc_nvenc"):
        venc = ["-c:v", encoder, "-preset", "p5", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    else:
        venc = ["-c:v", encoder, "-preset", "medium", "-crf", str(crf)]
    m, rng, tags = colour(src)  # encode with the source's matrix and range, and say so in the file
    aspect = sar(src)  # non-square pixels (anamorphic sources): keep the source's display aspect ratio
    n = ffmpeg_threads()
    if n:
        venc += ["-threads", str(n)]
    # a raw YUV input says what its frames are (matrix, range): untagged, ffmpeg takes them for something else and
    # converts them to the colour description of the output (a tagged BT.709 film changed by 21 grey levels on average)
    cmd = [_exe("ffmpeg"), "-v", "error", "-y", *(["-filter_threads", str(n)] if n else []),
           "-f", "rawvideo", *(tags if yuv else []), "-pix_fmt", "yuv420p" if yuv else "bgr24", "-s", f"{w}x{h}",
           "-r", _exact_rate(src, fps), "-i", "-"]
    cmd += ["-i", str(src), *SWS, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy"] if audio else [*SWS, "-map", "0:v:0"]
    if yuv:
        if aspect:
            cmd += ["-vf", f"setsar={aspect}"]
    else:
        cmd += ["-vf", f"scale=flags={SWS_FLAGS}:in_range=pc:out_color_matrix={m}:out_range={rng},format=yuv420p"
                       + (f",setsar={aspect}" if aspect else "")]
    cmd += [*venc, *tags, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
