"""--probe: look at every video of a batch before processing it and suggest its options.

A few minutes per film, no erasing. For each video it reports, in NAME.probe.json and in one line of the generated
batch file (probe-batch.txt, the --list format):

1. hardcoded subtitles: OCR of ~200 frames spread over the film (opening and credits left out), first in the
   bottom band, then in the top band; one or two lines (bilingual), where they sit
2. logo: a static overlay in a corner (watermark.detect), which corner
3. logo mask sanity: in a dark film the detector found only part of the logo (2771: 2060 pixels of ~6000), so the
   rest stayed in the picture. The detection is repeated with a looser consistency; many more pixels there means
   the strict mask is probably incomplete. A mask of the same frame size from --logo-library that contains the found
   one is suggested instead
4. black letterbox bars (reported only)
5. time stamps: video frames against duration x nominal frame rate (dropped or irregular frames), audio frames
   against the audio duration (2829: its audio held 10.8 s less sound than its time line), audio/video start offset
"""
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import watermark
from .ads import find_bars
from .common import log
from .pipeline import parse_band
from .subtitles import frame_boxes, subtitle_line
from .video import _exe

SUB_SAMPLES = 200
SUB_MIN_HIT = 0.12   # share of sampled frames with text on the subtitle line; dialogue films are far above it
SUB_MIN_LINE = 0.5   # share of the text boxes that sit on the detected line(s): scattered scene text is below it
LOOSE_RATIO = 1.8    # loose / strict logo mask size above this -> strict mask probably incomplete
AUDIO_FRAME = {"aac": 1024, "mp3": 1152, "ac3": 1536, "eac3": 1536, "mp2": 1152}


def _rate(s):
    try:
        num, den = (float(x) for x in s.split("/"))
        return num / den if den else 0.0
    except (ValueError, AttributeError):
        return 0.0


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _streams(video):
    out = subprocess.run([_exe("ffprobe"), "-v", "error", "-show_entries",
                          "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,"
                          "duration,start_time,field_order,sample_rate:format=duration",
                          "-of", "json", str(video)], capture_output=True, check=True).stdout
    j = json.loads(out)
    v = next((s for s in j["streams"] if s.get("codec_type") == "video"), None)
    a = next((s for s in j["streams"] if s.get("codec_type") == "audio"), None)
    return v, a, _float(j.get("format", {}).get("duration"))


def _count_packets(video, sel):
    """Packets of the first stream of type `sel` ('v' or 'a'); for containers without a frame count (mkv)"""
    out = subprocess.run([_exe("ffprobe"), "-v", "error", "-select_streams", f"{sel}:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(video)],
                         capture_output=True, text=True).stdout.strip().strip(",")
    return int(out) if out.isdigit() else None


def check_timing(video):
    """Item 5 (and the basic facts) -> dict with 'warnings'"""
    v, a, fdur = _streams(video)
    if v is None:
        raise ValueError("no video stream")
    info = {"codec": v.get("codec_name"), "width": v.get("width"), "height": v.get("height"),
            "field_order": v.get("field_order"), "warnings": []}
    r_fps, avg_fps = _rate(v.get("r_frame_rate")), _rate(v.get("avg_frame_rate"))
    vdur = _float(v.get("duration")) or fdur or 0.0
    nb = int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else _count_packets(video, "v")
    info.update(fps_nominal=round(r_fps, 4), fps_average=round(avg_fps, 4), video_duration=round(vdur, 3),
                video_frames=nb)
    if nb and r_fps and vdur:
        expected = vdur * r_fps
        missing = expected - nb
        info["video_frames_missing"] = round(missing)
        if abs(missing) > 0.001 * expected:
            info["warnings"].append(
                f"irregular video time stamps: {nb} frames for {vdur:.1f} s at {r_fps:g} fps "
                f"({missing:+.0f} frames, {missing / r_fps:+.1f} s); check audio/video sync of the result")
    if v.get("field_order") not in (None, "progressive", "unknown"):
        info["warnings"].append(f"field order '{v.get('field_order')}' (interlaced flag): check the result for combing")
    if a is not None:
        adur = _float(a.get("duration")) or fdur or 0.0
        sr = _float(a.get("sample_rate")) or 0.0
        na = int(a["nb_frames"]) if str(a.get("nb_frames", "")).isdigit() else _count_packets(video, "a")
        info.update(audio_codec=a.get("codec_name"), audio_duration=round(adur, 3))
        fs = AUDIO_FRAME.get(a.get("codec_name"))
        if fs and sr and na and adur:
            sound = na * fs / sr
            info["audio_sound_seconds"] = round(sound, 2)
            if adur - sound > 0.5:
                info["warnings"].append(f"audio has gaps: {adur - sound:.1f} s of its {adur:.0f} s time line hold no "
                                        "sound; never retime this audio with atempo")
        off = (_float(v.get("start_time")) or 0.0) - (_float(a.get("start_time")) or 0.0)
        info["av_start_offset"] = round(off, 3)
        if abs(off) > 0.1:
            info["warnings"].append(f"video starts {off:+.2f} s after the audio")
        if abs(vdur - adur) > 1.0:
            info["warnings"].append(f"video {vdur:.1f} s and audio {adur:.1f} s differ by {vdur - adur:+.1f} s")
    else:
        info["warnings"].append("no audio stream")
    return info


def _grab_band(video, t, W, band):
    h = band[1] - band[0]
    r = subprocess.run([_exe("ffmpeg"), "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                        "-vf", f"crop={W}:{h}:0:{band[0]}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                       capture_output=True)
    if len(r.stdout) != W * h * 3:
        return None
    return np.frombuffer(r.stdout, np.uint8).reshape(h, W, 3)


def _subtitles_in(video, W, H, band, duration, ocr, args):
    """OCR statistics of one band -> dict(hit, on_line, line)"""
    times = np.linspace(duration * 0.03, duration * 0.95, SUB_SAMPLES)
    with ThreadPoolExecutor(4) as ex:
        crops = list(ex.map(lambda t: _grab_band(video, t, W, band), times))
    min_h = max(8, int(H * args.min_height))
    per_frame = [frame_boxes(ocr(c), band[0], W, min_h, args.min_score) for c in crops if c is not None]
    if not per_frame:
        return {"hit": 0.0, "on_line": 0.0, "line": None}
    line = subtitle_line(per_frame)
    boxes = [b for f in per_frame for b in f]
    lines = [line[i:i + 2] for i in range(0, len(line), 2)] if line else []
    on = [b for b in boxes if any(abs((b[1] + b[3]) / 2 - y) <= 0.5 * h for y, h in lines)]
    hit = sum(1 for f in per_frame if any(b in on for b in f)) / len(per_frame)
    return {"hit": round(hit, 3), "on_line": round(len(on) / len(boxes), 3) if boxes else 0.0, "line": line}


def check_subtitles(video, W, H, duration, ocr, args):
    """Item 1 -> dict(present, lines, band, position, ...); tries the bottom band first, then the top band"""
    for name, spec in (("bottom", args.band), ("top", "0,0.3")):
        band = parse_band(spec, H)
        s = _subtitles_in(video, W, H, band, duration, ocr, args)
        found = s["hit"] >= SUB_MIN_HIT and s["on_line"] >= SUB_MIN_LINE and s["line"] is not None
        res = {"present": found, "band": spec, "where": name, "frames_with_text": s["hit"],
               "boxes_on_line": s["on_line"]}
        if found:
            line = s["line"]
            res["lines"] = 2 if len(line) == 4 else 1
            res["line_y"] = [round(line[i] / H, 3) for i in range(0, len(line), 2)]
            res["text_height"] = [round(line[i] / H, 3) for i in range(1, len(line), 2)]
            return res
        if name == "bottom":
            first = res
    first["top_frames_with_text"] = res["frames_with_text"]
    return first


def _corner(mask, W, H):
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean() / W, ys.mean() / H
    return ("top" if cy < 0.5 else "bottom") + "-" + ("left" if cx < 0.5 else "right")


def _library_match(mask, W, H, library):
    """A mask from `library` (PNG files, white = logo) of the same frame size that contains `mask` and is larger"""
    best = None
    for p in sorted(Path(library).glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (H, W):
            continue
        lib = img > 127
        inside = (mask & lib).sum() / max(1, mask.sum())
        if inside >= 0.9 and lib.sum() >= 1.2 * mask.sum() and (best is None or inside > best[1]):
            best = (str(p), float(inside), int(lib.sum()))
    return best


def check_logo(video, W, H, duration, library=None):
    """Items 2 and 3 -> dict(found, corner, pixels, warnings, suggest)"""
    res = {"warnings": []}
    strict = watermark.detect(video, W, H, duration)
    loose = watermark.detect(video, W, H, duration, consistency=0.6)
    res["found"] = strict is not None
    res["pixels"] = int(strict.sum()) if strict is not None else 0
    res["pixels_loose"] = int(loose.sum()) if loose is not None else 0
    ref = strict if strict is not None else loose
    if ref is None:
        return res
    ys, xs = np.nonzero(ref)
    res["corner"] = _corner(ref, W, H)
    res["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    if strict is None:
        res["warnings"].append(f"no logo with the normal test, but a looser one finds {res['pixels_loose']} pixels "
                               f"({res['corner']}): check the film for a faint logo")
    elif res["pixels_loose"] > LOOSE_RATIO * res["pixels"]:
        res["warnings"].append(f"logo mask probably incomplete: {res['pixels']} pixels, a looser test finds "
                               f"{res['pixels_loose']} (dark film?)")
    if library and (strict is None or res["warnings"]):
        m = _library_match(ref, W, H, library)
        if m:
            res["suggest"] = m[0]
            res["warnings"].append(f"suggesting the library mask {Path(m[0]).name} ({m[2]} pixels, contains "
                                   f"{m[1]:.0%} of the detected logo)")
    return res


def check_bars(video, duration, samples=60):
    """Item 4 -> (top, bottom) black bar rows in pixels, or None"""
    w, h = 320, 180
    rows = []
    for t in np.linspace(duration * 0.05, duration * 0.95, samples):
        r = subprocess.run([_exe("ffmpeg"), "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                            "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"], capture_output=True)
        if len(r.stdout) == w * h:
            rows.append(np.frombuffer(r.stdout, np.uint8).reshape(h, w).max(axis=1) < 40)
    if len(rows) < 10:
        return None
    return find_bars(np.mean(rows, axis=0))


def probe_video(video, ocr, args):
    video = Path(video)
    t0 = time.time()
    log(f"\n==> probe {video.name}")
    info = check_timing(video)
    W, H = info["width"], info["height"]
    duration = info["video_duration"]
    info["subtitles"] = check_subtitles(video, W, H, duration, ocr, args)
    info["logo"] = check_logo(video, W, H, duration, args.logo_library)
    bars = check_bars(video, duration)
    info["letterbox"] = None if bars is None else [round(b * H / 180) for b in bars]
    info["warnings"] += info["logo"].pop("warnings")

    opts = []
    sub = info["subtitles"]
    if not sub["present"]:
        opts += ["--subtitles", "off"]
    elif sub["where"] == "top":
        opts += ["--band", sub["band"]]
    logo = info["logo"]
    if logo.get("suggest"):
        opts += ["--watermark", logo["suggest"]]
    elif not logo["found"]:
        opts += ["--watermark", "off"]
    info["options"] = opts
    info["probe_seconds"] = round(time.time() - t0)
    log("  " + summary(info))
    for w in info["warnings"]:
        log(f"  WARN {w}")
    return info


def summary(info):
    sub = info["subtitles"]
    s = (f"subtitles: {sub['lines']} line(s) at the {sub['where']}" if sub["present"]
         else f"subtitles: none (text on {sub['frames_with_text']:.0%} of the samples)")
    logo = info["logo"]
    lg = f"logo: {logo['corner']}" if logo["found"] else "logo: none"
    lb = f"letterbox {info['letterbox'][0]}+{info['letterbox'][1]} px" if info["letterbox"] else "no letterbox"
    return (f"{info['width']}x{info['height']} {info['codec']} {info['fps_average']:g} fps, {s}, {lg}, {lb}; "
            f"options: {' '.join(info['options']) or '(defaults)'}")


def write_outputs(results, out_dir):
    """NAME.probe.json per video and probe-batch.txt (the --list format, WARN lines as comments)"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# generated by clean-clear --probe on {time.strftime('%Y-%m-%d %H:%M')}",
             "# check the WARN comments, then run: clean-clear --list probe-batch.txt -o OUTPUT", ""]
    for video, info in results:
        (out_dir / (Path(video).stem + ".probe.json")).write_text(
            json.dumps(info, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        lines.append(f"# {Path(video).name}: {summary(info)}")
        lines += [f"#   WARN {w}" for w in info["warnings"]]
        opts = " ".join(o if " " not in o else f'"{o}"' for o in info["options"])
        lines.append(f'"{Path(video).resolve().as_posix()}" {opts}'.rstrip())
        lines.append("")
    path = out_dir / "probe-batch.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
