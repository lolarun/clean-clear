"""--check: compare a result with its source against the delivery requirements (contract annex 1, section 3) and
write qc.json plus two contact sheets (source | result) for a person or an agent to look at.

Measured and judged here: frame count, resolution, frame rate, pixel aspect ratio, colour description, codec and
container, video bitrate (not below the source), duration, video start against the audio, audio stream, file size,
SRT validity. Not judged here: whether subtitles or logos are gone and the fills look right; the sheets are for that.
"""
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from .common import __version__, log
from .video import _exe, packet_count, video_offset

MAX_SIZE = 20 * 1024 ** 3  # contract: no file over 20 GB
SHEET_POINTS = 8           # evenly spread time points ...
DARK_POINTS = 2            # ... plus the darkest scenes (an incomplete logo mask shows there)
START_TOLERANCE = 0.005    # s, video start against the audio (mp4 time bases round to a few ms)

# common characters that only exist in traditional Chinese; their share tells zh-hant from zh-hans
TRADITIONAL = set("們這個說為來時會對與後還嗎麼裡點開關見長電東車門間問題實現發經過應讓從學國話聽寫讀錢買賣愛氣頭樣聲處覺幾邊")


def _probe(path):
    out = subprocess.run(
        [_exe("ffprobe"), "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,avg_frame_rate,sample_aspect_ratio,display_aspect_ratio,pix_fmt,"
         "color_space,color_primaries,color_transfer,color_range,bit_rate,duration,sample_rate,channels"
         ":format=format_name,duration,bit_rate,size", "-of", "json", str(path)],
        capture_output=True, check=True).stdout
    return json.loads(out)


def _num(x, cast=float):
    try:
        return cast(x)
    except (TypeError, ValueError):
        return None


def _rate(s):
    num, _, den = str(s or "").partition("/")
    n, d = _num(num), _num(den or 1)
    return n / d if n and d else None


def _ratio(s):
    """'4:3' -> '4:3'; square, unknown or '0:1' -> '1:1'"""
    num, _, den = str(s or "").partition(":")
    return s if num.isdigit() and den.isdigit() and int(num) and int(den) and num != den else "1:1"


def measure(path):
    """Technical description of a video file, as compared by check()"""
    j = _probe(path)
    streams = j.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})
    f = j.get("format", {})
    size = _num(f.get("size"), int) or Path(path).stat().st_size
    vdur = _num(v.get("duration")) or _num(f.get("duration"))
    vbr = _num(v.get("bit_rate"), int)
    estimated = False
    if not vbr and vdur:  # mkv and some mp4 sources: estimate from the file size minus the audio
        vbr = int(size * 8 / vdur - (_num(a.get("bit_rate"), int) or 0))
        estimated = True
    return {
        "width": v.get("width"), "height": v.get("height"),
        "frame_rate": round(_rate(v.get("avg_frame_rate")) or 0, 4),
        "frame_rate_exact": v.get("avg_frame_rate"),  # "24000/1001": compared as written, not as a number
        "frames": packet_count(path),
        "sar": _ratio(v.get("sample_aspect_ratio")), "dar": v.get("display_aspect_ratio"),
        "codec": v.get("codec_name"), "pix_fmt": v.get("pix_fmt"), "container": f.get("format_name"),
        "color": {k: v.get(k) for k in ("color_space", "color_primaries", "color_transfer", "color_range")},
        "video_bitrate_kbps": round(vbr / 1000) if vbr else None, "bitrate_estimated": estimated,
        "video_duration": round(vdur, 3) if vdur else None,
        "audio": ({"codec": a.get("codec_name"), "sample_rate": a.get("sample_rate"), "channels": a.get("channels"),
                   "duration": round(_num(a.get("duration")) or _num(f.get("duration")) or 0, 3)} if a else None),
        "video_start_after_audio": round(video_offset(path), 3),
        "size_bytes": size,
    }


def md5(path, block=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def read_srt(path):
    """-> list of (start s, end s, text); raises ValueError on a malformed file"""
    cues = []
    t = r"(\d\d):(\d\d):(\d\d),(\d{3})"
    for block in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8-sig").strip()):
        lines = block.strip().splitlines()
        if not lines:
            continue
        m = re.match(t + r" --> " + t, lines[1] if len(lines) > 1 else "")
        if not lines[0].strip().isdigit() or not m:
            raise ValueError(f"malformed cue: {block[:60]!r}")
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        cues.append((start, end, "\n".join(lines[2:])))
    return cues


def subtitle_language(texts):
    """Language code(s) of the subtitle texts: zh-hans, zh-hant, en, 'zh-hans;en' for bilingual, none"""
    texts = [t for t in texts if t.strip()]
    if not texts:
        return "none"
    cjk = [c for t in texts for c in t if "一" <= c <= "鿿"]
    latin = sum(1 for t in texts if len(re.findall(r"[A-Za-z]{2,}", t)) >= 2)
    langs = []
    if len(cjk) >= 10:
        langs.append("zh-hant" if sum(c in TRADITIONAL for c in cjk) > 0.05 * len(cjk) else "zh-hans")
    if latin >= 0.3 * len(texts):
        langs.append("en")
    return ";".join(langs) or "none"


def _frame(path, t, w, h):
    r = subprocess.run([_exe("ffmpeg"), "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                        "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
    if len(r.stdout) != w * h * 3:
        return None
    return np.frombuffer(r.stdout, np.uint8).reshape(h, w, 3)


def _sample_times(src, duration):
    times = [duration * (k + 0.5) / SHEET_POINTS for k in range(SHEET_POINTS)]
    lum = []
    for t in np.linspace(duration * 0.03, duration * 0.97, 48):
        f = _frame(src, t, 64, 36)
        if f is not None and 8 < f.mean() < 70:  # dark but not black (fades, black frames tell nothing)
            lum.append((f.mean(), t))
    for _, t in sorted(lum)[:DARK_POINTS]:
        if all(abs(t - x) > duration * 0.02 for x in times):
            times.append(t)
    return sorted(times)


def contact_sheets(src, dst, out_dir, width, height, duration):
    """sheet_full.jpg: whole frames, source | result; sheet_detail.jpg: the subtitle band (bottom 32 %) and the
    logo band (top 15 %) of the same frames, larger. Returns the file names and the time points."""
    import cv2
    times = _sample_times(src, duration)
    fw = 640
    fh = max(2, round(height * fw / width / 2) * 2)
    dw = min(960, width)
    dh = max(2, round(height * dw / width / 2) * 2)
    full, detail = [], []
    for t in times:
        a, b = _frame(src, t, fw, fh), _frame(dst, t, fw, fh)
        c, d = _frame(src, t, dw, dh), _frame(dst, t, dw, dh)
        if a is None or b is None or c is None or d is None:
            continue
        label = f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"
        row = np.hstack([a, np.full((fh, 6, 3), 255, np.uint8), b])
        cv2.putText(row, f"{label}  source | result", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        full.append(row)
        sub, top = slice(int(dh * 0.68), dh), slice(0, int(dh * 0.15))
        sep = np.full((6, dw * 2 + 6, 3), 255, np.uint8)
        vsep = lambda h: np.full((h, 6, 3), 255, np.uint8)
        band = np.vstack([np.hstack([c[top], vsep(top.stop), d[top]]), sep,
                          np.hstack([c[sub], vsep(dh - sub.start), d[sub]])])
        cv2.putText(band, f"{label}  source | result  (top: logo band, bottom: subtitle band)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        detail.append(band)
        detail.append(np.zeros((12, dw * 2 + 6, 3), np.uint8))
    names = []
    for name, rows in (("sheet_full.jpg", full), ("sheet_detail.jpg", detail)):
        if rows:
            cv2.imwrite(str(Path(out_dir) / name), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
            names.append(name)
    return names, [round(t, 2) for t in times]


def check(src, dst, out_dir=None, srt=None, sheets=True):
    """Compare `dst` with `src`; write qc.json (and the sheets) into out_dir; -> the qc dict ('pass': all checks ok)"""
    src, dst = Path(src), Path(dst)
    out_dir = Path(out_dir) if out_dir else dst.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if srt is None:
        cand = dst.with_name(dst.stem.removesuffix("_clean") + ".srt")
        srt = cand if cand.exists() else None
    t0 = time.time()
    s, r = measure(src), measure(dst)
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("frames", s["frames"] == r["frames"], f"source {s['frames']}, result {r['frames']}")
    add("resolution", (s["width"], s["height"]) == (r["width"], r["height"]),
        f"source {s['width']}x{s['height']}, result {r['width']}x{r['height']}")
    # the rate as the file states it: 24000/1001 and 991001/41333 are 1e-7 apart as numbers, but not the same rate
    add("frame_rate", s["frame_rate"] and s["frame_rate_exact"] == r["frame_rate_exact"]
        and abs(r["frame_rate"] - s["frame_rate"]) <= 1e-4 * s["frame_rate"],
        f"source {s['frame_rate_exact']} ({s['frame_rate']}), result {r['frame_rate_exact']} ({r['frame_rate']})")
    add("aspect", s["sar"] == r["sar"], f"SAR source {s['sar']}, result {r['sar']} (DAR {s['dar']} / {r['dar']})")
    diff = {k: (s["color"][k], r["color"][k]) for k in s["color"] if s["color"][k] and s["color"][k] != r["color"][k]}
    add("colour", not diff, "same" if not diff else f"source/result differ: {diff}")
    add("codec", r["codec"] == "h264" and any(x in (r["container"] or "") for x in ("mp4", "mov")),
        f"{r['codec']} in {r['container']}")
    if s["video_bitrate_kbps"]:
        add("bitrate", r["video_bitrate_kbps"] >= s["video_bitrate_kbps"],
            f"source {s['video_bitrate_kbps']} kbps{' (estimated)' if s['bitrate_estimated'] else ''}, "
            f"result {r['video_bitrate_kbps']} kbps")
    else:
        add("bitrate", False, "source bitrate unknown: compare by hand")
    tol = 1.5 / (s["frame_rate"] or 25)
    add("duration", s["video_duration"] and abs(r["video_duration"] - s["video_duration"]) <= tol,
        f"video source {s['video_duration']} s, result {r['video_duration']} s")
    add("av_start", abs(r["video_start_after_audio"] - s["video_start_after_audio"]) <= START_TOLERANCE,
        f"video starts after the audio by: source {s['video_start_after_audio']} s, "
        f"result {r['video_start_after_audio']} s")
    sa, ra = s["audio"], r["audio"]
    if sa:
        same = ra and all(sa[k] == ra[k] for k in ("codec", "sample_rate", "channels")) \
            and abs(sa["duration"] - ra["duration"]) <= 0.05
        add("audio", same, f"source {sa}, result {ra}")
    add("size", r["size_bytes"] <= MAX_SIZE, f"{r['size_bytes']} bytes")
    subs = None
    if srt:
        try:
            cues = read_srt(srt)
            ordered = all(a < b for a, b, _ in cues) and all(cues[k][0] >= cues[k - 1][0] for k in range(1, len(cues)))
            inside = not cues or cues[-1][1] <= (r["video_duration"] or 0) + 1
            subs = {"file": Path(srt).name, "cues": len(cues), "language": subtitle_language([c[2] for c in cues])}
            add("subtitles", ordered and inside, f"{len(cues)} cues, {'ordered' if ordered else 'NOT ordered'}, "
                f"{'within' if inside else 'BEYOND'} the video")
        except (OSError, ValueError) as e:
            add("subtitles", False, f"unreadable SRT: {e}")
    qc = {
        "tool_version": __version__, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {"file": src.name, **s}, "result": {"file": dst.name, "md5": md5(dst), **r},
        "subtitles": subs, "checks": checks, "pass": all(c["ok"] for c in checks),
    }
    if sheets:
        qc["sheets"], qc["sheet_times"] = contact_sheets(src, dst, out_dir, s["width"], s["height"],
                                                          s["video_duration"] or 0)
    (out_dir / "qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"check {dst.name}: {'PASS' if qc['pass'] else 'FAIL'} ({time.time() - t0:.0f}s) -> {out_dir / 'qc.json'}")
    for c in checks:
        log(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['name']}: {c['detail']}")
    return qc
