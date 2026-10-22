"""--qc-subtitles: look for subtitles that are still on screen in a finished video (delivery requirement: zero left).

The method follows the customer's self-check rules (version of 22 October 2026):
  1. calibration: three synthetic samples (a single character, light grey small text, a bilingual pair) are drawn on
     real frames and must be read by the very OCR settings of the scan; the detector size is raised until they are
  2. every subtitle of the SRT: five frames (start+0.1 s, middle, end-0.1 s, start-0.5 s, end+0.5 s; the middle only
     for a cue under 0.4 s), the bottom 55 % at full resolution, OCR at a low confidence, no filtering of single
     characters; the text is compared with this cue and its two neighbours
  3. blind scan: one frame per second, the bottom 45 %, text that is centred, 2-7 % of the picture high, one or two lines
  4. logo/watermark: ten frames, the four corners, content that stays put
What is added to the rules: the subtitle line, height and style are learned from the source (where the subtitles are
still on screen), and every suspect box is compared with the source at the same place. A box that looks like the
subtitles and is unchanged from the source is a leftover (verdict "残留"); text that is not like the subtitles and is
unchanged is scene text (listed as dismissed); anything else is left to a person ("待人工复核", on contact sheets).
Writes report.md, findings.csv, evidence/ and sheets into the output folder; the video files are only read.
"""
import csv
import difflib
import re
import shutil
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .common import __version__, cpu_share, log
from .masks import outline_score
from .qc import read_srt
from .subtitles import STYLE_MIN
from .video import _exe, packet_count, probe, read_frames

SCORE_MIN = 0.3     # OCR confidence kept; the erasing pass uses 0.6, a leftover is judged by eye anyway
CUE_BAND = 0.45     # cue frames: rows from 45 % of the height down (the bottom 55 %)
BLIND_BAND = 0.55   # blind scan: the bottom 45 %
UNCHANGED = 6.0     # mean absolute difference to the source below which the text is untouched (encoding noise 1-1.5 on a
#                     still picture, up to 4.6 behind moving content in 1304; erased text differs by 13 or more)
MIN_PSNR = 40.0     # outside the subtitle band the result must be this close to the source (customer's acceptance)
DET_SIDES = (1920, 2560, 3200)
FONTS = ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "C:/Windows/Fonts/msyh.ttc",
         "C:/Windows/Fonts/simhei.ttf", "/System/Library/Fonts/PingFang.ttc")

_KEEP = re.compile(r"[^0-9a-z\u4e00-\u9fff]")


def norm(text):
    """Only Han characters, letters and digits, letters lower case (both sides of every comparison)"""
    return _KEEP.sub("", str(text).lower())


def lcs(a, b):
    """Length of the longest common substring"""
    return difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size


def hms(t):
    t = int(round(t))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def suspect(line, cue_norms):
    """Is the OCR text `line` a leftover of one of the subtitles `cue_norms` (this cue and its neighbours)? -> reason or None.
    Two or more characters in common (an English word of the cue is covered by this), or a single character that is in
    one of them (a single Han character or letter: the commonest leftover of a short subtitle)."""
    n = norm(line)
    if not n:
        return None
    for cn in cue_norms:
        if lcs(n, cn) >= 2:
            return "最长公共子串>=2"
        if len(n) == 1 and n in cn:
            return "单字出现在相邻台词"
    return None


def cue_frame_times(cues, duration):
    """[(cue index, label, t)]: the five frames of every cue"""
    out = []
    for j, (a, b, _) in enumerate(cues):
        ts = [("mid", (a + b) / 2)]
        if b - a >= 0.4:
            ts += [("start+0.1", a + 0.1), ("end-0.1", b - 0.1)]
        ts += [("start-0.5", a - 0.5), ("end+0.5", b + 0.5)]
        out += [(j, label, t) for label, t in ts if 0 <= t < duration]
    return out


def read_lines(ocr, crop, y_off):
    """OCR of `crop` (BGR) -> [(x0, y0, x1, y1, text, score)] in full-frame coordinates, low-confidence lines kept"""
    out = []
    for box, text, score in ocr(np.ascontiguousarray(crop)):
        if score < SCORE_MIN or not norm(text):
            continue
        x0, y0 = box.min(0)
        x1, y1 = box.max(0)
        out.append((int(x0), int(y0) + y_off, int(x1), int(y1) + y_off, text.strip(), float(score)))
    return out


def find_font():
    for p in FONTS:
        if Path(p).exists():
            return p
    return None


def _grab(path, i, w, h):
    for f in read_frames(path, w, h, start=i, count=1):
        return f.copy()
    return None


def calibrate(ocr, video, W, H, n, out_dir):
    """Three synthetic samples drawn on real frames must be read by the scan's OCR settings. -> {passed, detail, det_side, ...}"""
    from PIL import Image, ImageDraw, ImageFont
    font_path = find_font()
    if not font_path:
        return {"passed": False, "detail": "没有找到中文字体，无法生成校准样例", "det_side": None, "samples": []}
    sets = [("单字", [("走", 0.055, (255, 255, 255), 2)]),
            ("浅灰小字", [("这是一行浅灰色的小字", 0.025, (200, 200, 200), 0)]),
            ("中英双语", [("你好吗朋友们", 0.05, (255, 255, 255), 2), ("How are you my friends", 0.04, (255, 255, 255), 2)])]
    frames = []
    for k, (name, lines) in enumerate(sets):
        base = _grab(video, int(n * (0.2 + 0.3 * k)), W, H)
        if base is None:
            return {"passed": False, "detail": "取不到校准用的帧", "det_side": None, "samples": []}
        img = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        y = int(H * (0.92 - 0.06 * len(lines)))
        for text, size, colour, stroke in lines:
            font = ImageFont.truetype(font_path, max(8, int(H * size)))
            w = d.textlength(text, font=font)
            d.text(((W - w) / 2, y), text, font=font, fill=colour, stroke_width=stroke, stroke_fill=(0, 0, 0))
            y += int(H * size * 1.3)
        frames.append((name, [norm(t) for t, *_ in lines], cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)))
    cal = Path(out_dir) / "calibration"
    cal.mkdir(parents=True, exist_ok=True)
    result = {"passed": False, "det_side": None, "samples": []}
    for side in DET_SIDES:
        if hasattr(ocr, "set_det_side"):
            ocr.set_det_side(side)
        rows, ok_all = [], True
        for name, expect, frame in frames:
            y0 = int(H * CUE_BAND)
            got = norm("".join(t for *_, t, _s in read_lines(ocr, frame[y0:], y0)))
            ok = all(lcs(e, got) >= max(1, int(len(e) * 0.8)) for e in expect)
            ok_all &= ok
            rows.append({"name": name, "expected": expect, "read": got, "ok": ok})
            if side == DET_SIDES[0] or ok_all:
                cv2.imwrite(str(cal / f"{name}.jpg"), frame[y0:], [cv2.IMWRITE_JPEG_QUALITY, 90])
        result = {"passed": ok_all, "det_side": side, "samples": rows}
        if ok_all:
            break
    result["detail"] = ("三个样例全部识别" if result["passed"] else "有样例没有识别出来") + f"，检测边长 {result['det_side']}"
    return result


_WORKER_OCR = None


def _init_worker(spec, side):
    global _WORKER_OCR
    from .subtitles import OCR
    cv2.setNumThreads(1)   # the workers are the parallelism; OpenCV's own thread pools only compete for the CPU quota
    _WORKER_OCR = OCR(*spec)
    _WORKER_OCR.set_det_side(side)


def _in_worker(fn, *args):
    return fn(_WORKER_OCR, *args)


class Runner:
    """Runs the OCR jobs of the check here (workers <= 1, or an OCR that cannot be rebuilt) or in `workers` processes
    that each hold an OCR of their own: one OCR call leaves the GPU and most cores idle."""

    def __init__(self, ocr, workers, side):
        self.ocr, self.pool = ocr, None
        spec = getattr(ocr, "spec", None)
        self.workers = workers if spec else 1
        if self.workers > 1:
            import multiprocessing
            import os
            from concurrent.futures import ProcessPoolExecutor
            # the container's CPU quota is shared by the workers: each sizes its ffmpeg threads for its own share
            os.environ["CLEAN_CLEAR_CPUS"] = str(max(2, cpu_share() // self.workers))
            self.pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"),
                                            initializer=_init_worker, initargs=(spec, side))

    def submit(self, fn, *args):
        if self.pool:
            return self.pool.submit(_in_worker, fn, *args)
        done = Future()
        done.set_result(fn(self.ocr, *args))
        return done

    def close(self):
        if self.pool:
            self.pool.shutdown()

    def chunks(self, items):
        """`items` (sorted) in groups of consecutive ones: three per worker, so a slow group does not hold the rest"""
        k = max(1, min(3 * self.workers, len(items)))
        size = -(-len(items) // k)
        return [items[j:j + size] for j in range(0, len(items), size)]


def start_learn(runner, source, cues, W, H, fps, limit=150):
    """Where the subtitles are and how they look, from the source: OCR at the middle of up to `limit` cues, the boxes that
    read like the cue. Returns the pending jobs for finish_learn."""
    if not cues:
        return []
    cues = cues[:max(limit, len(cues) // 3)]   # the first third of the film is enough, and ends the read of the source early
    step = max(1, len(cues) // limit)
    want = {}
    for j in range(0, len(cues), step):
        a, b, text = cues[j]
        if len(norm(text)) >= 2:
            want[min(int(((a + b) / 2) * fps), 10 ** 9)] = norm(text)
    if not want:
        return []
    return [runner.submit(_learn_chunk, source, W, H, {i: want[i] for i in group})
            for group in runner.chunks(sorted(want))]


def _learn_chunk(ocr, source, W, H, want):
    """(line centres, heights, outline scores) of the boxes that read like the cue, for the frames in `want` {frame: text}"""
    y0 = int(H * CUE_BAND)
    first, last = min(want), max(want)
    ys, hs, styles = [], [], []
    for i, strip in enumerate(read_frames(source, W, H, start=first, count=last - first + 1, crop=(y0, H)), first):
        if i in want:
            for x0, by0, x1, by1, text, score in read_lines(ocr, strip, y0):
                n = norm(text)
                if score >= 0.6 and lcs(n, want[i]) >= max(2, int(0.6 * min(len(n), len(want[i])))):
                    ys.append((by0 + by1) / 2)
                    hs.append(by1 - by0)
                    styles.append(outline_score(strip[by0 - y0:by1 - y0, x0:x1]))
    return ys, hs, styles


def finish_learn(jobs):
    """-> {line_y, line_h, style, outlined, samples}"""
    ys, hs, styles = [], [], []
    for job in jobs:
        a, b, c = job.result()
        ys += a
        hs += b
        styles += c
    if not ys:
        return {"samples": 0}
    style = float(np.median(styles))
    return {"line_y": float(np.median(ys)), "line_h": float(np.median(hs)), "style": style,
            "outlined": style >= STYLE_MIN, "samples": len(ys)}


def looks_like_subtitle(box, W, look, style):
    """The customer's "字幕外观" rules 1-3 and 5: centred (35-65 % of the width), on the subtitle line (within one text
    height), 0.6-1.5 times the subtitle height, light with a dark outline when the video's subtitles are like that"""
    x0, y0, x1, y1 = box
    if not 0.35 <= (x0 + x1) / 2 / W <= 0.65:
        return False
    if "line_y" in look:
        if abs((y0 + y1) / 2 - look["line_y"]) > look["line_h"] or not 0.6 <= (y1 - y0) / look["line_h"] <= 1.5:
            return False
    return style >= STYLE_MIN if look.get("outlined") else True


def start_scan(runner, result, cues, W, H, fps, n, duration, out_dir):
    """The frames of the checks (cue frames and the blind scan), OCR'd over the result in groups. -> (pending jobs, stats)"""
    ev = Path(out_dir) / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    needed = {}
    for j, label, t in cue_frame_times(cues, duration):
        i = min(n - 1, max(0, int(round(t * fps))))
        needed.setdefault(i, {"cue": [], "blind": False})["cue"].append((j, label, t))
    blind_frames = 0
    k = 0
    while k * 1.0 < duration:
        i = min(n - 1, int(round(k * fps)))
        needed.setdefault(i, {"cue": [], "blind": False})["blind"] = True
        blind_frames += 1
        k += 1
    norms = [norm(c[2]) for c in cues]
    dialogues = [c[2].replace("\n", " / ") for c in cues]
    jobs = [runner.submit(_scan_chunk, result, W, H, fps, {i: needed[i] for i in group}, str(ev), norms, dialogues)
            for group in runner.chunks(sorted(needed))]
    return jobs, {"cue_frames": sum(len(v["cue"]) for v in needed.values()), "blind_frames": blind_frames}


def finish_scan(jobs, stats):
    """-> (findings, stats)"""
    findings, ocr_frames = [], 0
    for job in jobs:
        f, k = job.result()
        findings += f
        ocr_frames += k
    return findings, {**stats, "ocr_frames": ocr_frames}


def _scan_chunk(ocr, result, W, H, fps, needed, ev, norms, dialogues):
    """OCR of the frames in `needed` {frame: {cue, blind}}; the suspect boxes -> (findings, frames read)"""
    ev = Path(ev)
    y0 = int(H * CUE_BAND)
    findings, ocr_frames = [], 0
    first, last = min(needed), max(needed)
    for i, strip in enumerate(read_frames(result, W, H, start=first, count=last - first + 1, crop=(y0, H)), first):
        info = needed.get(i)
        if info:
            ocr_frames += 1
            off = 0 if info["cue"] else int(H * BLIND_BAND) - y0  # a blind-only frame: just the bottom 45 %
            lines = read_lines(ocr, strip[off:], y0 + off)
            t_frame = i / fps
            saved = False
            for j, label, t in info["cue"]:
                around = [norms[m] for m in (j - 1, j, j + 1) if 0 <= m < len(norms)]
                for x0, by0, x1, by1, text, score in lines:
                    why = suspect(text, around)
                    if why:
                        findings.append({"method": "cue", "frame": i, "t": t_frame, "label": label, "text": text,
                                         "dialogue": dialogues[j], "cue": j, "reason": why,
                                         "box": (x0, by0, x1, by1), "score": score,
                                         "style": outline_score(strip[by0 - y0:by1 - y0, x0:x1])})
                        saved = True
            if info["blind"]:
                cand = [b for b in lines if 0.3 <= (b[0] + b[2]) / 2 / W <= 0.7 and 0.02 <= (b[3] - b[1]) / H <= 0.07]
                rows = []
                for b in sorted(cand, key=lambda b: b[1]):
                    if rows and abs((b[1] + b[3]) / 2 - rows[-1][0]) <= 0.6 * (b[3] - b[1]):
                        rows[-1][1].append(b)
                    else:
                        rows.append([(b[1] + b[3]) / 2, [b]])
                if 1 <= len(rows) <= 2:
                    for _, bs in rows:
                        for x0, by0, x1, by1, text, score in bs:
                            findings.append({"method": "blind", "frame": i, "t": t_frame, "label": "每秒盲扫", "text": text,
                                             "dialogue": "", "cue": None, "reason": "盲扫: 居中、字高2%-7%",
                                             "box": (x0, by0, x1, by1), "score": score,
                                             "style": outline_score(strip[by0 - y0:by1 - y0, x0:x1])})
                            saved = True
            if saved:
                cv2.imwrite(str(ev / f"_f{i}.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if i >= last:
            break
    return findings, ocr_frames


def compare_with_source(findings, source, W, H, result_dir, workers=1):
    """Mean absolute difference between the source and the result inside every suspect box (grey levels, 0-255)"""
    if not findings:
        return
    y0 = int(H * CUE_BAND)
    frames = sorted({f["frame"] for f in findings})
    got = {}

    def grab(group):   # one decode over the frames of the group
        want = set(group)
        for i, s in enumerate(read_frames(source, W, H, start=group[0], count=group[-1] - group[0] + 1, crop=(y0, H)),
                              group[0]):
            if i in want:
                got[i] = s.copy()

    if len(frames) <= 150:   # a seek per frame (~0.15 s) is cheaper than decoding the whole film again (~4 s per minute of film)
        groups = [[i] for i in frames]
    else:
        size = -(-len(frames) // (3 * workers))
        groups = [frames[j:j + size] for j in range(0, len(frames), size)]
    with ThreadPoolExecutor(max(1, workers)) as pool:   # ffmpeg does the work, the threads only wait for it
        list(pool.map(grab, groups))
    for f in findings:
        res = cv2.imread(str(Path(result_dir) / "evidence" / f"_f{f['frame']}.jpg"))
        src = got.get(f["frame"])
        x0, by0, x1, by1 = f["box"]
        if res is None or src is None:
            f["diff"] = None
            continue
        a = src[by0 - y0:by1 - y0, x0:x1].astype(np.int16)
        b = res[by0 - y0:by1 - y0, x0:x1].astype(np.int16)
        f["diff"] = float(np.abs(a - b).mean()) if a.size else None


def judge(findings, W, look, keep=()):
    """verdict per finding. A text that matches the cue (or its neighbours) and looks like the subtitles is a leftover
    however much of it was changed (a half-erased sentence is read as well as an untouched one); a text that is not
    tied to a cue (blind scan) is a leftover only for a person to say, unless it is in a kept range (credits) or
    does not look like the subtitles and is unchanged (scene text)."""
    for f in findings:
        like = looks_like_subtitle(f["box"], W, look, f["style"])
        untouched = f.get("diff") is not None and f["diff"] < UNCHANGED
        f["like_subtitle"] = like
        if any(a <= f["t"] <= b for a, b in keep):
            f["verdict"] = "片头片尾(保留)"
        elif f["method"] == "cue" and like:
            f["verdict"] = "残留"
        elif not like and untouched:
            f["verdict"] = "场景文字(自动排除)"
        else:
            f["verdict"] = "待人工复核"


def merge_runs(findings, fps):
    """A text that stays on screen shows in several frames: one row per run of the same text in the same place"""
    findings.sort(key=lambda f: (f["method"], f["box"][1] // 20, norm(f["text"]), f["frame"]))
    out = []
    for f in findings:
        p = out[-1] if out else None
        if (p and p["method"] == f["method"] == "blind" and norm(p["text"]) == norm(f["text"])
                and abs(p["box"][1] - f["box"][1]) < 20 and f["frame"] - p["last_frame"] <= 2 * fps and p["verdict"] == f["verdict"]):
            p["last_frame"] = f["frame"]
            p["count"] += 1
            continue
        out.append({**f, "last_frame": f["frame"], "count": 1})
    out.sort(key=lambda f: f["frame"])
    return out


def logos(video, W, H, n, out_dir):
    """Content that stays put in a corner over ten evenly spread frames: [(corner, evidence file)]"""
    cw, ch = int(W * 0.2), int(H * 0.2)
    corners = {"左上": (0, 0), "右上": (W - cw, 0), "左下": (0, H - ch), "右下": (W - cw, H - ch)}
    stack = {c: [] for c in corners}
    for k in range(10):
        f = _grab(video, int(n * (0.05 + 0.09 * k)), W, H)
        if f is None:
            continue
        for c, (x, y) in corners.items():
            stack[c].append(f[y:y + ch, x:x + cw].astype(np.int16))
    found = []
    for c, crops in stack.items():
        if len(crops) < 6:
            continue
        a = np.stack(crops)
        med = np.median(a, 0)
        steady = float(np.mean([np.abs(x - med).mean() for x in crops]))
        edges = cv2.Laplacian(cv2.cvtColor(med.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_16S)
        busy = float((np.abs(edges) > 40).mean())
        if steady < 6 and busy > 0.004:
            name = f"logo_{c}.jpg"
            cv2.imwrite(str(Path(out_dir) / name), med.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
            found.append({"corner": c, "file": name, "steadiness": round(steady, 1), "edge_share": round(busy, 4)})
    return found


def start_psnr(source, result, out_dir, tag, top=None, t0=None, dur=None):
    """ffmpeg's psnr filter over the two videos in the background (frame by frame, both decoded once, no Python in the
    loop). top: only the top part of the picture (0.6 = outside the subtitle band); t0/dur: a range of the film.
    -> (process, stats file)"""
    stats = Path(out_dir) / f"_psnr_{tag}.txt"
    cmd = [_exe("ffmpeg"), "-v", "error", "-y"]
    for p in (source, result):
        cmd += (["-ss", f"{t0:.3f}", "-t", f"{dur:.3f}"] if t0 is not None else []) + ["-i", str(p)]
    crop = f"crop=iw:trunc(ih*{top}/2)*2:0:0," if top else ""
    graph = f"[0:v:0]{crop}format=yuv420p[a];[1:v:0]{crop}format=yuv420p[b];[a][b]psnr=stats_file={stats.as_posix()}"
    return subprocess.Popen(cmd + ["-filter_complex", graph, "-f", "null", "-"], stderr=subprocess.PIPE), stats


def finish_psnr(job, fps):
    """-> {n, mean, min, share_below, worst: [(second, dB)]} of a start_psnr job (dB capped at 99 for identical frames)"""
    proc, stats = job
    err = proc.communicate()[1].decode("utf-8", "replace").strip()
    vals = []
    if stats.exists():
        for line in stats.read_text().splitlines():
            m = re.search(r"psnr_avg:(\S+)", line)
            if m:
                vals.append(99.0 if m.group(1) == "inf" else min(99.0, float(m.group(1))))
        stats.unlink()
    if not vals:
        return {"n": 0, "error": err[-200:] or "no PSNR output"}
    v = np.array(vals)
    worst = [(round(int(i) / fps, 2), round(float(v[i]), 1)) for i in np.argsort(v)[:5]]
    return {"n": len(v), "mean": round(float(v.mean()), 2), "min": round(float(v.min()), 2),
            "share_below": round(float((v < MIN_PSNR).mean()), 4), "worst": worst}


def sheets(rows, source, result, W, H, out_dir, asset):
    """Contact sheets (source above, result below) of the findings that need a person; -> file names"""
    from PIL import Image, ImageDraw, ImageFont
    items = [r for r in rows if r["verdict"] != "场景文字(自动排除)"]
    if not items:
        return []
    font_path = find_font()
    font = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()
    names, y0 = [], int(H * CUE_BAND)
    tiles = []
    for r in items:
        x0, by0, x1, by1 = r["box"]
        h = max(60, by1 - by0)
        cx = (x0 + x1) // 2
        xa, xb = max(0, cx - 360), min(W, cx + 360)
        ya, yb = max(0, (by0 + by1) // 2 - 2 * h), min(H, (by0 + by1) // 2 + 2 * h)
        res = cv2.imread(str(Path(out_dir) / "evidence" / f"_f{r['frame']}.jpg"))
        src = next(iter(read_frames(source, W, H, start=r["frame"], count=1, crop=(y0, H))), None)
        if res is None or src is None:
            continue
        a, b = src[ya - y0:yb - y0, xa:xb], res[ya - y0:yb - y0, xa:xb]
        tile = np.full((2 * (yb - ya) + 40, 720, 3), 255, np.uint8)
        tile[40:40 + b.shape[0], :b.shape[1]] = a
        tile[40 + b.shape[0]:40 + 2 * b.shape[0], :b.shape[1]] = b
        img = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(img).text((4, 8), f"{asset} {hms(r['t'])} 「{r['text']}」 {r['verdict']}  (上:原片 下:成片)", font=font,
                                 fill=(0, 0, 0))
        tiles.append(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    for s in range(0, len(tiles), 8):
        grid = tiles[s:s + 8]
        hh = max(t.shape[0] for t in grid)
        grid = [np.pad(t, ((0, hh - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in grid]
        cols = [np.vstack(grid[k::2]) for k in range(2)] if len(grid) > 1 else [grid[0]]
        hmax = max(c.shape[0] for c in cols)
        cols = [np.pad(c, ((0, hmax - c.shape[0]), (0, 0), (0, 0)), constant_values=255) for c in cols]
        name = f"sheet_{s // 8 + 1:02d}.jpg"
        cv2.imwrite(str(Path(out_dir) / name), np.hstack(cols), [cv2.IMWRITE_JPEG_QUALITY, 88])
        names.append(name)
    return names


def auto_workers():
    """OCR processes for the check: a quarter of the usable cores, 2-6 (the PSNR runs next to them; on 25 cores the scan
    of 1304 took 114 s with 4 processes and 68 s with 6, and no less with more)"""
    return max(2, min(6, cpu_share() // 4))


def check_subtitles(source, result, out_dir, ocr, srt=None, keep_ranges="", workers=0):
    """Run all checks; write report.md, findings.csv, evidence/ and sheets into out_dir. -> summary dict.
    keep_ranges: the --keep-ranges of the film; findings there are credits ("片头片尾(保留)"), and their pictures are
    compared with the source frame by frame. workers: OCR processes (0 = automatic, 1 = in this process)."""
    t0 = time.time()
    laps = {}

    def lap(name):
        laps[name] = round(time.time() - t0 - sum(laps.values()), 1)

    source, result = Path(source), Path(result)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if srt is None:
        cand = result.with_name(result.stem.removesuffix("_clean") + ".srt")
        srt = cand if cand.exists() else None
    if srt is None:
        raise SystemExit("--qc-subtitles needs the subtitle file next to the result (NAME.srt) or --srt FILE")
    asset = result.stem.removesuffix("_clean")
    W, H, fps, n = probe(result)
    n = packet_count(result) or n
    duration = n / fps
    cues = read_srt(srt)
    from .pipeline import parse_keep
    keep = parse_keep(keep_ranges, duration)
    log(f"qc: {asset}  {W}x{H} {fps:.3f}fps  {n} frames, {len(cues)} subtitles")
    # the picture outside the subtitle band and the kept ranges, source against result: ffmpeg decodes both while the OCR runs
    psnr_jobs = {"outside": start_psnr(source, result, out_dir, "outside", top=0.6)}
    for k, (a, b) in enumerate(keep):
        psnr_jobs[f"keep{k}"] = start_psnr(source, result, out_dir, f"keep{k}", t0=a, dur=b - a)

    cal = calibrate(ocr, result, W, H, n, out_dir)
    log(f"  calibration: {cal['detail']}")
    lap("校准")
    runner = Runner(ocr, workers or auto_workers(), cal["det_side"] or DET_SIDES[0])
    log(f"  OCR in {runner.workers} process(es)")
    try:   # the look of the source and the scan of the result run side by side in the same processes
        learning = start_learn(runner, source, cues, W, H, fps)
        scanning, stats = start_scan(runner, result, cues, W, H, fps, n, duration, out_dir)
        look = finish_learn(learning)
        log(f"  subtitle look from the source: {look}")
        findings, stats = finish_scan(scanning, stats)
    finally:
        runner.close()
    log(f"  scanned {stats['ocr_frames']} frames ({stats['cue_frames']} cue frames, {stats['blind_frames']} blind): "
        f"{len(findings)} suspect boxes")
    lap("字幕外观、逐句截帧和盲扫")
    compare_with_source(findings, source, W, H, out_dir, runner.workers)
    judge(findings, W, look, keep)
    rows = merge_runs(findings, fps)
    lap("与原片比对")
    logo = logos(result, W, H, n, out_dir)
    lap("台标")

    ev = out_dir / "evidence"
    for r in rows:
        src = ev / f"_f{r['frame']}.jpg"
        r["frame_path"] = ""
        if r["verdict"] in ("残留", "待人工复核") and src.exists():
            name = f"{asset}_{hms(r['t']).replace(':', '-')}_{norm(r['text'])[:12] or 'x'}.jpg"
            shutil.copy(src, ev / name)
            r["frame_path"] = f"evidence/{name}"
    sheet_names = sheets(rows, source, result, W, H, out_dir, asset)
    for p in ev.glob("_f*.jpg"):
        p.unlink()
    lap("证据和复核图")
    psnr = {k: finish_psnr(j, fps) for k, j in psnr_jobs.items()}
    lap("等待PSNR")

    leftover = [r for r in rows if r["verdict"] == "残留"]
    review = [r for r in rows if r["verdict"] == "待人工复核"]
    conclusion = "需返工" if leftover else ("待人工复核" if review or not cal["passed"] else "合格")
    with open(out_dir / "findings.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["asset_id", "time_sec", "hms", "ocr_text", "matched_dialogue", "method", "verdict", "frame_path",
                    "frame_label", "ocr_score", "subtitle_style", "diff_to_source", "seconds_seen"])
        for r in rows:
            w.writerow([asset, f"{r['t']:.2f}", hms(r["t"]), r["text"], r["dialogue"], r["method"], r["verdict"],
                        r["frame_path"], r["label"], f"{r['score']:.2f}", f"{r['style']:.2f}",
                        "" if r.get("diff") is None else f"{r['diff']:.1f}", r["count"]])
    summary = {"asset_id": asset, "cues": len(cues), "cues_checked": len({c[0] for c in cue_frame_times(cues, duration)}),
               "frames_ocr": stats["ocr_frames"], "blind_frames": stats["blind_frames"], "leftover": len(leftover),
               "review": len(review), "dismissed": len(rows) - len(leftover) - len(review), "logo": bool(logo),
               "calibration": cal["passed"], "conclusion": conclusion, "seconds": round(time.time() - t0),
               "psnr_outside": psnr["outside"],
               "psnr_ok": bool(psnr["outside"].get("n")) and psnr["outside"]["min"] >= MIN_PSNR}
    write_report(out_dir, summary, cal, look, rows, logo, sheet_names, stats, str(srt), psnr, keep, laps)
    log(f"  {conclusion}: {len(leftover)} 残留, {len(review)} 待人工复核, {summary['dismissed']} 自动排除  "
        f"({summary['seconds']} s; {laps}) -> {out_dir}")
    log(f"  PSNR outside the subtitle band: {psnr['outside']}")
    return summary


def write_report(out_dir, s, cal, look, rows, logo, sheet_names, stats, srt, psnr=None, keep=(), laps=None):
    L = [f"# 字幕清除自检报告 {s['asset_id']}", "",
         f"- 工具版本：Clean Clear {__version__}；检查时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；耗时 {s['seconds']} 秒",
         f"- 字幕文件：{Path(srt).name}（{s['cues']} 条）", "",
         "## 校准", "",
         f"- 结果：{'通过' if cal['passed'] else '未通过'}（{cal['detail']}）",
         f"- 最终参数：检测边长 {cal['det_side']}，OCR 置信度下限 {SCORE_MIN}，不过滤单字，截取画面下方 55%（逐句）/45%（盲扫），原分辨率",
         ""]
    for r in cal.get("samples", []):
        L.append(f"  - {r['name']}：应读出 {r['expected']}，实际 {r['read']!r} → {'OK' if r['ok'] else '失败'}")
    L += ["", "## 字幕外观（从原片学习）", ""]
    if look.get("samples"):
        L.append(f"- 字幕行 y≈{look['line_y']:.0f}，字高≈{look['line_h']:.0f}px，描边得分 {look['style']:.2f}"
                 f"（{'白字深描边' if look['outlined'] else '无明显深描边'}），样本 {look['samples']} 条")
    else:
        L.append("- 没有从原片学到字幕外观：判定只用位置规则，结果多数交人工复核")
    L += ["", "## 汇总", "",
          "| asset_id | SRT字幕条数 | 已检查字幕条数 | 截帧总数(OCR) | 盲扫帧数 | 残留数 | 待人工复核 | 台标/水印 | 结论 |",
          "|---|---|---|---|---|---|---|---|---|",
          f"| {s['asset_id']} | {s['cues']} | {s['cues_checked']} | {stats['ocr_frames']} | {stats['blind_frames']} | "
          f"{s['leftover']} | {s['review']} | {'有' if s['logo'] else '无'} | {s['conclusion']} |", ""]
    if s["cues_checked"] != s["cues"]:
        L.append(f"> 注意：已检查条数 {s['cues_checked']} 少于字幕条数 {s['cues']}（有字幕的截帧时间落在视频之外），需人工确认。\n")
    for title, verdict in (("残留明细", "残留"), ("待人工复核", "待人工复核")):
        sel = [r for r in rows if r["verdict"] == verdict]
        L += [f"## {title}（{len(sel)}）", ""]
        if sel:
            L += ["| 时间 | 画面上的文字 | 对应台词 | 方式 | 与原片差异 | 截图 |", "|---|---|---|---|---|---|"]
            for r in sel:
                d = "—" if r.get("diff") is None else f"{r['diff']:.1f}"
                L.append(f"| {hms(r['t'])} | {r['text']} | {r['dialogue'] or '—'} | {r['method']}/{r['label']} | {d} | "
                         f"{r['frame_path'] or '—'} |")
        L.append("")
    psnr = psnr or {}
    L += ["## 擦除范围以外的画面（与原片 PSNR）", ""]
    o = psnr.get("outside", {})
    if o.get("n"):
        L += [f"- 画面上部 60%（字幕带以外），共 {o['n']} 帧：平均 {o['mean']} dB，最低 {o['min']} dB，"
              f"低于 {MIN_PSNR:.0f} dB 的帧占 {100 * o['share_below']:.2f}%（验收要求不低于 {MIN_PSNR:.0f} dB）："
              f"{'达标' if o['min'] >= MIN_PSNR else '有帧未达标'}",
              f"- 最低的几帧（秒, dB）：{o['worst']}"]
    else:
        L.append(f"- 没有算出 PSNR：{o.get('error', '')}")
    for k, (a, b) in enumerate(keep):
        v = psnr.get(f"keep{k}", {})
        L.append(f"- 片头片尾保留段 {hms(a)}-{hms(b)}（整帧）：" + (
            f"平均 {v['mean']} dB，最低 {v['min']} dB" if v.get("n") else f"没有算出 PSNR {v.get('error', '')}"))
    L.append("")
    dismissed = [r for r in rows if r["verdict"].startswith("场景文字")]
    L += [f"## 自动排除的场景文字（{len(dismissed)}）", "",
          "与原片逐像素一致、且外观不像本片字幕的文字（标牌、衣服字、道具等）；清单见 findings.csv。", ""]
    L += ["## 台标/水印（不判定）", ""]
    L += [f"- {x['corner']}角：位置不变的图标或文字（{x['file']}）" for x in logo] or ["- 四个角均未发现位置固定的内容"]
    L += ["", "## 人工复核用图", ""] + [f"- {n}" for n in sheet_names] if sheet_names else []
    if laps:
        L += ["", "## 各步骤耗时（秒）", "", " / ".join(f"{k} {v}" for k, v in laps.items())]
    L += ["", "## 说明", "",
          "- 只读检查，没有修改视频和字幕文件。数字均为本次运行的实际结果。",
          "- 判定规则：与台词吻合、外观像字幕、且与原片几乎一致 → 残留；外观不像字幕且与原片一致 → 场景文字；其余交人工看图。",
          "- 演职员表、片名、说明字幕、角落台标按验收标准不算问题，需人工在复核图上确认。", ""]
    (Path(out_dir) / "report.md").write_text("\n".join(L), encoding="utf-8")
