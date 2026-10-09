"""Subtitle recognition: OCR, subtitle-line detection, segmentation into sentences, SRT output."""
import difflib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .common import usable_cpus


class _FixedShape:
    """Wraps the recognition session and always feeds it one shape (batch x 3 x 48 x width).

    On some GPUs (seen on an RTX 5090 with onnxruntime-gpu 1.23) every change of input shape costs
    2-3 s, while repeated calls with the same shape take ~8 ms. Recognition batches vary in width, so
    padding them (zeros, exactly like RapidOCR pads within a batch) to a fixed shape removes that cost.
    Batches that do not fit are passed through unchanged."""

    def __init__(self, inner, batch=6, width=1280):
        self.inner, self.batch, self.width = inner, batch, width

    def __call__(self, x):
        n, c, h, w = x.shape
        if n > self.batch or w > self.width:
            return self.inner(x)
        pad = np.zeros((self.batch, c, h, self.width), x.dtype)
        pad[:n, :, :, :w] = x
        out = self.inner(pad)  # (batch, time steps, classes); time steps grow with the width
        return out[:n, :-(-out.shape[1] * w // self.width)]

    def __getattr__(self, name):
        return getattr(self.inner, name)


class OCR:
    def __init__(self, device, fixed_shape=False):
        from rapidocr import RapidOCR
        threads = min(4, usable_cpus())  # the default (all cores) is throttled hard under a container CPU quota
        params = {"Det.limit_type": "max", "Det.limit_side_len": 960,
                  "Global.log_level": "error",
                  "EngineConfig.onnxruntime.intra_op_num_threads": threads,
                  "EngineConfig.onnxruntime.inter_op_num_threads": 1}
        if device == "cuda":
            params["EngineConfig.onnxruntime.use_cuda"] = True
            params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] = "HEURISTIC"
        elif device == "dml":
            params["EngineConfig.onnxruntime.use_dml"] = True
        self.engine = RapidOCR(params=params)
        if fixed_shape:
            rec = self.engine.text_rec
            rec.session = _FixedShape(rec.session, batch=rec.rec_batch_num)

    def __call__(self, img):
        r = self.engine(img, use_cls=False)
        if r.boxes is None or r.txts is None:
            return []
        return [(np.asarray(b), t, float(s)) for b, t, s in zip(r.boxes, r.txts, r.scores)]


@dataclass
class Seg:
    """One subtitle: frame range, text votes and all its text boxes."""
    start: int
    end: int
    texts: Counter = field(default_factory=Counter)
    boxes: list = field(default_factory=list)  # (x0, y0, x1, y1) in full-frame coordinates
    ext_lo: int = None  # first/last frame where the same glyphs are still on screen although OCR lost them
    ext_hi: int = None

    @property
    def span(self):
        """Frames to erase: the OCR range widened by the glyph-matched extension"""
        return min(self.start, self.start if self.ext_lo is None else self.ext_lo), \
            max(self.end, self.end if self.ext_hi is None else self.ext_hi)

    @property
    def text(self):
        return self.texts.most_common(1)[0][0]

    @property
    def xr(self):
        return min(b[0] for b in self.boxes), max(b[2] for b in self.boxes)


def frame_boxes(items, band_y0, W, min_h, min_score):
    """First-pass filter of one frame's OCR results -> [(x0, y0, x1, y1, text)] in full-frame coordinates"""
    keep = []
    for box, txt, score in items:
        x0, y0 = box.min(0)
        x1, y1 = box.max(0)
        cx = (x0 + x1) / 2
        if score < min_score or (y1 - y0) < min_h or not txt.strip():
            continue
        if not (0.15 * W < cx < 0.85 * W):  # subtitles are normally centred
            continue
        keep.append((int(x0), int(y0) + band_y0, int(x1), int(y1) + band_y0, txt.strip()))
    return keep


def subtitle_line(per_frame):
    """Find where subtitles usually sit from all text boxes in the video: (bottom line y-centre, text height)"""
    ys = [((b[1] + b[3]) / 2, b[3] - b[1]) for boxes in per_frame for b in boxes]
    if not ys:
        return None
    yc = np.array([y for y, _ in ys])
    hs = np.array([h for _, h in ys])
    hist = np.bincount((yc // 4).astype(int))
    mode = (np.argmax(hist) + 0.5) * 4
    h = float(np.median(hs[np.abs(yc - mode) < 8]))
    return mode, h


def filter_line(boxes, line):
    """Keep only boxes matching the subtitle line position and height (one extra line above is allowed for two-line subtitles)"""
    if line is None:
        return boxes
    mode, h = line
    # Upper bound 1.6: a detector box that also swallows a light streak or edge next to the text (seen at 14:28 of
    # the test film: 74 px against the usual 52 px) is still the subtitle; the position check below keeps scene text out
    keep = [b for b in boxes
            if 0.75 * h <= b[3] - b[1] <= 1.6 * h
            and mode - 2.5 * h <= (b[1] + b[3]) / 2 <= mode + 0.5 * h]
    main = [b for b in keep if abs((b[1] + b[3]) / 2 - mode) <= 0.5 * h]
    if not main:
        return []
    # An upper line must lie entirely above the subtitle line (boxes overlapping it vertically are scene texture/objects)
    top = min(b[1] for b in main)
    return main + [b for b in keep if b not in main and b[3] <= top + 0.1 * h]


def join_text(keep):
    """Sort into lines and join -> (text, [boxes])"""
    if not keep:
        return None, []
    # Group boxes whose y-centres are close into the same line
    keep.sort(key=lambda b: (b[1] + b[3]) / 2)
    lines, cur = [], [keep[0]]
    for b in keep[1:]:
        ph = cur[-1][3] - cur[-1][1]
        if abs((b[1] + b[3]) / 2 - (cur[-1][1] + cur[-1][3]) / 2) < 0.5 * ph:
            cur.append(b)
        else:
            lines.append(cur)
            cur = [b]
    lines.append(cur)
    text = "\n".join(" ".join(b[4] for b in sorted(l, key=lambda b: b[0])) for l in lines)
    # a light streak or edge inside a detector box is read as a stray ASCII symbol at the end of the line
    text = "\n".join(line.strip(" \\/|_`~") for line in text.split("\n")).strip("\n")
    return text, [b[:4] for b in keep]


def similar(a, b):
    return difflib.SequenceMatcher(None, a.replace(" ", ""), b.replace(" ", "")).ratio()


def _x_overlap(a, b):
    (ax0, ax1), (bx0, bx1) = a.xr, b.xr
    return min(ax1, bx1) - max(ax0, bx0) > 0.4 * min(ax1 - ax0, bx1 - bx0)


def build_segments(per_frame, max_gap, min_len, keep_edges=False):
    """per_frame: [(text, boxes)] -> [Seg]. keep_edges: also keep short segments that touch the first or last frame
    (a subtitle cut in two where a long video was split for --jobs is not noise)"""
    # 1) Consecutive frames with identical text (ignoring spaces) -> runs
    runs = []
    for i, (text, boxes) in enumerate(per_frame):
        if not text:
            continue
        r = runs[-1] if runs else None
        if r and i - r.end <= max_gap + 1 and r.text.replace(" ", "") == text.replace(" ", ""):
            r.end = i
            r.texts[text] += 1
            r.boxes += boxes
        else:
            runs.append(Seg(i, i, Counter({text: 1}), list(boxes)))
    # 2) Merge OCR jitter: very similar text, or a short run fairly similar to the previous one.
    #    Two long runs with different text are kept as two separate subtitles.
    short = max(min_len, 4)
    segs = []
    for r in runs:
        p = segs[-1] if segs else None
        if p and r.start - p.end <= max_gap + 1 and _x_overlap(p, r):
            s = similar(p.text, r.text)
            rl, pl = r.end - r.start + 1, p.end - p.start + 1
            if s >= 0.9 or (s >= 0.6 and min(rl, pl) < short):
                p.end = r.end
                p.texts.update(r.texts)
                p.boxes += r.boxes
                continue
        segs.append(r)
    last = len(per_frame) - 1
    return [s for s in segs if s.end - s.start + 1 >= min_len
            or (keep_edges and (s.start <= max_gap or s.end >= last - max_gap))]


def segments_from_ocr(per_frame, fps, max_gap=5, min_dur=0.4, line="auto", keep_edges=False):
    """Raw per-frame OCR boxes -> (subtitle line, [Seg]). `line` = a known (y centre, height) to reuse
    instead of detecting it, e.g. when checking an already cleaned video where almost no text is left"""
    if line == "auto":
        line = subtitle_line(per_frame)
    texts = [join_text(filter_line(list(b), line)) for b in per_frame]
    return line, build_segments(texts, max_gap=max_gap, min_len=max(2, round(min_dur * fps)), keep_edges=keep_edges)


def ts(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segs, fps, path):
    with open(path, "w", encoding="utf-8") as f:
        for k, s in enumerate(segs, 1):
            f.write(f"{k}\n{ts(s.start / fps)} --> {ts((s.end + 1) / fps)}\n{s.text}\n\n")
