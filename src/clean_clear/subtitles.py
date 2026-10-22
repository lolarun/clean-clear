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


# Longest side the text detector sees. At 960 a 1080p subtitle band (1920 wide) was halved, and thin, widely spaced
# strokes were found or not with a grey level of difference: 1304 at 15:44 "二愣子" was boxed around "愣子" in 19 of 20
# frames, and the "二" stayed on screen. At 1920 it is boxed in all 20 (detector + recogniser ~38 ms per call
# instead of ~17 ms on an RTX 5090). Part of the OCR cache key (pipeline.ocr_cache_path).
DET_SIDE = 1920
# The verify pass only looks for subtitle text that is still readable; at 1920 it took 67 % longer (1304: 871 s
# against 520 s over the 8 parts) and also picked up a sign and the title as "left over" subtitles.
VERIFY_DET_SIDE = 960


class OCR:
    def __init__(self, device, fixed_shape=False):
        from rapidocr import RapidOCR
        threads = min(4, usable_cpus())  # the default (all cores) is throttled hard under a container CPU quota
        params = {"Det.limit_type": "max", "Det.limit_side_len": DET_SIDE,
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
            if rec is None and hasattr(self.engine, "_load_rec_model"):  # rapidocr >= 3.10 loads models lazily
                rec = self.engine._load_rec_model()
            rec.session = _FixedShape(rec.session, batch=rec.rec_batch_num)

    def set_det_side(self, side):
        """Longest side the text detector sees from now on (its preprocessing is rebuilt on every call)"""
        det = self.engine.text_det
        if det is None and hasattr(self.engine, "_load_det_model"):  # rapidocr >= 3.10 loads models lazily
            det = self.engine._load_det_model()
        det.limit_side_len = side

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
    gaps: list = field(default_factory=list)  # (first, last) frames where OCR lost the text and the two sides were joined
    skip: set = field(default_factory=set)  # frames of those gaps that show no subtitle (masks.glyph_masks): not erased

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
    """Find where subtitles usually sit from all text boxes in the video: (bottom line y-centre, text height), or
    (y, height, y2, height2) when a second line is regularly present (bilingual subtitles: the most frequent line
    is the main one, the second line may be above or below it)"""
    ys = [((b[1] + b[3]) / 2, b[3] - b[1]) for boxes in per_frame for b in boxes]
    if not ys:
        return None
    yc = np.array([y for y, _ in ys])
    hs = np.array([h for _, h in ys])
    hist = np.bincount((yc // 4).astype(int))
    mode = (np.argmax(hist) + 0.5) * 4
    near = np.abs(yc - mode) < 8
    h = float(np.median(hs[near]))
    # second line: another peak within two text heights that is at least 30 % as frequent as the main line
    other = (np.abs(yc - mode) > 0.6 * h) & (np.abs(yc - mode) < 2.5 * h) & (hs > 0.4 * h) & (hs < 1.6 * h)
    if other.sum() >= 0.3 * near.sum():
        hist2 = np.bincount((yc[other] // 4).astype(int))
        mode2 = (np.argmax(hist2) + 0.5) * 4
        near2 = other & (np.abs(yc - mode2) < 8)
        if near2.sum() >= 0.3 * near.sum():
            return mode, h, mode2, float(np.median(hs[near2]))
    return mode, h


def filter_line(boxes, line):
    """Keep only boxes matching the subtitle line position and height (the second line of a bilingual subtitle, or
    one extra line above the main line for two-line subtitles, is allowed)"""
    if line is None:
        return boxes
    mode, h = line[:2]
    lines = [line[i:i + 2] for i in range(0, len(line), 2)]
    main = [b for b in boxes
            if any(0.75 * lh <= b[3] - b[1] <= 1.6 * lh and abs((b[1] + b[3]) / 2 - ly) <= 0.5 * lh
                   for ly, lh in lines)]
    if not main:
        return []
    # Upper bound 1.6: a detector box that also swallows a light streak or edge next to the text (seen at 14:28 of
    # the test film: 74 px against the usual 52 px) is still the subtitle; the position check keeps scene text out
    keep = [b for b in boxes
            if 0.75 * h <= b[3] - b[1] <= 1.6 * h
            and mode - 2.5 * h <= (b[1] + b[3]) / 2 <= mode + 0.5 * h]
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


def _contains(a, b):
    """True if one text is the other with characters lost: OCR read "刚开始啊两三毛" for the last 0.4 s of a subtitle
    that reads "刚开始啊两三毛一斤", and the mask of that short segment did not cover the two characters left"""
    a, b = a.replace(" ", ""), b.replace(" ", "")
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 2 and short in long


def _same_place(a, b):
    """For a pair that `_contains` accepts: do the shorter text's characters sit where they are in the longer text?
    A subtitle that lost characters keeps the rest in place (1304 at 09:12: offset 0.08 characters), but a short
    subtitle of its own is centred (1304 at 11:22 "远程教育" after "要架那个远程教育网": 1.5 characters off, at 06:27
    and 07:41: 3 and 4). Merged, its glyphs were outvoted in the segment's mask and partly left on screen.
    The comparison uses the line of the longer text at the height of the shorter one: a line of scene text above
    the subtitle ("MONTBLANC" on a T-shirt) must not move the expected position."""
    short, long = (a, b) if len(a.text.replace(" ", "")) <= len(b.text.replace(" ", "")) else (b, a)
    st = short.text.replace(" ", "")
    if "\n" in st or not short.boxes:
        return True  # only a one-line text is checked
    yc = float(np.median([(y0 + y1) / 2 for _, y0, _, y1 in short.boxes]))
    h = float(np.median([y1 - y0 for _, y0, _, y1 in short.boxes]))
    row = [bx for bx in long.boxes if abs((bx[1] + bx[3]) / 2 - yc) <= 0.5 * h]
    line = next((t.replace(" ", "") for t in long.text.split("\n") if st in t.replace(" ", "")), None)
    if not row or not line:
        return True
    lx0, lx1 = min(bx[0] for bx in row), max(bx[2] for bx in row)
    sx0, sx1 = short.xr
    cw = (lx1 - lx0) / len(line)
    i = line.find(st)
    off0 = (sx0 - (lx0 + i * cw)) / cw
    off1 = (sx1 - (lx0 + (i + len(st)) * cw)) / cw
    return min(abs(off0), abs(off1)) <= 0.5


def _x_overlap(a, b):
    (ax0, ax1), (bx0, bx1) = a.xr, b.xr
    return min(ax1, bx1) - max(ax0, bx0) > 0.4 * min(ax1 - ax0, bx1 - bx0)


def _same_line_place(boxes_a, boxes_b):
    """Do two sets of boxes (one frame's, or a run's) sit in the same place: the same horizontal extent (80 % of the
    narrower) and the same height of the line (centres within half a text height)?"""
    if not boxes_a or not boxes_b:
        return False
    ax0, ax1 = min(b[0] for b in boxes_a), max(b[2] for b in boxes_a)
    bx0, bx1 = min(b[0] for b in boxes_b), max(b[2] for b in boxes_b)
    if min(ax1, bx1) - max(ax0, bx0) < 0.8 * min(ax1 - ax0, bx1 - bx0):
        return False
    h = float(np.median([b[3] - b[1] for b in boxes_a]))
    ya = float(np.median([(b[1] + b[3]) / 2 for b in boxes_a]))
    yb = float(np.median([(b[1] + b[3]) / 2 for b in boxes_b]))
    return abs(ya - yb) <= 0.5 * h


def build_segments(per_frame, max_gap, min_len, keep_edges=False, bridge=0):
    """per_frame: [(text, boxes)] -> [Seg]. keep_edges: also keep short segments that touch the first or last frame
    (a subtitle cut in two where a long video was split for --jobs is not noise).
    bridge: frames of OCR dropout tolerated within one subtitle, when the same text comes back in the same place
    (max_gap applies otherwise). 1304 at 12:29: "行行行" was read for 10 frames, lost for 13 behind a sign that
    slid past, read again for 2 frames - which were dropped as noise and stayed on screen."""
    bridge = max(bridge, max_gap)
    # 1) Consecutive frames with identical text (ignoring spaces) -> runs
    runs = []
    for i, (text, boxes) in enumerate(per_frame):
        if not text:
            continue
        r = runs[-1] if runs else None
        if (r and r.text.replace(" ", "") == text.replace(" ", "")
                and (i - r.end <= max_gap + 1 or (i - r.end <= bridge + 1 and _same_line_place(r.boxes, boxes)))):
            if i - r.end > max_gap + 1:
                r.gaps.append((r.end + 1, i - 1))
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
        if p and r.start - p.end <= bridge + 1 and _x_overlap(p, r):
            s = similar(p.text, r.text)
            rl, pl = r.end - r.start + 1, p.end - p.start + 1
            if r.start - p.end <= max_gap + 1:
                join = (s >= 0.9 or (s >= 0.6 and min(rl, pl) < short)
                        or (_contains(p.text, r.text) and _same_place(p, r)))
            else:  # a longer dropout: only the very same text, in the same place ("卖什么油桃" and "还卖什么油桃" 0.7 s
                # apart are two subtitles with a blank between them, and similar enough for s >= 0.9)
                join = (p.text.replace(" ", "") == r.text.replace(" ", "")) and _same_line_place(p.boxes, r.boxes)
                if join:
                    p.gaps.append((p.end + 1, r.start - 1))
            if join:
                p.gaps += r.gaps
                p.end = r.end
                p.texts.update(r.texts)
                p.boxes += r.boxes
                continue
        segs.append(r)
    last = len(per_frame) - 1
    return [s for s in segs if s.end - s.start + 1 >= min_len
            or (keep_edges and (s.start <= max_gap or s.end >= last - max_gap))]


STYLE_MIN = 0.25  # outline score (masks.outline_score) below which a box is not a subtitle in an outlined video


def outlined_subtitles(per_frame, line):
    """Are this video's subtitles white with a dark outline? Judged from the outline scores run_ocr stores as each
    box's 6th value, on the boxes of the subtitle line: True when at least half of them (and at least 25) score
    STYLE_MIN or more; False otherwise (yellow subtitles, no outline: nothing is filtered by style); None without
    scores. Every frame OCR read once (repeated results of unchanged frames count once)."""
    if not line:
        return None
    mode, h = line[:2]
    scores, prev = [], None
    for boxes in per_frame:
        if boxes == prev:
            continue
        prev = boxes
        for b in boxes:
            if len(b) < 6:
                return None
            if abs((b[1] + b[3]) / 2 - mode) <= 0.5 * h:
                scores.append(b[5])
    good = sum(s >= STYLE_MIN for s in scores)
    return good >= 25 and good >= 0.5 * len(scores)


def segments_from_ocr(per_frame, fps, max_gap=5, min_dur=0.4, line="auto", keep_edges=False, outlined=None, bridge=0):
    """Raw per-frame OCR boxes -> (subtitle line, [Seg]). `line` = a known (y centre, height) to reuse
    instead of detecting it, e.g. when checking an already cleaned video where almost no text is left.
    outlined=True (outlined_subtitles): boxes that do not look like a subtitle are dropped first. Titles, credits,
    signs and print in the picture were read as subtitles (on the subtitle line, or as the upper line of a
    two-line subtitle) and erased; a title in red even fell back to a box mask and was smeared (1304 at 1:12)."""
    if outlined:
        per_frame = [[b for b in boxes if len(b) < 6 or b[5] >= STYLE_MIN] for boxes in per_frame]
    if line == "auto":
        line = subtitle_line(per_frame)
    texts = [join_text(filter_line(list(b), line)) for b in per_frame]
    return line, build_segments(texts, max_gap=max_gap, min_len=max(2, round(min_dur * fps)), keep_edges=keep_edges,
                                bridge=bridge)


def ts(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segs, fps, path, offset=0.0):
    """offset: seconds the first video frame comes after the start of the film (see video.restore_start)"""
    with open(path, "w", encoding="utf-8") as f:
        for k, s in enumerate(segs, 1):
            f.write(f"{k}\n{ts(s.start / fps + offset)} --> {ts((s.end + 1) / fps + offset)}\n{s.text}\n\n")
