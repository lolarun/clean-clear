"""Per-video pipeline: OCR -> SRT -> masks -> [watermark] -> erase (LaMa or ProPainter) -> encode."""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .common import Progress, fmt_time, log
from .masks import box_masks, frame_masks, glyph_masks, white_pixels
from .subtitles import frame_boxes, segments_from_ocr, write_srt
from . import watermark
from .rewrite import rewrite
from .stabilize import stabilize
from .video import ThreadedWriter, detect_cuts, open_writer, prefetch, probe, read_frames


def parse_band(s, H):
    """'0.7,1.0' (fractions) or '800,1080' (pixels) -> (y0, y1) in pixels, even-aligned"""
    a, b = (float(v) for v in s.split(","))
    if a <= 1 and b <= 1:
        a, b = a * H, b * H
    a, b = int(a) // 2 * 2, int(b) // 2 * 2
    return max(0, a), min(H, b)


def run_ocr(src, W, H, n, band, ocr, args, label="ocr"):
    """Pass 1: per-frame OCR boxes of the subtitle band. If the white-pixel layout barely changed since
    the last OCR'd frame, the subtitle is the same and the result is reused (OCR is still forced every
    --ocr-interval frames)."""
    t0 = time.time()
    per_frame = []
    min_h = max(8, int(H * args.min_height))
    ref, last, n_ocr, txt = None, -1, 0, ""
    prog = Progress(label, n)
    for i, strip in enumerate(read_frames(src, W, H, band)):
        sig = white_pixels(strip[::2, ::2])
        if ref is not None and i - last < args.ocr_interval:
            diff = np.count_nonzero(sig ^ ref)
            if diff <= max(30, 0.08 * np.count_nonzero(ref)):
                per_frame.append(per_frame[-1])
                prog.update(i + 1, txt)
                continue
        boxes = frame_boxes(ocr(strip), band[0], W, min_h, args.min_score)
        per_frame.append(boxes)
        ref, last = sig, i
        n_ocr += 1
        if boxes:
            txt = " / ".join(b[4] for b in boxes)
        prog.update(i + 1, txt)
    log(f"  OCR ran on {n_ocr}/{len(per_frame)} frames ({time.time() - t0:.0f}s)")
    return per_frame


def watermark_mask(src, out_dir, W, H, fps, n, args):
    """--watermark off | auto | <mask image>: HxW bool mask of a static overlay to remove on every frame, or None"""
    if args.watermark == "off":
        return None
    if args.watermark != "auto":
        img = cv2.imread(args.watermark, cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (H, W):
            raise ValueError(f"--watermark mask must be a {W}x{H} image (white = watermark): {args.watermark}")
        return img > 127
    cache = out_dir / ".cache" / f"{src.stem}.watermark.png"  # cached like OCR; an all-black image means none found
    if cache.exists() and not args.no_cache:
        m = cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE) > 127
        log(f"  using watermark cache {cache}")
    else:
        t0 = time.time()
        found = watermark.detect(src, W, H, n / fps)
        if found is not None:  # semi-transparent bits of the logo (a green tick) that the static-colour test misses
            acc = watermark.leftover_accents(src, W, H, n / fps, found)
            if acc is not None:
                found = found | acc
        m = found if found is not None else np.zeros((H, W), bool)
        cache.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(cache), m.astype(np.uint8) * 255)
        log(f"  watermark detection ({time.time() - t0:.0f}s), mask saved to {cache}")
    if not m.any():
        log("  no watermark found")
        return None
    return m


def verify_and_fix(src, dst, W, H, fps, n, band, line, cuts, args, ocr, backend, encoder):
    """Second look at the finished video: OCR its subtitle band again. Whatever is still readable was missed by
    the first pass (OCR lost it, mask too small, ...); it is erased in one more pass that re-encodes only the
    keyframe windows around it and copies the rest of the file."""
    if line is None:
        return
    per_frame = run_ocr(dst, W, H, n, band, ocr, args, label="verify")
    _, segs = segments_from_ocr(per_frame, fps, max_gap=args.max_gap, min_dur=args.min_dur * 0.5, line=line)
    if not segs:
        log("  verify: no subtitle left in the result")
        return
    log(f"  verify: {len(segs)} subtitle(s) still visible ({sum(s.end - s.start + 1 for s in segs)} frames) at "
        + ", ".join(fmt_time(s.start / fps) for s in segs[:20]) + (" ..." if len(segs) > 20 else "") + "; erasing")
    if args.mask == "glyph":
        masks = glyph_masks(dst, W, H, band, segs, args.dilate, args.grow, args.shadow, total=n,
                            extend=round(args.extend_sec * fps))
    else:
        masks = box_masks(segs, H, W, args.dilate)
    frame_seg, masks = frame_masks(segs, masks, args.pad_frames)
    tmp = dst.with_name(dst.stem + ".fix.mp4")

    def fix(frames, a, b):  # one keyframe-aligned window; indices relative to its first frame
        fs = {i - a: k for i, k in frame_seg.items() if a <= i < b}
        cw = [c - a for c in cuts if a < c < b]
        return stabilize(backend.erase(frames, fs, masks, cw), fs, masks, cw, args.stabilize, label="verify")

    # only the windows around the residual subtitles are re-encoded; the rest of the file is copied untouched
    rewrite(dst, src, tmp, [(min(r), max(r)) for r in _runs(frame_seg)], fix, encoder, args.crf)
    tmp.replace(dst)


def _runs(frame_seg, gap=10):
    """Frame indices of `frame_seg` grouped into runs (gaps <= `gap` frames joined)"""
    runs = []
    for i in sorted(frame_seg):
        if runs and i - runs[-1][-1] <= gap:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def process(src, out_dir, args, ocr, backend, encoder):
    src = Path(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H, fps, n = probe(src)
    band = parse_band(args.band, H)
    log(f"\n==> {src.name}  {W}x{H} {fps:.3f}fps  ~{n} frames  subtitle band y={band[0]}-{band[1]}")

    # Pass 1: OCR (raw results are cached as JSON so reruns skip recognition)
    t0 = time.time()
    cache = out_dir / ".cache" / f"{src.stem}.{band[0]}-{band[1]}.ocr.json"
    if cache.exists() and not args.no_cache:
        per_frame = json.loads(cache.read_text(encoding="utf-8"))
        log(f"  using OCR cache {cache}")
    else:
        per_frame = run_ocr(src, W, H, n, band, ocr, args)
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(per_frame, ensure_ascii=False), encoding="utf-8")
    line, segs = segments_from_ocr(per_frame, fps, max_gap=args.max_gap, min_dur=args.min_dur)
    if line:
        log(f"  subtitle line y~{line[0]:.0f}, text height ~{line[1]:.0f}px")
    srt = out_dir / (src.stem + ".srt")
    write_srt(segs, fps, srt)
    log(f"  {len(segs)} subtitles -> {srt}  ({time.time() - t0:.0f}s)")
    if backend is None:
        return

    # One fixed mask per subtitle keeps the result temporally stable
    t0 = time.time()
    if args.mask == "glyph":
        masks = glyph_masks(src, W, H, band, segs, args.dilate, args.grow, args.shadow, total=n,
                            extend=round(args.extend_sec * fps))
    else:
        masks = box_masks(segs, H, W, args.dilate)
    frame_seg, masks = frame_masks(segs, masks, args.pad_frames)
    log(f"  {args.mask} masks done ({time.time() - t0:.0f}s)")

    cuts = []
    if getattr(backend, "uses_cuts", False):
        t0 = time.time()
        cuts = detect_cuts(src, total=n)
        log(f"  {len(cuts)} shot cuts ({time.time() - t0:.0f}s)")

    wm = watermark_mask(src, out_dir, W, H, fps, n, args)

    # Pass 2: erase + encode
    t0 = time.time()
    dst = out_dir / (src.stem + "_clean.mp4")
    wr = open_writer(src, dst, W, H, fps, encoder, args.crf)
    tw = ThreadedWriter(wr)  # encoding overlaps with the next frame's inference
    written = 0
    # relative cost per frame for the ETA (A10 measurements, ms): pass-through ~7, watermark ~39, subtitle frame ~147
    cost = np.full(n + 1, 7.0 + (39.0 if wm is not None else 0.0))
    cost[[i for i in frame_seg if i <= n]] += 147.0
    prog = Progress("erase", n, cum=np.concatenate([[0.0], np.cumsum(cost)]))
    try:
        # decoding (ffmpeg) overlaps with inference in the main thread instead of alternating with it
        stream = prefetch(read_frames(src, W, H))
        if wm is not None:  # watermark: the same erase() on every frame; chained lazily, so still one decode + one encode
            log("  erasing watermark on every frame")
            stream = backend.erase(stream, {i: 0 for i in range(n + 50)}, {0: wm}, cuts)
            if args.wm_guard > 0:
                stream = watermark.guard_fill(stream, wm, args.wm_guard)
            stream = stabilize(stream, {i: 0 for i in range(n + 50)}, {0: wm}, cuts, args.stabilize, label="watermark")
        stream = stabilize(backend.erase(stream, frame_seg, masks, cuts), frame_seg, masks, cuts, args.stabilize,
                           label="subtitles")
        for frame in stream:
            tw.write(np.ascontiguousarray(frame).tobytes())
            written += 1
            prog.update(written)
    finally:
        tw.close()
    if wr.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed (code {wr.returncode})")
    log(f"  video -> {dst}  ({time.time() - t0:.0f}s, {backend.name})")

    if args.verify == "on":
        t0 = time.time()
        verify_and_fix(src, dst, W, H, fps, n, band, line, cuts, args, ocr, backend, encoder)
        log(f"  verify done ({time.time() - t0:.0f}s)")
