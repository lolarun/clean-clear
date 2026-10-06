"""--refine-of ORIGINAL: post-fix an already cleaned video in one decode and one encode.

Looks at the cleaned video again: any subtitle that is still readable is erased, and logo pieces the first pass
did not cover (see watermark.detect near_small) are filled. Much cheaper than a complete run, because the
expensive watermark pass over every frame is not repeated."""
import subprocess
import time
from pathlib import Path

import numpy as np

from .common import fmt_time, log
from .masks import box_masks, frame_masks, glyph_masks
from .pipeline import parse_band, run_ocr
from .subtitles import frame_boxes, segments_from_ocr, subtitle_line
from .video import ThreadedWriter, _exe, detect_cuts, open_writer, prefetch, probe, read_frames
from . import watermark


def estimate_line(src, W, H, band, ocr, args, duration, samples=300):
    """Where the subtitles sit in `src` (y centre, text height), from OCR of evenly spaced frames"""
    min_h = max(8, int(H * args.min_height))
    per_frame = []
    for t in np.linspace(duration * 0.02, duration * 0.98, samples):
        raw = subprocess.run([_exe("ffmpeg"), "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
                              "-vf", f"crop={W}:{band[1] - band[0]}:0:{band[0]}", "-f", "rawvideo", "-pix_fmt", "bgr24",
                              "-"], capture_output=True).stdout
        if len(raw) == W * (band[1] - band[0]) * 3:
            img = np.frombuffer(raw, np.uint8).reshape(band[1] - band[0], W, 3)
            per_frame.append(frame_boxes(ocr(img), band[0], W, min_h, args.min_score))
    return subtitle_line(per_frame)


def refine(cleaned, original, out_dir, args, ocr, backend, encoder):
    cleaned, original = Path(cleaned), Path(original)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H, fps, n = probe(cleaned)
    band = parse_band(args.band, H)
    log(f"\n==> refine {cleaned.name} (original: {original.name})  {W}x{H} {fps:.3f}fps  ~{n} frames")
    t0 = time.time()
    line = estimate_line(original, W, H, band, ocr, args, n / fps)
    if line is None:
        raise RuntimeError("no subtitle line found in the original")
    log(f"  subtitle line y~{line[0]:.0f}, text height ~{line[1]:.0f}px ({time.time() - t0:.0f}s)")

    extra = None
    if args.watermark == "auto":  # logo pieces the first pass did not erase
        old = watermark.detect(original, W, H, n / fps, near_small=False)  # exactly what the first pass erased
        new = watermark.detect(original, W, H, n / fps)
        if old is not None and new is not None:
            extra = new.copy()
            acc = watermark.leftover_accents(original, W, H, n / fps, new)
            if acc is not None:
                extra |= acc
            extra &= ~old
            if extra.any():
                log(f"  {int(extra.sum())} logo pixels were not covered by the first pass; filling them on every frame")
            else:
                extra = None

    per_frame = run_ocr(cleaned, W, H, n, band, ocr, args, label="refine")
    _, segs = segments_from_ocr(per_frame, fps, max_gap=args.max_gap, min_dur=args.min_dur * 0.5, line=line)
    log(f"  {len(segs)} subtitle(s) still visible" + (" at " + ", ".join(fmt_time(s.start / fps) for s in segs[:30]) if segs else ""))

    frame_seg, masks = {}, {}
    if segs:
        if args.mask == "glyph":
            masks = glyph_masks(cleaned, W, H, band, segs, args.dilate, args.grow, args.shadow, total=n,
                                extend=round(args.extend_sec * fps))
        else:
            masks = box_masks(segs, H, W, args.dilate)
        frame_seg, masks = frame_masks(segs, masks, args.pad_frames)

    cuts = detect_cuts(cleaned, total=n) if getattr(backend, "uses_cuts", False) and frame_seg else []
    dst = out_dir / (cleaned.stem.removesuffix("_clean") + "_refined.mp4")
    wr = open_writer(original, dst, W, H, fps, encoder, args.crf)  # audio from the original
    tw = ThreadedWriter(wr)
    try:
        stream = prefetch(read_frames(cleaned, W, H))
        if extra is not None:
            stream = watermark.spatial_fill(stream, extra)
        if frame_seg:
            stream = backend.erase(stream, frame_seg, masks, cuts)
        for frame in stream:
            tw.write(np.ascontiguousarray(frame).tobytes())
    finally:
        tw.close()
    if wr.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed (code {wr.returncode})")
    log(f"  refined video -> {dst}  ({time.time() - t0:.0f}s)")
    return dst
