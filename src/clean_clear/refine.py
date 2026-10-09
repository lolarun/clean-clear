"""--refine-of ORIGINAL: post-fix an already cleaned video in one decode and one encode.

Looks at the cleaned video again: any subtitle that is still readable is erased, and logo pieces the first pass
did not cover (see watermark.detect near_small) are filled. Much cheaper than a complete run, because the
expensive watermark pass over every frame is not repeated."""
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .common import fmt_time, log
from .masks import box_masks, frame_masks, glyph_masks
from .pipeline import _runs, parse_band, run_ocr
from .subtitles import frame_boxes, segments_from_ocr, subtitle_line
from .video import ThreadedWriter, _exe, detect_cuts, open_writer, prefetch, probe, read_frames
from . import watermark
from .rewrite import plain_timeline, rewrite
from .stabilize import stabilize


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


def first_pass_watermark(original, dirs, H, W):
    """The watermark mask the first pass really erased, from its cache in one of `dirs` (an all-black mask: none was
    erased); None if no cache is found. Detecting it again with today's code would also return the pieces that only
    newer versions find, and refine would then re-fill (and blur) pixels that were already clean."""
    from .pipeline import watermark_cache_path
    for d in dirs:
        for p in (watermark_cache_path(d, original), Path(d) / ".cache" / f"{original.stem}.watermark.png"):
            if p.exists():
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if m is not None and m.shape == (H, W):
                    log(f"  first-pass watermark mask: {p}")
                    return m > 127
    return None


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
        old = first_pass_watermark(original, [out_dir, cleaned.parent], H, W)
        if old is None:
            log("  no watermark cache of the first pass found; assuming it erased what version 0.3.2 detected")
            old = watermark.detect(original, W, H, n / fps, near_small=False)
        new = watermark.detect(original, W, H, n / fps)
        if old is not None and old.any() and new is not None:
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
    if extra is None and not frame_seg:
        log("  nothing to fix")
        return None
    if extra is None and not plain_timeline(cleaned):
        log("  input has hidden pre-roll frames (stream-copy cut): re-encoding the whole file instead of windows")
        extra = np.zeros((H, W), bool)  # take the full pass below; an empty fill mask changes nothing
    tmp = out_dir / f".{dst.stem}.tmp.mp4"  # renamed to dst only when complete
    try:
        if extra is None:  # only subtitles to fix: re-encode just the windows around them, copy the rest

            def fix(frames, a, b):
                fs = {i - a: k for i, k in frame_seg.items() if a <= i < b}
                cw = [c - a for c in cuts if a < c < b]
                return stabilize(backend.erase(frames, fs, masks, cw), fs, masks, cw, args.stabilize, label="refine")

            rewrite(cleaned, original, tmp, [(min(r), max(r)) for r in _runs(frame_seg)], fix, encoder, args.crf)
        else:  # logo pixels change on every frame: one full pass
            wr = open_writer(original, tmp, W, H, fps, encoder, args.crf)  # audio from the original
            tw = ThreadedWriter(wr)
            stream = None
            try:
                stream = prefetch(read_frames(cleaned, W, H))
                if extra.any():
                    stream = watermark.spatial_fill(stream, extra)
                if frame_seg:
                    stream = stabilize(backend.erase(stream, frame_seg, masks, cuts), frame_seg, masks, cuts,
                                       args.stabilize, label="refine")
                for frame in stream:
                    tw.write(np.ascontiguousarray(frame).tobytes())
            finally:
                if stream is not None:
                    stream.close()
                tw.close()
            if wr.returncode != 0:
                raise RuntimeError(f"ffmpeg encoding failed (code {wr.returncode})")
        tmp.replace(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    log(f"  refined video -> {dst}  ({time.time() - t0:.0f}s)")
    return dst
