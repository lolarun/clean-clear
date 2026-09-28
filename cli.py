"""Command-line entry point: batch-process files and directories."""
import argparse
import os
import sys
from pathlib import Path

import backends
from common import ROOT, __version__, log
from device import onnx_providers
from pipeline import process
from subtitles import OCR
from video import VIDEO_EXTS, pick_encoder


def collect_inputs(inputs):
    files = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        elif p.exists():
            files.append(p)
        else:
            log(f"Skipping missing path: {p}")
    return files


def build_argparser():
    ap = argparse.ArgumentParser(prog="cleanframe",
                                 description="Clean Frame: remove hardcoded subtitles and extract them as SRT")
    ap.add_argument("inputs", nargs="+", help="video files or directories")
    ap.add_argument("-o", "--out", default="output", help="output directory (default: ./output)")
    ap.add_argument("-m", "--model", default="lama", choices=backends.MODELS,
                    help="inpainting model: lama (fast, default) or propainter (video model, slower)")
    ap.add_argument("--band", default="0.70,1.0",
                    help="horizontal band containing subtitles, as fractions or pixels, e.g. 0.7,1.0 or 800,1080 "
                         "(default: bottom 30%%)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "dml", "cpu"],
                    help="device for OCR and LaMa (ProPainter always uses CUDA if available)")
    ap.add_argument("--srt-only", action="store_true", help="extract subtitles only, do not erase")
    ap.add_argument("--no-cache", action="store_true", help="ignore the OCR cache and run OCR again")
    ap.add_argument("--ocr-interval", type=int, default=10,
                    help="max frames to skip OCR while the subtitle band is unchanged (1 = OCR every frame)")
    ap.add_argument("--encoder", default="auto", help="auto / h264_nvenc / libx264 / hevc_nvenc ...")
    ap.add_argument("--crf", type=int, default=18, help="quality, lower is better (default: 18)")
    ap.add_argument("--mask", default="glyph", choices=["glyph", "box"],
                    help="glyph: erase only the strokes, sharper (default); box: erase whole text boxes")
    ap.add_argument("--dilate", type=int, default=10, help="text box dilation in pixels")
    ap.add_argument("--grow", type=int, default=8, help="glyph mask dilation in pixels (covers the outline)")
    ap.add_argument("--shadow", type=int, default=3,
                    help="extra glyph mask dilation towards the lower right (covers the shadow)")
    ap.add_argument("--pad-frames", type=int, default=1,
                    help="extra frames erased before/after each subtitle (fade in/out)")
    ap.add_argument("--max-gap", type=int, default=5, help="missed frames tolerated within one subtitle")
    ap.add_argument("--min-dur", type=float, default=0.4,
                    help="subtitles shorter than this many seconds are treated as noise")
    ap.add_argument("--min-score", type=float, default=0.6, help="OCR confidence threshold")
    ap.add_argument("--min-height", type=float, default=0.015, help="minimum text height as a fraction of frame height")

    ap.add_argument("--watermark", default="off", metavar="off|auto|MASK.png",
                    help="also erase a static corner logo on every frame: auto = detect it, or give a mask image "
                         "of the video's size (white = watermark) (default: off)")

    pp = ap.add_argument_group("ProPainter options")
    pp.add_argument("--propainter-dir", default=os.environ.get("PROPAINTER_DIR", str(ROOT / "ProPainter")),
                    help="ProPainter checkout with weights/ (default: $PROPAINTER_DIR or ./ProPainter)")
    pp.add_argument("--pp-chunk", type=int, default=120,
                    help="frames per ProPainter chunk; lower it if GPU memory runs out (80 for 8 GB cards)")
    pp.add_argument("--pp-raft-iter", type=int, default=12,
                    help="optical flow iterations, fewer is faster (default: 12; was 20)")
    pp.add_argument("--pp-ctx", type=int, default=10,
                    help="context frames added on each side of a chunk for flow estimation")
    pp.add_argument("--pp-pad", type=int, default=8,
                    help="extra frames inpainted before/after each run of subtitle frames")
    pp.add_argument("--pp-margin", type=int, default=80,
                    help="horizontal margin (px) kept around the subtitle when cropping columns to process")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    files = collect_inputs(args.inputs)
    if not files:
        sys.exit("No video files found")
    device, prov = onnx_providers(args.device)
    log(f"device: {device}  providers={prov}")
    ocr = OCR(device)
    backend = encoder = None
    if not args.srt_only:
        backend = backends.create(args.model, providers=prov, propainter_dir=args.propainter_dir,
                                  chunk=args.pp_chunk, raft_iter=args.pp_raft_iter,
                                  ctx=args.pp_ctx, pad=args.pp_pad, margin=args.pp_margin)
        encoder = pick_encoder(args.encoder)
        log(f"model: {args.model}  encoder: {encoder}")
    out_dir = Path(args.out)
    failed = []
    for f in files:
        try:
            process(f, out_dir, args, ocr, backend, encoder)
        except Exception as e:
            log(f"!! failed {f}: {e}")
            failed.append(f)
    log(f"\ndone {len(files) - len(failed)}/{len(files)}")
    return 1 if failed else 0
