"""Command-line entry point: batch-process files and directories."""
import warnings

warnings.filterwarnings("ignore")  # PyTorch / RAFT deprecation noise would bury the progress lines

import argparse
import os
import sys
import time
from pathlib import Path

from . import backends
from .common import ROOT, __version__, fmt_time, log, set_log_file, usable_cpus

# BLAS / OpenMP pools default to every core of the host; under a container CPU quota that is very slow
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, str(min(8, usable_cpus())))
from .device import gpu_compute_capability, onnx_providers
from .pipeline import process
from .subtitles import OCR
from .video import VIDEO_EXTS, pick_encoder


def collect_inputs(inputs):
    """Video files of the given files/directories. Our own `*_clean.*` outputs are skipped, so the
    output can safely go to the input folder and reruns do not process results again."""
    files = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += sorted(f for f in p.iterdir()
                            if f.suffix.lower() in VIDEO_EXTS and not f.stem.endswith("_clean"))
        elif p.exists():
            files.append(p)
        else:
            log(f"Skipping missing path: {p}")
    return files


def build_argparser():
    ap = argparse.ArgumentParser(prog="clean-clear",
                                 description="Clean Clear: remove hardcoded subtitles and extract them as SRT")
    ap.add_argument("inputs", nargs="*", default=["."],
                    help="video files or directories (default: the current directory)")
    ap.add_argument("-o", "--out", default=".", help="output directory (default: the current directory)")
    ap.add_argument("-m", "--model", default="propainter", choices=backends.MODELS,
                    help="inpainting model: propainter (video model, default) or lama (much faster, lower quality)")
    ap.add_argument("--band", default="0.70,1.0",
                    help="horizontal band containing subtitles, as fractions or pixels, e.g. 0.7,1.0 or 800,1080 "
                         "(default: bottom 30%%)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "dml", "cpu"],
                    help="device for OCR and LaMa (ProPainter always uses CUDA if available)")
    ap.add_argument("--srt-only", action="store_true", help="extract subtitles only, do not erase")
    ap.add_argument("--no-cache", action="store_true", help="ignore the OCR cache and run OCR again")
    ap.add_argument("--ocr-interval", type=int, default=10,
                    help="max frames to skip OCR while the subtitle band is unchanged (1 = OCR every frame)")
    ap.add_argument("--ocr-fixed-shape", default="auto", choices=["auto", "on", "off"],
                    help="feed OCR recognition one fixed input shape. Works around multi-second stalls per new input shape "
                         "observed on an RTX 5090 with onnxruntime-gpu 1.23. auto = on for compute capability 12+ (default)")
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
    ap.add_argument("--extend-sec", type=float, default=3.0,
                    help="when OCR loses a subtitle part of the time, keep erasing for up to this many seconds before/after "
                         "it while its glyphs are still visible (0 = off, glyph masks only)")
    ap.add_argument("--verify", default="on", choices=["on", "off"],
                    help="after erasing, OCR the result again and erase any subtitle that is still readable (default: on)")
    ap.add_argument("--wm-guard", type=float, default=25.0,
                    help="repair watermark fills whose mean brightness differs from their surroundings by more than this many "
                         "grey levels (0 = off)")
    ap.add_argument("--min-score", type=float, default=0.6, help="OCR confidence threshold")
    ap.add_argument("--min-height", type=float, default=0.015, help="minimum text height as a fraction of frame height")

    ap.add_argument("--watermark", default="auto", metavar="auto|off|MASK.png",
                    help="erase a static corner logo on every frame: auto = detect it (default), off = leave it, "
                         "or a mask image of the video's size (white = watermark)")

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
    pp.add_argument("--pp-ref-stride", type=int, default=10,
                    help="frame interval between global reference frames fed to the transformer fusion step; "
                         "lower it for more temporal consistency (less flicker in dark/fast-motion scenes) at the "
                         "cost of speed and VRAM")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # never crash on a console code page when printing subtitles
    args = build_argparser().parse_args(argv)
    import cv2
    cv2.setNumThreads(min(8, usable_cpus()))
    files = collect_inputs(args.inputs)
    if not files:
        sys.exit("No video files found")
    out_dir = Path(args.out)
    set_log_file(out_dir / "clean-clear.log")
    log(f"Clean Clear {__version__}: {len(files)} video(s) -> {out_dir.resolve()}")
    device, prov = onnx_providers(args.device)
    log(f"device: {device}  providers={prov}")
    fixed = args.ocr_fixed_shape == "on" or (
        args.ocr_fixed_shape == "auto" and device == "cuda" and gpu_compute_capability() >= 12.0)
    if fixed:
        log("OCR: fixed recognition input shape")
    ocr = OCR(device, fixed_shape=fixed)
    backend = encoder = None
    if not args.srt_only:
        backend = backends.create(args.model, providers=prov, propainter_dir=args.propainter_dir,
                                  chunk=args.pp_chunk, raft_iter=args.pp_raft_iter,
                                  ctx=args.pp_ctx, pad=args.pp_pad, margin=args.pp_margin,
                                  ref_stride=args.pp_ref_stride)
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
