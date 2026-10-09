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
from .ads import find_ads, write_report
from .cutter import cut, parse_ranges
from .device import gpu_compute_capability, gpu_memory_gb, onnx_providers
from .parallel import process_parallel
from .pipeline import process
from .refine import refine
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
    ap.add_argument("--subtitles", default="on", choices=["on", "off"],
                    help="off: the video has no hardcoded subtitles; skip OCR, SRT and subtitle erasing and only erase the "
                         "watermark (default: on)")
    ap.add_argument("--list", metavar="FILE",
                    help="batch file: one video per line, '\"path\" --option value ...' (options as on the command line, "
                         "they override the DEFAULT line); '#' starts a comment. Videos run one after another, a failed "
                         "one does not stop the rest, videos whose NAME_clean.mp4 already exists are skipped")
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
    ap.add_argument("--stabilize", type=float, default=0.6,
                    help="blend each inpainted area with the previous frame by up to this weight where the surroundings are "
                         "static, against the frame-to-frame shimmer of the fill (0 = off)")
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
    ap.add_argument("--jobs", default="auto",
                    help="process a long video in this many concurrent processes (it is split into up to 4x this many "
                         "parts at keyframes, parts are merged afterwards). Uses the idle CPU/GPU capacity; videos "
                         "shorter than 4 min per process run normally. auto: up to 4, limited by one process per "
                         "7 GB of GPU memory and per 4 usable CPUs; 1 with LaMa or without a GPU (default: auto)")
    ap.add_argument("--mps", default="auto", choices=["auto", "on", "off"],
                    help="with --jobs on Linux, start the NVIDIA MPS daemon for the run so the processes share the GPU "
                         "concurrently (needs root; auto = silently skip if unavailable; default: auto)")
    ap.add_argument("--keep-parts", action="store_true", help="with --jobs, keep the per-part work directory")
    ap.add_argument("--refine-of", metavar="ORIGINAL",
                    help="INPUT is an already cleaned video of ORIGINAL: erase any subtitle still readable in it and fill "
                         "logo pieces the first pass missed, in one decode/encode (writes NAME_refined.mp4)")
    ap.add_argument("--cut", action="append", metavar="START-END",
                    help="remove a time range (e.g. an advertisement page) from each INPUT video, frame-exact, writing "
                         "NAME_cut.mp4 (+ NAME_cut.srt); no OCR or erasing. Times as seconds, M:S or H:M:S.ms, "
                         "repeatable: --cut 0:40:01.667-0:40:46.958")
    ap.add_argument("--find-ads", action="store_true",
                    help="find advertisement pages spliced into each INPUT film (full-frame pages inside a letterboxed "
                         "film, confirmed by a web address in the picture) and list them in NAME.ads.txt; nothing is "
                         "changed")
    ap.add_argument("--cut-ads", action="store_true",
                    help="like --find-ads, then cut the confirmed ads out (writes NAME_cut.mp4 and NAME_cut.srt)")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def auto_jobs(args):
    """Concurrent processes for --jobs auto: one ProPainter process needs ~6.5 GB of GPU memory and ~4 CPU cores,
    more than 4 did not help on an L20 or an RTX 5090 (GPU and CPU quota are then the limit)"""
    if args.model != "propainter" or args.device == "cpu" or args.srt_only:
        return 1
    return max(1, min(4, int(gpu_memory_gb() // 7), usable_cpus() // 4))


def read_list(path):
    """Batch file -> [(video path, [option tokens])]; the DEFAULT line's options come first (so a line overrides them)"""
    import shlex
    default, entries = [], []
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        try:
            tokens = shlex.split(raw, comments=True)
        except ValueError as e:
            sys.exit(f"{path}:{n}: {e}")
        if not tokens:
            continue
        if tokens[0] == "DEFAULT":
            default = tokens[1:]
        elif tokens[0].startswith("-"):
            sys.exit(f"{path}:{n}: a line starts with the video path, got {tokens[0]!r}")
        else:
            entries.append((tokens[0], default + tokens[1:]))
    return entries


def run_list(args, argv):
    """--list: run every video of the batch file with its own options; -> exit code (1 if any video failed)"""
    base = []  # command-line options of this call, without --list itself, apply to every video
    skip = False
    for tok in argv:
        if skip:
            skip = False
        elif tok == "--list":
            skip = True
        elif not tok.startswith("--list="):
            base.append(tok)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_log_file(out_dir / "clean-clear.log")
    entries = read_list(args.list)
    log(f"Clean Clear {__version__}: batch of {len(entries)} video(s) from {args.list} -> {out_dir.resolve()}")
    results = []
    for k, (video, opts) in enumerate(entries, 1):
        stem = Path(video).stem
        t0 = time.time()
        if (out_dir / (stem + "_clean.mp4")).exists():
            results.append((video, "skipped (already done)", 0.0))
            log(f"\n##### [{k}/{len(entries)}] {Path(video).name}: already done, skipped")
            continue
        log(f"\n##### [{k}/{len(entries)}] {Path(video).name}  options: {' '.join(opts) or '(defaults)'}")
        try:
            code = main(base + opts + [video])
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            log(f"!! {Path(video).name}: {e.code}")
        except Exception as e:
            code = 1
            log(f"!! {Path(video).name}: {e}")
        made = (out_dir / (stem + "_clean.mp4")).exists()
        status = "ok" if code in (0, None) and made else "no output (nothing to erase)" if code in (0, None) else "FAILED"
        results.append((video, status, time.time() - t0))
    log("\n===== batch summary =====")
    lines = [f"{Path(v).name}\t{st}\t{fmt_time(sec) if sec else '-'}" for v, st, sec in results]
    for line in lines:
        log("  " + line.replace("\t", "   "))
    (out_dir / "batch-status.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if any(st == "FAILED" for _, st, _ in results) else 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # never crash on a console code page when printing subtitles
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_argparser().parse_args(argv)
    if args.list:
        return run_list(args, argv)
    args.jobs = auto_jobs(args) if args.jobs == "auto" else max(1, int(args.jobs))
    import cv2
    cv2.setNumThreads(min(8, usable_cpus()))
    files = collect_inputs(args.inputs)
    if not files:
        sys.exit("No video files found")
    out_dir = Path(args.out)
    set_log_file(out_dir / "clean-clear.log")
    if args.find_ads or args.cut_ads:
        device, _ = onnx_providers(args.device)
        ocr = OCR(device, fixed_shape=args.ocr_fixed_shape == "on" or (
            args.ocr_fixed_shape == "auto" and device == "cuda" and gpu_compute_capability() >= 12.0))
        for f in files:
            log(f"\n==> {f.name}: looking for ad pages")
            ads = find_ads(f, ocr)
            log(f"  report -> {write_report(f, out_dir, ads)}")
            sure = [(a["start"], a["end"]) for a in ads if a["status"] == "confirmed"]
            if args.cut_ads and sure:
                cut(f, out_dir, sure, pick_encoder(args.encoder), args.crf)
        return 0
    if args.cut:
        ranges = parse_ranges(args.cut)
        encoder = pick_encoder(args.encoder)
        for f in files:
            cut(f, out_dir, ranges, encoder, args.crf)
        return 0
    log(f"Clean Clear {__version__}: {len(files)} video(s) -> {out_dir.resolve()}")
    models = []

    def load_models():  # lazily: with --jobs the part processes load their own, the parent keeps the GPU free
        if not models:
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
            models.extend([ocr, backend, encoder])
        return models

    out_dir = Path(args.out)
    failed = []
    for f in files:
        try:
            if args.refine_of:
                refine(f, args.refine_of, out_dir, args, *load_models())
                continue
            if args.jobs > 1 and not args.srt_only and process_parallel(f, out_dir, args):
                continue
            process(f, out_dir, args, *load_models())
        except Exception as e:
            log(f"!! failed {f}: {e}")
            failed.append(f)
    log(f"\ndone {len(files) - len(failed)}/{len(files)}")
    return 1 if failed else 0
