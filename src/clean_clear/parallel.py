"""--jobs N: split a long video at keyframes, process the parts in N concurrent Clean Clear processes, merge.

One process leaves most of a big machine idle (GPU calls are separated by single-threaded Python work), so
several processes on disjoint CPU sets finish the film much sooner. Parts are independent runs of the normal
pipeline (OCR, masks, watermark, erase, verify), so OCR runs in parallel as well. The watermark mask is
detected once on the whole film so that every part erases the same pixels."""
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .common import fmt_time, log
from .device import gpu_memory_gb
from .pipeline import output_paths, watermark_cache_path, watermark_mask
from .subtitles import similar
from .video import _exe, duration_check, probe

# options forwarded to the part processes (argparse dest names)
_VALUE_OPTS = ["model", "subtitles", "band", "device", "ocr_interval", "ocr_fixed_shape", "encoder", "crf", "mask", "dilate",
               "grow", "shadow", "pad_frames", "max_gap", "min_dur", "min_score", "min_height", "extend_sec",
               "verify", "wm_guard", "stabilize", "propainter_dir", "pp_chunk", "pp_raft_iter", "pp_ctx", "pp_pad",
               "pp_margin", "pp_ref_stride"]
_SRT_TIME = re.compile(r"(\d+):(\d\d):(\d\d),(\d{3}) --> (\d+):(\d\d):(\d\d),(\d{3})")


def start_mps(mode):
    """Start the NVIDIA MPS daemon so that the part processes share the GPU concurrently instead of taking turns
    (about 14 % faster on an L20). `mode`: auto = try, fall back silently; on = try, warn if it fails; off = never.
    Needs Linux, root and nvidia-cuda-mps-control. -> stop function (a no-op if MPS was not started by us)"""
    nop = lambda: None
    if mode == "off" or not sys.platform.startswith("linux"):
        return nop
    exe = shutil.which("nvidia-cuda-mps-control")
    if os.environ.get("CUDA_MPS_PIPE_DIRECTORY") and Path(os.environ["CUDA_MPS_PIPE_DIRECTORY"], "control").exists():
        log("  MPS: already running")
        return nop
    if not exe or os.geteuid() != 0:
        if mode == "on":
            log("  MPS not started: needs root and nvidia-cuda-mps-control; continuing without it")
        return nop
    pipe, logd = "/tmp/nvidia-mps", "/tmp/nvidia-mps-log"
    os.makedirs(pipe, exist_ok=True)
    os.makedirs(logd, exist_ok=True)
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=pipe, CUDA_MPS_LOG_DIRECTORY=logd)
    try:
        subprocess.run([exe, "-d"], env=env, check=True, capture_output=True, timeout=30)
    except (subprocess.SubprocessError, OSError) as e:
        if mode == "on":
            log(f"  MPS not started ({e}); continuing without it")
        return nop
    os.environ.update(CUDA_MPS_PIPE_DIRECTORY=pipe, CUDA_MPS_LOG_DIRECTORY=logd)
    log("  MPS: started")

    def stop():
        try:
            subprocess.run([exe], input="quit\n", text=True, env=env, capture_output=True, timeout=30)
        except (subprocess.SubprocessError, OSError):
            pass
        for k in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"):
            os.environ.pop(k, None)
        log("  MPS: stopped")
    return stop


def _split(src, work, parts, duration):
    times = ",".join(f"{duration * k / parts:.3f}" for k in range(1, parts))
    subprocess.run([_exe("ffmpeg"), "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-c", "copy", "-f", "segment",
                    "-segment_times", times, "-reset_timestamps", "1", str(work / "part%02d.mkv")], check=True)
    return sorted(work.glob("part*.mkv"))


def _child_args(args, wm):
    cmd = []
    for name in _VALUE_OPTS:
        cmd += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    cmd += ["--watermark", wm, "--split-part", "--jobs", "1"]  # a part is never split again
    if args.no_cache:
        cmd.append("--no-cache")
    return cmd


def _fmt_ms(ms):
    h, ms = divmod(int(round(ms)), 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _shift_srt(text, offset):
    """Cues of an SRT with every time stamp shifted by `offset` seconds -> [(start ms, end ms, text)]"""
    def ms(h, m, s, f):
        return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(f)

    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        m = _SRT_TIME.match(lines[1])
        if not m:
            continue
        cues.append((ms(*m.groups()[:4]) + offset * 1000, ms(*m.groups()[4:]) + offset * 1000, "\n".join(lines[2:])))
    return cues


def merge_cues(parts, tol_ms=250):
    """parts: [(offset in seconds, SRT text)] in order -> [(start ms, end ms, text)].

    A subtitle on screen at a split point was recognised in both parts; the two halves are joined into one cue
    (keeping the text of the longer half) when they meet at the split point and read alike."""
    cues = []
    for offset, text in parts:
        new = _shift_srt(text, offset)
        boundary = offset * 1000
        if cues and new:
            a, b = cues[-1], new[0]
            if a[1] >= boundary - tol_ms and b[0] <= boundary + tol_ms and similar(a[2], b[2]) >= 0.6:
                text_ = a[2] if a[1] - a[0] >= b[1] - b[0] else b[2]
                cues[-1] = (a[0], b[1], text_)
                new = new[1:]
        cues += new
    return cues


def _vram_check(args):
    """Warn if `--jobs` processes will not fit in GPU memory (each loads its own models)"""
    total = gpu_memory_gb()
    if not total:
        return
    # measured: ProPainter ~13 GB at --pp-chunk 120 for a 1080p subtitle band (grows with the chunk); LaMa + OCR ~2 GB
    need = (13.0 * args.pp_chunk / 120 + 1.0) if args.model == "propainter" else 2.0
    if args.jobs * need > 0.95 * total:
        log(f"  WARNING: {args.jobs} processes need about {args.jobs * need:.0f} GB of GPU memory, the GPU has "
            f"{total:.0f} GB; expect out-of-memory failures. Use fewer --jobs or a smaller --pp-chunk")


def _last_progress(logfile):
    try:
        for line in reversed(Path(logfile).read_text(encoding="utf-8", errors="replace").splitlines()[-40:]):
            if "[" in line and "%" in line:
                return line.strip()
    except OSError:
        pass
    return "starting"


STALL_SECONDS = 900  # a part that has written nothing for this long is hung
START_STALL_SECONDS = 300  # ... or has not printed its first progress line this long after its start. Seen in 3 of ~35
MAX_RESTARTS = 2           # parts: the process stayed alive right after loading the models, using no CPU, for hours
LAUNCH_GAP = 20  # seconds between two part starts: the hangs happened in the CUDA/ONNX start-up of a process that
                 # started together with others under MPS, so the processes are started one after another


def _stalled(work, k, started):
    """Seconds since the part's output files last changed (counted from its start before the first write)"""
    last = started
    for f in (work / f"part{k:02d}.out", work / f"out{k:02d}" / "clean-clear.log"):
        try:
            last = max(last, f.stat().st_mtime)
        except OSError:
            pass
    return time.time() - last


def _dump_stack(p, work, k):
    """Python stack of a hung part (py-spy, if installed) -> hang_dumps/ next to the work directory"""
    exe = shutil.which("py-spy") or str(Path(sys.executable).with_name("py-spy"))
    if not Path(exe).exists():
        return
    try:
        out = subprocess.run([exe, "dump", "--pid", str(p.pid), "--native"], capture_output=True, text=True,
                             timeout=30).stdout
        d = work.parent / "hang_dumps"
        d.mkdir(exist_ok=True)
        (d / f"{work.name}_part{k:02d}_{time.strftime('%H%M%S')}.txt").write_text(out, encoding="utf-8")
    except (subprocess.SubprocessError, OSError):
        pass


def _kill(p, fh=None):
    try:
        os.killpg(p.pid, 9)  # the part runs in its own session: its ffmpeg children go with it
    except (OSError, ProcessLookupError):
        p.kill()
    p.wait()
    if fh is not None:
        fh.close()


def _run_parts(args, files, work, wm, groups, pending, t_all):
    jobs = args.jobs
    running = {}  # slot -> (part index, Popen, log file)
    done, started, restarts = [], {}, {}
    last_report = last_launch = 0.0
    while pending or running:
        for slot in range(jobs):
            if slot not in running and pending and time.time() - last_launch >= LAUNCH_GAP:
                last_launch = time.time()
                k = pending.pop(0)
                odir = work / f"out{k:02d}"
                shutil.rmtree(odir, ignore_errors=True)
                odir.mkdir()
                cmd = [sys.executable, "-m", "clean_clear", str(files[k]), "-o", str(odir)] + _child_args(args, wm)
                pin = (lambda c=groups[slot]: os.sched_setaffinity(0, c)) if groups[slot] else None
                fh = open(work / f"part{k:02d}.out", "w")
                running[slot] = (k, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, preexec_fn=pin,
                                                     start_new_session=True), fh)
                started[k] = time.time()
                log(f"  part {k + 1}/{len(files)} started (slot {slot})")
        time.sleep(3)
        for slot, (k, p, fh) in list(running.items()):
            rc = p.poll()
            if rc is None:
                no_progress = _last_progress(work / f"out{k:02d}" / "clean-clear.log") == "starting"
                if _stalled(work, k, started[k]) > (START_STALL_SECONDS if no_progress else STALL_SECONDS):
                    _dump_stack(p, work, k)
                    _kill(p, fh)
                    del running[slot]
                    restarts[k] = restarts.get(k, 0) + 1
                    if restarts[k] > MAX_RESTARTS:
                        for _, q, qfh in running.values():
                            _kill(q, qfh)
                        raise RuntimeError(f"part {k + 1} hung {restarts[k]} times; see {work / f'part{k:02d}.out'}")
                    log(f"  part {k + 1} hung (no output, no progress): killed, restarting "
                        f"({restarts[k]}/{MAX_RESTARTS})")
                    pending.insert(0, k)
                continue
            del running[slot]
            fh.close()
            if rc != 0:
                for _, q, qfh in running.values():
                    _kill(q, qfh)
                raise RuntimeError(f"part {k + 1} failed (code {rc}); see {work / f'part{k:02d}.out'}")
            done.append(k)
            log(f"  part {k + 1}/{len(files)} done  ({len(done)}/{len(files)}, {fmt_time(time.time() - t_all)} elapsed)")
        if time.time() - last_report > 60 and running:
            last_report = time.time()
            for slot, (k, _, _) in sorted(running.items()):
                log(f"  part {k + 1}: {_last_progress(work / f'out{k:02d}' / 'clean-clear.log')}")


def process_parallel(src, out_dir, args):
    """-> True if the video was processed in parallel, False if it is too short (the caller runs the normal path)"""
    src = Path(src)
    W, H, fps, n = probe(src)
    duration = n / fps
    jobs = args.jobs
    if duration < 240 * jobs:  # less than 2 min per part with the minimum of 2 parts per process
        return False
    # 4 parts per process, not fewer: the work per part differs a lot (the dense credits at the end of a film took
    # 73 min where the others took 46-56, so with 2 parts per process 3 of 4 processes idled for 29 min at the end)
    parts = min(jobs * 4, int(duration // 120))
    out_dir.mkdir(parents=True, exist_ok=True)
    _vram_check(args)
    t_all = time.time()
    work = out_dir / f".parts_{src.stem}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    log(f"\n==> {src.name}  {W}x{H} {fps:.3f}fps  {fmt_time(duration)}: {parts} parts, {jobs} processes at a time")

    wm = args.watermark
    if wm == "auto":  # detect once on the whole film, hand the same mask to every part
        m = watermark_mask(src, out_dir, W, H, fps, n, args)
        wm = str(watermark_cache_path(out_dir, src)) if m is not None else "off"
    if args.subtitles == "off" and wm == "off":  # nothing to erase: let the normal path say so
        shutil.rmtree(work, ignore_errors=True)
        return False
    files = _split(src, work, parts, duration)
    log(f"  split into {len(files)} parts ({time.time() - t_all:.0f}s)")

    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    groups = [cpus[k::jobs] for k in range(jobs)] if cpus else [None] * jobs  # interleaved: hyper-thread siblings stay together
    pending = list(range(len(files)))
    pending.insert(0, pending.pop())  # the last part first: it holds the credits, the densest subtitles of a film
    stop_mps = start_mps(args.mps)
    try:
        _run_parts(args, files, work, wm, groups, pending, t_all)
    finally:
        stop_mps()

    # merge: subtitles (time-shifted by the length of the cleaned parts before them), video (stream copy) + the
    # original audio. The offsets come from the frames really written, which is what the merged video consists of.
    cleaned = [work / f"out{k:02d}" / (f.stem + "_clean.mp4") for k, f in enumerate(files)]
    srt_parts, offset, frames = [], 0.0, 0
    for k, f in enumerate(files):
        srt = work / f"out{k:02d}" / (f.stem + ".srt")
        srt_parts.append((offset, srt.read_text(encoding="utf-8") if srt.exists() else ""))
        cnt = probe(cleaned[k])[3]
        offset += cnt / fps
        frames += cnt
    cues = merge_cues(srt_parts)
    dst, srt_path, tmp = output_paths(src, out_dir)
    with open(srt_path, "w", encoding="utf-8") as fh:
        for i, (a, b, t) in enumerate(cues, 1):
            fh.write(f"{i}\n{_fmt_ms(a)} --> {_fmt_ms(b)}\n{t}\n\n")
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{c.resolve().as_posix()}'\n" for c in cleaned), encoding="utf-8")
    try:
        subprocess.run([_exe("ffmpeg"), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(src),
                        "-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-movflags", "+faststart", str(tmp)], check=True)
        if abs(frames - n) > 2:
            log(f"  WARNING: the parts have {frames} frames in total, the source about {n}")
        warn = duration_check(src, tmp)
        if warn:
            log(f"  WARNING: {warn}")
        tmp.replace(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if not args.keep_parts:
        shutil.rmtree(work, ignore_errors=True)
    log(f"  {len(cues)} subtitles -> {srt_path}")
    log(f"  video -> {dst}  ({time.time() - t_all:.0f}s total, {jobs} processes)")
    return True
