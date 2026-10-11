"""Shared constants and helpers."""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

__version__ = "0.3.4"

# Repository root (source checkout / editable install): holds models/, an optional ffmpeg/ folder and ProPainter/
ROOT = Path(__file__).resolve().parents[2]

_logfile = None


def usable_cpus():
    """CPUs this process may really use: the affinity mask, capped by a container CPU quota (cgroup).
    Thread pools sized from os.cpu_count() (128 on a big host) are throttled hard under a small quota."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


def set_log_file(path):
    """Also append every log line (with a timestamp) to `path`"""
    global _logfile
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _logfile = open(path, "a", encoding="utf-8", buffering=1)


def log(*a):
    msg = " ".join(str(x) for x in a)
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:  # e.g. a legacy Windows code page
        enc = sys.stdout.encoding or "ascii"
        print(msg.encode(enc, "replace").decode(enc), flush=True)
    if _logfile:
        _logfile.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def fmt_time(sec):
    h, r = divmod(int(max(0, sec)), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}"


class Progress:
    """Throttled progress line: `[label] done/total  pct  fps  elapsed  ETA`.

    The ETA extrapolates from the work done so far. If the per-frame cost is very uneven (erasing a
    subtitle frame costs ~15x a pass-through frame), pass `cum`: cumulative relative cost with
    cum[i] = cost of frames 0..i-1, so that the ETA is not badly optimistic in the cheap early part."""

    def __init__(self, label, total, every=10.0, cum=None):
        self.label, self.total, self.every, self.cum = label, max(1, int(total)), every, cum
        self.t0 = self.last = time.time()

    def update(self, done, extra=""):
        now = time.time()
        if now - self.last < self.every:
            return
        self.last = now
        el = now - self.t0
        if self.cum is not None:
            k = min(done, len(self.cum) - 1)
            work_done, work_total = float(self.cum[k]), float(self.cum[-1])
        else:
            work_done, work_total = done, self.total
        eta = (work_total - work_done) * el / work_done if work_done > 0 else 0
        pct = min(100.0, 100.0 * done / self.total)
        log(f"  [{self.label}] {done}/{self.total} {pct:.1f}%  {done / max(el, 1e-6):.1f} fps  "
            f"elapsed {fmt_time(el)}  ETA {fmt_time(eta)}" + (f"  {extra}" if extra else ""))
