#!/usr/bin/env python3
"""Server worker: processes the jobs the controller (Sol) puts into <work>/queue/, one after another.

Per job: download the source from Google Drive by file ID (rclone) and verify its size and MD5, probe it, run
clean-clear with the probe's options plus the job's own, run clean-clear --check, upload the result to the Drive
staging folder (not to the delivery: the controller reviews every result first) and the check material to
log/qc/, then delete the local copies. The first job on a machine starts with a GPU health check on a 3-minute clip.

Writes, locally and to Drive (log/servers/<server_id>/): heartbeat.json (every few minutes and on every change),
jobs/<asset_id>.json (the job's state). Locally also status.txt, a human summary.

Run under supervisor:  <tool>/.venv/bin/python <tool>/scripts/worker.py /workspace/worker.json
Only the Python standard library, rclone, ffmpeg and the clean-clear command are used.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ACTIVE = ("downloading", "health", "probing", "processing", "checking", "uploading")
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".flv", ".ts", ".m4v", ".wmv", ".webm")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Worker:
    def __init__(self, cfg_path):
        self.cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        c = self.cfg
        self.server = c["server_id"]
        self.tool = Path(c["tool"])
        self.cc = str(self.tool / ".venv" / "bin" / "clean-clear")
        self.work = Path(c.get("work", "/workspace"))
        self.remote = c.get("remote", "gdrive:")
        self.staging = c.get("staging", "staging")
        self.logdir = c.get("log", "log")
        self.hb_every = int(c.get("heartbeat_seconds", 300))
        for d in ("queue", "jobs", "done", "failed"):
            (self.work / d).mkdir(parents=True, exist_ok=True)
        self.current = None  # (asset_id, state)
        self.child = None
        self.stop = False
        self.last_hb = 0.0
        self.gpu = {}
        self.degraded = False
        self.samples = []
        self.version = self._version()
        threading.Thread(target=self._gpu_loop, daemon=True).start()

    # ---------- helpers ----------
    def _version(self):
        try:
            return subprocess.run([self.cc, "--version"], capture_output=True, text=True).stdout.strip()
        except OSError:
            return "unknown"

    def log(self, msg):
        line = f"{now()} {msg}"
        print(line, flush=True)
        if self.current:
            with open(self.job_dir / "worker.log", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def rclone(self, *args, retries=3):
        for k in range(retries):
            r = subprocess.run(["rclone", *args], capture_output=True, text=True)
            if r.returncode == 0:
                return r.stdout
            time.sleep(10 * (k + 1))
        raise RuntimeError(f"rclone {' '.join(args[:2])} failed: {r.stderr.strip()[-300:]}")

    def put(self, local, remote_path):
        self.rclone("copyto", str(local), self.remote + remote_path)

    def remote_md5(self, remote_path):
        out = json.loads(self.rclone("lsjson", "--hash", "--files-only", self.remote + remote_path))
        return out[0].get("Hashes", {}).get("md5") if out else None

    def _gpu_loop(self):
        """Sample the GPU every 10 s. Degraded: under full load (>= 85 %) the card sat in P8 or below 40 % of its
        maximum clock in 54 of the last 60 samples (10 minutes); the host limits it, other parameters will not help."""
        while True:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=pstate,clocks.sm,clocks.max.sm,power.draw,"
                                      "utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=20).stdout.splitlines()[0]
                p, clk, mx, pw, util, mem = [x.strip() for x in out.split(",")]
                g = {"pstate": p, "clock": int(clk), "max_clock": int(mx), "power_w": float(pw),
                     "util": int(util), "memory_mb": int(mem), "time": now()}
                self.gpu = g
                busy = g["util"] >= 85
                low = p == "P8" or g["clock"] < 0.4 * max(1, g["max_clock"])
                self.samples = (self.samples + [busy and low])[-60:]
                self.degraded = len(self.samples) == 60 and sum(self.samples) >= 54
            except Exception:  # no GPU tool or a transient error: keep the last sample
                pass
            time.sleep(10)

    @property
    def job_dir(self):
        a, n = self.current_job["asset_id"], self.current_job.get("attempt", 1)
        return self.work / "jobs" / a / f"attempt_{n}"

    def health(self):
        p = self.work / "health.json"
        return json.loads(p.read_text()) if p.exists() else {"status": "unknown"}

    def heartbeat(self, force=False):
        if not force and time.time() - self.last_hb < self.hb_every:
            return
        self.last_hb = time.time()
        free = shutil.disk_usage(self.work).free / 1e9
        hb = {"server_id": self.server, "time": now(), "epoch": round(time.time()), "version": self.version,
              "health": self.health(),
              "gpu": self.gpu, "gpu_degraded": self.degraded, "disk_free_gb": round(free, 1),
              "current": {"asset_id": self.current[0], "state": self.current[1]} if self.current else None,
              "queue": sorted(p.stem for p in (self.work / "queue").glob("*.json")),
              "done": len(list((self.work / "done").glob("*.json"))),
              "failed": len(list((self.work / "failed").glob("*.json")))}
        f = self.work / "heartbeat.json"
        f.write_text(json.dumps(hb, ensure_ascii=False, indent=1), encoding="utf-8")
        tail = ""
        if self.current:
            cl = self.job_dir / "out" / "clean-clear.log"
            if cl.exists():
                tail = "".join(cl.read_text(encoding="utf-8", errors="replace").splitlines(True)[-3:])
        (self.work / "status.txt").write_text(
            f"{hb['time']}  {self.server}  {self.version}\nhealth: {hb['health'].get('status')}  "
            f"gpu: {self.gpu}  degraded: {self.degraded}\ndisk free: {hb['disk_free_gb']} GB\n"
            f"current: {hb['current']}\nqueue: {hb['queue']}\ndone: {hb['done']}  failed: {hb['failed']}\n{tail}",
            encoding="utf-8")
        try:
            self.put(f, f"{self.logdir}/servers/{self.server}/heartbeat.json")
        except RuntimeError as e:
            print(f"{now()} heartbeat upload failed: {e}", flush=True)

    def set_state(self, state, **extra):
        a = self.current_job["asset_id"]
        self.current = (a, state)
        st_path = self.job_dir / "state.json"
        st = json.loads(st_path.read_text()) if st_path.exists() else {
            "asset_id": a, "batch_id": self.current_job["batch_id"], "attempt": self.current_job.get("attempt", 1),
            "server_id": self.server, "history": []}
        st.update(extra, state=state, updated_at=now())
        st["history"].append([now(), state])
        st_path.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        self.log(f"{a}: {state}")
        try:
            self.put(st_path, f"{self.logdir}/servers/{self.server}/jobs/{a}.json")
        except RuntimeError as e:
            self.log(f"state upload failed: {e}")
        self.heartbeat(force=True)

    def run(self, cmd, logfile):
        """Run a command, output to logfile; returns the exit code. The process can be stopped by stop()."""
        with open(logfile, "a", encoding="utf-8") as fh:
            self.child = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            while self.child.poll() is None:
                self.heartbeat()
                time.sleep(5)
            code, self.child = self.child.returncode, None
        return code

    # ---------- job steps ----------
    def download(self, job, d):
        src = job["source"]
        inn = d / "in"
        inn.mkdir(parents=True, exist_ok=True)
        ext = Path(src["name"]).suffix.lower() or ".mp4"
        dst = inn / (job["asset_id"] + ext)
        if not (dst.exists() and dst.stat().st_size == int(src["size"])):
            for f in inn.iterdir():
                f.unlink()
            self.rclone("backend", "copyid", self.remote, src["id"], str(inn) + "/")
            got = [f for f in inn.iterdir() if f.is_file()]
            if len(got) != 1:
                raise RuntimeError(f"download gave {len(got)} files")
            got[0].rename(dst)
        if dst.stat().st_size != int(src["size"]):
            raise RuntimeError(f"size {dst.stat().st_size} != {src['size']}")
        if src.get("md5"):
            out = subprocess.run(["md5sum", str(dst)], capture_output=True, text=True, check=True).stdout.split()[0]
            if out != src["md5"]:
                raise RuntimeError(f"md5 {out} != {src['md5']}")
        return dst

    def health_check(self, src, d):
        """Erase 3 minutes of this source with one process and watch the GPU. Pass: in P0-P2 at >= 50 % of its
        maximum clock during most of the erase stage. A clip without anything to erase leaves the health unknown."""
        hd = d / "health"
        hd.mkdir(exist_ok=True)
        clip = hd / ("clip" + src.suffix)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(src)], capture_output=True, text=True).stdout.strip() or 0)
        secs = int(self.cfg.get("health_clip_seconds", 180))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0, dur * 0.3):.1f}", "-t", str(secs), "-i",
                        str(src), "-c", "copy", str(clip)], check=True)
        samples, stop = [], threading.Event()
        log = hd / "clean-clear.log"

        def watch():
            while not stop.is_set():
                lines = log.read_text(encoding="utf-8", errors="replace")[-400:].splitlines() if log.exists() else []
                if lines and "[erase]" in lines[-1]:  # the erase stage is running: that is when the GPU must be busy
                    samples.append(dict(self.gpu))
                stop.wait(5)

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        code = self.run([self.cc, str(clip), "-o", str(hd), "--jobs", "1"], hd / "run.log")
        stop.set()
        t.join()
        fps = [float(m) for m in re.findall(r"\[erase\][^\n]*?([\d.]+) fps", log.read_text(errors="replace"))] \
            if log.exists() else []
        res = {"time": now(), "exit_code": code, "erase_samples": len(samples),
               "median_erase_fps": sorted(fps)[len(fps) // 2] if fps else None}
        good = [s for s in samples if s.get("pstate") in ("P0", "P1", "P2")
                and s.get("clock", 0) >= 0.5 * max(1, s.get("max_clock", 1))]
        if code != 0:
            res["status"] = "unknown"
            res["note"] = "the clip run failed; see health/run.log"
        elif len(samples) < 3:
            res["status"] = "unknown"
            res["note"] = "too few samples in the erase stage (nothing to erase in the clip?)"
        else:
            res["status"] = "ok" if len(good) >= 0.8 * len(samples) else "unhealthy"
            res["good_samples"] = len(good)
            res["example"] = samples[len(samples) // 2]
        (self.work / "health.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        shutil.rmtree(hd, ignore_errors=True)
        return res["status"]

    def process(self, job):
        a, b = job["asset_id"], job["batch_id"]
        d = self.job_dir
        d.mkdir(parents=True, exist_ok=True)
        self.set_state("downloading")
        src = self.download(job, d)
        if self.health()["status"] != "ok":
            self.set_state("health")
            if self.health_check(src, d) == "unhealthy":
                raise RuntimeError("GPU health check failed: the card does not reach its clocks under load")
        self.set_state("probing")
        lib = ["--logo-library", self.cfg["logo_library"]] if self.cfg.get("logo_library") else []
        self.run([self.cc, "--probe", str(src), "-o", str(d / "probe"), *lib], d / "probe.log")
        pj = d / "probe" / f"{a}.probe.json"
        probe = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else {"options": [], "warnings": ["probe failed"]}
        opts = list(probe.get("options", [])) + list(job.get("options", []))  # the job's options win (they come last)
        self.set_state("processing", options=opts, probe_warnings=probe.get("warnings", []))
        out = d / "out"
        code = self.run([self.cc, str(src), "-o", str(out), *opts], d / "run.log")
        if code != 0:
            raise RuntimeError(f"clean-clear exit code {code}; see run.log")
        result, srt = out / f"{a}_clean.mp4", out / f"{a}.srt"
        if not result.exists():
            self.set_state("no_output", note="nothing to erase (no subtitles and no logo found)")
            return "done"
        self.set_state("checking")
        qd = d / "qc"
        self.run([self.cc, "--check", str(src), str(result), "-o", str(qd)], d / "check.log")
        qc = json.loads((qd / "qc.json").read_text(encoding="utf-8"))
        cues = (qc.get("subtitles") or {}).get("cues", 0)
        wm = opts[len(opts) - 1 - opts[::-1].index("--watermark") + 1] if "--watermark" in opts else "auto"
        removed = (["subtitle"] if cues else []) + (["logo"] if wm != "off" and (
            probe.get("logo", {}).get("found") or wm not in ("auto", "")) else [])
        self.set_state("uploading", qc_pass=qc["pass"], removed_items=removed)
        n = job.get("attempt", 1)  # in the name: a late upload of an older attempt never replaces a newer one
        rv = f"{self.staging}/{b}/videos/{a}.attempt{n}.mp4"
        self.put(result, rv)
        if self.remote_md5(rv) != qc["result"]["md5"]:
            raise RuntimeError("uploaded video MD5 does not match")
        rs = None
        if cues:
            rs = f"{self.staging}/{b}/subtitles/{a}.attempt{n}.srt"
            self.put(srt, rs)
        qc_remote = f"{self.logdir}/qc/{a}/attempt_{n}"
        for f in [*qd.iterdir(), pj, out / "clean-clear.log", d / "run.log", d / "worker.log"]:
            if f.exists() and f.is_file():
                self.put(f, f"{qc_remote}/{f.name}")
        self.set_state("staged", video=rv, subtitles=rs, video_md5=qc["result"]["md5"], qc_dir=qc_remote,
                       subtitle_language=(qc.get("subtitles") or {}).get("language", "none"))
        shutil.rmtree(d / "in", ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
        return "done"

    # ---------- main loop ----------
    def next_job(self):
        jobs = sorted((self.work / "queue").glob("*.json"), key=lambda p: p.stat().st_mtime)
        return jobs[0] if jobs else None

    def loop(self):
        self.heartbeat(force=True)
        while not self.stop:
            if self.health().get("status") == "unhealthy" or self.degraded:
                self.heartbeat()
                time.sleep(30)
                continue
            jf = self.next_job()
            if not jf:
                self.heartbeat()
                time.sleep(20)
                continue
            self.current_job = json.loads(jf.read_text(encoding="utf-8"))
            try:
                self.process(self.current_job)
                jf.rename(self.work / "done" / jf.name)
            except Exception as e:
                if self.stop:
                    break
                self.set_state("failed", error=str(e), trace=traceback.format_exc()[-2000:])
                jf.rename(self.work / "failed" / jf.name)
            self.current = None
            self.heartbeat(force=True)

    def shutdown(self, *_):
        """supervisorctl stop: end the running clean-clear and its part processes (they run in their own sessions)"""
        self.stop = True
        if self.child and self.child.poll() is None:
            self.child.terminate()
        for p in Path("/proc").iterdir():
            if p.name.isdigit():
                try:
                    if b"--split-part" in (p / "cmdline").read_bytes():
                        os.kill(int(p.name), signal.SIGTERM)
                except OSError:
                    pass
        time.sleep(5)
        # a clean-clear killed by a signal cannot stop the MPS daemon it started; without the daemon gone the next
        # run's processes would fall back to the CPU
        subprocess.run(["sh", "-c", "echo quit | nvidia-cuda-mps-control"], capture_output=True)
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: worker.py /workspace/worker.json")
    w = Worker(sys.argv[1])
    signal.signal(signal.SIGTERM, w.shutdown)
    w.loop()
