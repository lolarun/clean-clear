#!/usr/bin/env python3
"""Controller (Sol) tool: the ledger of all films, dispatch to the GPU servers, review, batch delivery.

The ledger (state/ledger.csv, a copy in Drive log/ledger.csv) has one row per source file and is written only by
this tool. Servers report through Drive (log/servers/<server>/heartbeat.json and jobs/<asset_id>.json).

  python3 tools/ledger.py scan              register new source files from Drive input/ (asset ids from assets.csv)
  python3 tools/ledger.py servers           servers of server-ssh-config and their last heartbeat
  python3 tools/ledger.py sync              take over the servers' job states; flag jobs of silent servers as lost
  python3 tools/ledger.py assign            give ready films to healthy servers (queue_depth per server)
  python3 tools/ledger.py status            counts, and everything that needs attention
  python3 tools/ledger.py fetch ID          download the check material of ID into review/ID/ (qc.json, sheets, logs)
  python3 tools/ledger.py approve ID --note TEXT [--force]       staged -> delivery folder, state delivered
  python3 tools/ledger.py reject ID --note TEXT [--rerun OPTS]   drop the staged result; rerun or needs_user
  python3 tools/ledger.py retry ID [--options OPTS] [--force]    failed / lost / no_output -> ready again
  python3 tools/ledger.py done ID --note TEXT  a film handled outside this ledger (e.g. the old workflow): never dispatch it
  python3 tools/ledger.py batch BATCH_ID    all delivered: write and upload records CSV, subtitles CSV, README
  python3 tools/ledger.py deploy HOST ARCHIVE   copy the tool and rclone.conf to HOST and start its setup

Only the Python standard library, rclone and ssh are used. Settings: ops.json next to the tools folder.
"""
import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state"
LEDGER = STATE / "ledger.csv"
FIELDS = ["asset_id", "batch_id", "work_order", "source_id", "source_name", "size", "md5", "state", "server", "attempt",
          "assigned_at", "updated_at", "options", "qc_pass", "removed_items", "subtitle_language", "qc_dir", "video",
          "subtitles", "video_md5", "review_note", "error"]
ACTIVE = ("assigned", "processing")
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".flv", ".ts", ".m4v", ".wmv", ".webm")
RECORD_FIELDS = ["asset_id", "relative_path", "source_file_name", "duration_seconds", "frame_count",
                 "source_frame_count", "resolution", "frame_rate", "video_codec", "video_bitrate_kbps",
                 "file_size_bytes", "removed_items", "subtitle_language", "needs_processing", "qc_status", "qc_notes"]
DEFAULTS = {"remote": "gdrive:", "input": "input", "output": "output", "staging": "staging", "log": "log",
            "ssh_config": "server-ssh-config", "work": "/workspace", "queue_depth": 2, "heartbeat_timeout_minutes": 60,
            "max_attempts": 2, "deliver_subtitles": True, "csv_bom": False,
            "rclone_conf": "~/.config/rclone/rclone.conf"}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def cfg():
    p = ROOT / "ops.json"
    return {**DEFAULTS, **(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {})}


C = cfg()


# ---------- external commands ----------
def rclone(*args, check=True, input=None):
    r = subprocess.run(["rclone", *args], capture_output=True, text=True, input=input)
    if check and r.returncode != 0:
        sys.exit(f"rclone {' '.join(args[:2])} failed: {r.stderr.strip()[-400:]}")
    return r.stdout if r.returncode == 0 else None


def rpath(*parts):
    return C["remote"] + "/".join(p.strip("/") for p in parts if p)


def rjson(*parts):
    out = rclone("cat", rpath(*parts), check=False)
    try:
        return json.loads(out) if out else None
    except json.JSONDecodeError:
        return None


def rmd5(*parts):
    out = rclone("lsjson", "--hash", "--files-only", rpath(*parts), check=False)
    items = json.loads(out) if out else []
    return items[0].get("Hashes", {}).get("md5") if items else None


def ssh(host, command, input=None, timeout=60):
    base = ["ssh", "-F", str(ROOT / C["ssh_config"]), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command]
    return subprocess.run(base, capture_output=True, text=True, input=input, timeout=timeout)


def hosts():
    p = ROOT / C["ssh_config"]
    out = []
    for line in p.read_text(encoding="utf-8").splitlines() if p.exists() else []:
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower() == "host":
            out += [h for h in parts[1:] if not any(ch in h for ch in "*?!")]
    return out


# ---------- ledger ----------
class Lock:
    def __enter__(self):
        STATE.mkdir(exist_ok=True)
        self.p = STATE / ".lock"
        for _ in range(60):
            try:
                self.fd = os.open(self.p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() - self.p.stat().st_mtime > 600:  # a crashed run left it behind
                    self.p.unlink(missing_ok=True)
                time.sleep(1)
        sys.exit("ledger is locked by another run (state/.lock)")

    def __exit__(self, *a):
        os.close(self.fd)
        self.p.unlink(missing_ok=True)


def load():
    if not LEDGER.exists():
        return []
    with open(LEDGER, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def save(rows):
    STATE.mkdir(exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    tmp.replace(LEDGER)
    rclone("copyto", str(LEDGER), rpath(C["log"], "ledger.csv"), check=False)


def find(rows, key):
    hit = [r for r in rows if key in (r["asset_id"], r["source_id"])]
    if len(hit) != 1:
        sys.exit(f"{key}: {len(hit)} matching rows")
    return hit[0]


def touch(r, state, **kv):
    r.update({k: ("" if v is None else str(v)) for k, v in kv.items()}, state=state, updated_at=now())


def heartbeat(host):
    """The server's last heartbeat with its age in minutes (from the epoch seconds it carries; clocks are UTC-synced)"""
    hb = rjson(C["log"], "servers", host, "heartbeat.json")
    if hb:
        e = hb.get("epoch")
        hb["age_minutes"] = round((time.time() - float(e)) / 60, 1) if e else None
    return hb


def healthy(hb):
    return (hb and hb.get("age_minutes") is not None and hb["age_minutes"] <= C["heartbeat_timeout_minutes"]
            and (hb.get("health") or {}).get("status") != "unhealthy" and not hb.get("gpu_degraded"))


# ---------- commands ----------
def cmd_scan(a):
    mapping = {}
    mp = ROOT / "assets.csv"
    if mp.exists():
        with open(mp, encoding="utf-8-sig", newline="") as fh:
            for m in csv.DictReader(fh):
                mapping[m.get("source_file_id") or m["source_file_name"]] = m
    files = json.loads(rclone("lsjson", "-R", "--hash", "--files-only", rpath(C["input"])))
    with Lock():
        rows = load()
        by_id = {r["source_id"]: r for r in rows}
        md5s = {r["md5"]: r for r in rows if r["md5"]}
        new = 0
        for f in files:
            if not f["Name"].lower().endswith(VIDEO_EXTS):
                continue
            r = by_id.get(f["ID"])
            if r is None:
                md5 = f.get("Hashes", {}).get("md5", "")
                r = {"source_id": f["ID"], "source_name": f["Name"], "size": f["Size"], "md5": md5, "attempt": "0",
                     "state": "unmapped", "updated_at": now()}
                if md5 and md5 in md5s:
                    touch(r, "duplicate", error=f"same MD5 as {md5s[md5]['source_name']}")
                rows.append(r)
                by_id[f["ID"]] = r
                new += 1
            m = mapping.get(f["ID"]) or mapping.get(f["Name"])
            if r["state"] == "unmapped" and m:
                touch(r, "ready", asset_id=m["asset_id"], batch_id=m["batch_id"],
                      work_order=m.get("work_order", ""), options=m.get("options", ""))
        ids = [r["asset_id"] for r in rows if r["asset_id"]]
        dup = {x for x in ids if ids.count(x) > 1}
        save(rows)
    print(f"scan: {len(files)} files in {C['input']}, {new} new; "
          f"{sum(r['state'] == 'unmapped' for r in rows)} without asset id (add them to assets.csv)")
    if dup:
        print(f"WARNING: asset ids used for more than one source: {sorted(dup)}")


def cmd_servers(a):
    for h in hosts():
        hb = heartbeat(h)
        if not hb:
            print(f"{h:14} no heartbeat (not deployed, or the worker never started)")
            continue
        g = hb.get("gpu") or {}
        print(f"{h:14} {hb.get('age_minutes')} min ago  health {(hb.get('health') or {}).get('status')}  "
              f"degraded {hb.get('gpu_degraded')}  gpu {g.get('pstate')} {g.get('clock')}/{g.get('max_clock')} MHz "
              f"{g.get('power_w')} W  disk {hb.get('disk_free_gb')} GB  current {hb.get('current')}  "
              f"queue {len(hb.get('queue') or [])}  {hb.get('version')}")


def cmd_sync(a):
    with Lock():
        rows = load()
        beats = {h: heartbeat(h) for h in {r["server"] for r in rows if r["state"] in ACTIVE and r["server"]}}
        changed = 0
        for r in rows:
            if r["state"] not in ACTIVE:
                continue
            js = rjson(C["log"], "servers", r["server"], "jobs", f"{r['asset_id']}.json")
            if js and str(js.get("attempt")) == r["attempt"]:
                st = js["state"]
                new = {"staged": "staged", "failed": "failed", "no_output": "no_output"}.get(st, "processing")
                if new != r["state"] or new == "processing":
                    touch(r, new, qc_pass=js.get("qc_pass"), removed_items=";".join(js.get("removed_items") or []),
                          subtitle_language=js.get("subtitle_language"), qc_dir=js.get("qc_dir"),
                          video=js.get("video"), subtitles=js.get("subtitles"), video_md5=js.get("video_md5"),
                          error=js.get("error") or js.get("note"))
                    changed += 1
            hb = beats.get(r["server"])
            if r["state"] in ACTIVE and not (hb and hb.get("age_minutes") is not None
                                             and hb["age_minutes"] <= C["heartbeat_timeout_minutes"]):
                touch(r, "lost", error=f"no heartbeat from {r['server']} for over {C['heartbeat_timeout_minutes']} min")
                changed += 1
        save(rows)
    print(f"sync: {changed} row(s) updated")


def cmd_assign(a):
    with Lock():
        rows = load()
        ready = sorted((r for r in rows if r["state"] == "ready"), key=lambda r: (r["batch_id"], r["asset_id"]))
        given = 0
        for h in hosts():
            hb = heartbeat(h)
            if not healthy(hb):
                continue
            free = int(C["queue_depth"]) - sum(r["server"] == h and r["state"] in ACTIVE for r in rows)
            while free > 0 and ready:
                r = ready.pop(0)
                attempt = int(r["attempt"] or 0) + 1
                job = {"asset_id": r["asset_id"], "batch_id": r["batch_id"], "attempt": attempt,
                       "source": {"id": r["source_id"], "name": r["source_name"], "size": int(r["size"]),
                                  "md5": r["md5"]},
                       "options": shlex.split(r["options"] or ""), "assigned_at": now()}
                q = f"{C['work']}/queue"
                if a.dry_run:
                    print(f"would give {r['asset_id']} to {h}")
                else:
                    res = ssh(h, f"mkdir -p {q} && cat > {q}/{r['asset_id']}.json.tmp && "
                                 f"mv {q}/{r['asset_id']}.json.tmp {q}/{r['asset_id']}.json", input=json.dumps(job))
                    if res.returncode != 0:
                        print(f"{h}: could not place {r['asset_id']}: {res.stderr.strip()[-200:]}")
                        ready.insert(0, r)
                        break
                    touch(r, "assigned", server=h, attempt=attempt, assigned_at=job["assigned_at"], error="")
                given += 1
                free -= 1
        if not a.dry_run:
            save(rows)
    print(f"assign: {given} film(s) given; {len(ready)} ready film(s) still waiting")


def cmd_status(a):
    rows = load()
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    print("states: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for title, states in (("to review (staged)", ("staged", "no_output")), ("failed", ("failed",)),
                          ("lost (server silent)", ("lost",)), ("needs the user", ("needs_user", "duplicate"))):
        hit = [r for r in rows if r["state"] in states]
        if hit:
            print(f"{title}:")
            for r in hit:
                print(f"  {r['asset_id'] or '-':12} {r['source_name'][:40]:40} {r['server']:10} attempt {r['attempt']} "
                      f"qc_pass {r['qc_pass'] or '-'}  {r['error'][:120]}")
    for b in sorted({r["batch_id"] for r in rows if r["batch_id"]}):
        br = [r for r in rows if r["batch_id"] == b]
        print(f"batch {b}: {sum(r['state'] == 'delivered' for r in br)}/{len(br)} delivered")


def cmd_fetch(a):
    r = find(load(), a.id)
    if not r["qc_dir"]:
        sys.exit(f"{a.id}: no check material yet (state {r['state']})")
    dst = ROOT / "review" / r["asset_id"] / f"attempt_{r['attempt']}"
    rclone("copy", rpath(r["qc_dir"]), str(dst))
    print(dst)


def cmd_approve(a):
    with Lock():
        rows = load()
        r = find(rows, a.id)
        if r["state"] != "staged":
            sys.exit(f"{a.id} is {r['state']}, not staged")
        if r["qc_pass"] != "True" and not a.force:
            sys.exit(f"{a.id}: the measured checks did not all pass; approve with --force and say why in --note")
        b = r["batch_id"]
        dst = rpath(C["output"], b, "videos", f"{r['asset_id']}.mp4")
        rclone("moveto", rpath(r["video"]), dst)
        if rmd5(C["output"], b, "videos", f"{r['asset_id']}.mp4") != r["video_md5"]:
            sys.exit(f"{a.id}: MD5 in the delivery folder does not match the checked file; nothing marked")
        if r["subtitles"] and C["deliver_subtitles"]:
            rclone("moveto", rpath(r["subtitles"]), rpath(C["output"], b, "subtitles", f"{r['asset_id']}.srt"))
        touch(r, "delivered", review_note=a.note, video=f"{C['output']}/{b}/videos/{r['asset_id']}.mp4")
        save(rows)
    print(f"{a.id}: delivered")


def cmd_reject(a):
    with Lock():
        rows = load()
        r = find(rows, a.id)
        if r["state"] not in ("staged", "no_output"):
            sys.exit(f"{a.id} is {r['state']}")
        for p in (r["video"], r["subtitles"]):
            if p:
                rclone("deletefile", rpath(p), check=False)
        if a.rerun is not None and int(r["attempt"]) < int(C["max_attempts"]):
            touch(r, "ready", options=(r["options"] + " " + a.rerun).strip(), review_note=a.note, video="",
                  subtitles="", qc_pass="")
        else:
            touch(r, "needs_user", review_note=a.note)
        save(rows)
    print(f"{a.id}: {r['state']}")


def cmd_retry(a):
    with Lock():
        rows = load()
        r = find(rows, a.id)
        if r["state"] not in ("failed", "lost", "no_output", "needs_user"):
            sys.exit(f"{a.id} is {r['state']}")
        if int(r["attempt"] or 0) >= int(C["max_attempts"]) and not a.force:
            touch(r, "needs_user")
            save(rows)
            sys.exit(f"{a.id}: {r['attempt']} attempts used; marked needs_user (use --force if the user agreed)")
        if r["state"] == "lost" and r["server"]:  # best effort: take the job off the silent server's queue
            try:
                ssh(r["server"], f"rm -f {C['work']}/queue/{r['asset_id']}.json", timeout=40)
            except subprocess.TimeoutExpired:
                pass
        opts = (r["options"] + " " + a.options).strip() if a.options else r["options"]
        touch(r, "ready", options=opts, error="")
        save(rows)
    print(f"{a.id}: ready (attempt {int(r['attempt']) + 1} next)")


def cmd_done(a):
    with Lock():
        rows = load()
        r = find(rows, a.id)
        if r["state"] not in ("unmapped", "ready", "failed", "lost", "needs_user", "duplicate"):
            sys.exit(f"{a.id} is {r['state']}; only films that are not running can be marked")
        touch(r, "done_elsewhere", review_note=a.note)
        save(rows)
    print(f"{a.id}: done_elsewhere (will not be dispatched; not part of ledger.py batch)")


def _fmt_rate(x):
    return f"{round(float(x), 3):g}"


def cmd_batch(a):
    rows = [r for r in load() if r["batch_id"] == a.batch]
    if not rows:
        sys.exit(f"no films in batch {a.batch}")
    missing = [r["asset_id"] for r in rows if r["state"] != "delivered"]
    if missing:
        sys.exit(f"batch {a.batch}: not delivered yet: {missing}")
    out = STATE / "batches" / a.batch
    out.mkdir(parents=True, exist_ok=True)
    recs, subs, tot_dur, tot_size = [], [], 0.0, 0
    for r in sorted(rows, key=lambda r: r["asset_id"]):
        qc = rjson(r["qc_dir"], "qc.json")
        if not qc:
            sys.exit(f"{r['asset_id']}: qc.json not found in {r['qc_dir']}")
        s, v = qc["source"], qc["result"]
        removed = r["removed_items"]
        recs.append({
            "asset_id": r["asset_id"], "relative_path": f"videos/{r['asset_id']}.mp4",
            "source_file_name": r["source_name"], "duration_seconds": f"{v['video_duration']:.3f}",
            "frame_count": v["frames"], "source_frame_count": s["frames"], "resolution": f"{v['width']}x{v['height']}",
            "frame_rate": _fmt_rate(v["frame_rate"]), "video_codec": v["codec"],
            "video_bitrate_kbps": v["video_bitrate_kbps"], "file_size_bytes": v["size_bytes"],
            "removed_items": removed, "subtitle_language": r["subtitle_language"] or "none",
            "needs_processing": "yes" if removed else "no", "qc_status": "pass", "qc_notes": r["review_note"]})
        if r["subtitles"] and C["deliver_subtitles"]:
            subs.append({"asset_id": r["asset_id"], "relative_path": f"subtitles/{r['asset_id']}.srt",
                         "subtitle_language": r["subtitle_language"], "cue_count": (qc.get("subtitles") or {}).get("cues", 0)})
        tot_dur += v["video_duration"]
        tot_size += v["size_bytes"]
    enc = "utf-8-sig" if C["csv_bom"] else "utf-8"
    files = []
    for name, fields, data in ((f"records_{a.batch}.csv", RECORD_FIELDS, recs),
                               (f"subtitles_{a.batch}.csv", ["asset_id", "relative_path", "subtitle_language", "cue_count"], subs)):
        if not data:
            continue
        with open(out / name, "w", encoding=enc, newline="") as fh:
            w = csv.DictWriter(fh, fields)
            w.writeheader()
            w.writerows(data)
        files.append(name)
    notes = [f"- {x['asset_id']}: {x['qc_notes']}" for x in recs if x["qc_notes"]]
    wo = sorted({r["work_order"] for r in rows if r["work_order"]})
    (out / "README.md").write_text(
        f"# Batch {a.batch}\n\n- Work order: {', '.join(wo) or '-'}\n- Delivery date: {time.strftime('%Y-%m-%d')}\n"
        f"- Films: {len(recs)}\n- Total duration: {tot_dur / 3600:.2f} h ({tot_dur:.0f} s)\n"
        f"- Total size: {tot_size / 1e9:.2f} GB\n- Format: H.264 video in MP4, original audio"
        f"{', SRT subtitles (UTF-8)' if subs else ''}\n\n## Files\n\n"
        + "".join(f"- `{f}`\n" for f in files) + "- `videos/<asset_id>.mp4`\n"
        + ("- `subtitles/<asset_id>.srt`\n" if subs else "")
        + f"\n## Fields\n\n`records_{a.batch}.csv` has the fields of annex 2, section 5, in that order: "
        + ", ".join(RECORD_FIELDS) + ".\n\n## Notes\n\n" + ("\n".join(notes) if notes else "None.") + "\n",
        encoding="utf-8")
    files.append("README.md")
    for f in files:
        rclone("copyto", str(out / f), rpath(C["output"], a.batch, f))
        local = hashlib.md5((out / f).read_bytes()).hexdigest()
        if rmd5(C["output"], a.batch, f) != local:
            sys.exit(f"{f}: uploaded copy does not match")
    print(f"batch {a.batch}: {len(recs)} films, {', '.join(files)} uploaded to {C['output']}/{a.batch}/ and verified")


def cmd_deploy(a):
    archive = Path(a.archive)
    tool = archive.name.removesuffix(".tar.gz").removesuffix("_with_propainter")
    w = C["work"]
    conf = Path(os.path.expanduser(C["rclone_conf"]))
    r = ssh(a.host, "mkdir -p ~/.config/rclone && cat > ~/.config/rclone/rclone.conf && chmod 600 ~/.config/rclone/rclone.conf",
            input=conf.read_text(encoding="utf-8"))
    if r.returncode != 0:
        sys.exit(f"{a.host}: {r.stderr.strip()}")
    subprocess.run(["scp", "-F", str(ROOT / C["ssh_config"]), str(archive), f"{a.host}:{w}/"], check=True)
    r = ssh(a.host, f"cd {w} && tar xzf {archive.name} && cd {tool} && "
                    f"setsid nohup bash scripts/server_setup.sh {a.host} {w} > {w}/setup.log 2>&1 < /dev/null & echo started")
    print(r.stdout.strip() or r.stderr.strip())
    print(f"follow it with: ssh -F {C['ssh_config']} {a.host} tail -n 20 {w}/setup.log")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "servers", "sync", "status"):
        sp.add_parser(name)
    p = sp.add_parser("assign")
    p.add_argument("--dry-run", action="store_true")
    p = sp.add_parser("fetch")
    p.add_argument("id")
    p = sp.add_parser("approve")
    p.add_argument("id")
    p.add_argument("--note", default="")
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("reject")
    p.add_argument("id")
    p.add_argument("--note", required=True)
    p.add_argument("--rerun", metavar="OPTS", help="extra clean-clear options for the next attempt ('' = same options)")
    p = sp.add_parser("retry")
    p.add_argument("id")
    p.add_argument("--options", default="")
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("done")
    p.add_argument("id")
    p.add_argument("--note", required=True)
    p = sp.add_parser("batch")
    p.add_argument("batch")
    p = sp.add_parser("deploy")
    p.add_argument("host")
    p.add_argument("archive")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
