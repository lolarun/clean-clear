---
name: video-tool
description: Run the film processing project: erase burned-in subtitles and corner logos with clean-clear on rented GPU servers, deploy and watch the servers (Luna), dispatch films, review every result and deliver batches to Google Drive (Sol). Use for any task about this project's films, servers, results or deliveries.
---

# Video processing tool / 视频处理工具

Agent instructions are in English. A Chinese translation for people is in `SKILL.zh-CN.md` next to this file; agents do not need to read it. Roles and the round are in `AGENTS.md`; `RUN_DECISIONS.md` overrides the defaults here.

## Parts

| Part | Where | What it does |
|---|---|---|
| `clean-clear` | each server, in the tool folder | erases subtitles and logos, `--probe` checks a source, `--check` compares a result with its source |
| worker | each server, supervisor service `video-worker` | takes jobs from `/workspace/queue/`, downloads, probes, processes, checks, uploads to staging, reports |
| `tools/ledger.py` | project folder (Sol's machine) | ledger, dispatch, review commands, batch delivery |

Tool version 0.3.6 or later is needed (worker and `--check`). Versions before 0.3.5 repeated the first frame of a source whose picture starts after its audio, dropped its last frame, wrote square pixels for anamorphic sources and started the picture at 0; 0.3.5 and later do it right, so never apply an extra start-offset remux to their results.

## Places

Google Drive, one rclone remote `gdrive:` rooted at the project's Drive folder (`root_folder_id` in the rclone config):

```
input/                                   source films, added over time (any names)
staging/<batch_id>/videos/<asset_id>.attempt<N>.mp4    results waiting for review (and subtitles/)
output/<batch_id>/                       the delivery: videos/, subtitles/, records_<batch_id>.csv, README.md
log/ledger.csv                           copy of the ledger
log/servers/<host>/heartbeat.json        every 5 minutes; jobs/<asset_id>.json per job
log/qc/<asset_id>/attempt_<N>/           qc.json, sheet_full.jpg, sheet_detail.jpg, probe and logs
```

Project folder (where Codex runs): `AGENTS.md`, `RUN_DECISIONS.md`, `ops.json` (settings), `assets.csv` (source file name or Drive file ID -> asset_id, batch_id, work_order, options; filled from work orders), `server-ssh-config` (one `Host` per server; the host name is the server id), `tools/ledger.py`, `state/` (ledger, batch files), `review/` (downloaded check material), the tool archive `video_tool_<version>_with_propainter.tar.gz`.

Server: tool in `/workspace/video_tool_<version>/`, `/workspace/worker.json`, `queue/ jobs/ done/ failed/`, `status.txt`, `heartbeat.json`, `health.json`, `worker.out.log`, `setup.log`.

## Luna: one server

### Deploy a new server

A host in `server-ssh-config` without a heartbeat is new.
1. `ssh -F server-ssh-config HOST 'nvidia-smi; df -h /workspace'`. Needed: NVIDIA GPU with 16 GB or more, driver for CUDA 12.8 or newer on RTX 50 cards, 200 GB or more free. If not, report and stop.
2. `python3 tools/ledger.py deploy HOST video_tool_<version>_with_propainter.tar.gz` copies the rclone config and the archive and starts `scripts/server_setup.sh` (system packages, install with PyTorch, the worker service). It takes 10-30 minutes.
3. Watch `ssh ... tail -n 20 /workspace/setup.log` every few minutes until `== HOST ready`. It must have printed `cuda available: True` and `CUDAExecutionProvider`; otherwise the driver is the problem: report.
4. Done when `ledger.py servers` shows a heartbeat for HOST. The worker runs a 3-minute GPU health check with its first film.

### Check a server

Read `ssh ... cat /workspace/status.txt` (and `tail -n 30 /workspace/worker.out.log` if something looks wrong). Report one line when all is well: health, current film and state, queue length, disk free.

| Sign | Meaning | Action |
|---|---|---|
| health `unhealthy`, or `degraded: True` | the GPU does not reach its clocks under load (host limit) | report "replace this machine"; do nothing else |
| heartbeat or status older than 15 min, `supervisorctl status video-worker` not RUNNING | worker stopped | `supervisorctl start video-worker`; report the last lines of `worker.out.log` |
| disk free below 30 GB | leftovers | delete `/workspace/jobs/*/attempt_*/in` and `out` of jobs in `done/` or `failed/` only |
| a job in `failed/` | see `jobs/<asset_id>/attempt_<N>/state.json` (`error`) and `run.log` | report the error; Sol decides about a retry |
| `clean-clear` alive but no new line in its `clean-clear.log` for 20 min | hang the tool did not catch | report; do not kill unless Sol says so |

Never start `clean-clear` by hand while the worker runs (under MPS a second process falls back to the CPU). Kill only by process id, never `pkill -f`. After killing a run by hand: `echo quit | nvidia-cuda-mps-control`.

## Sol: review every staged result

`ledger.py fetch ID` downloads `qc.json` and the sheets to `review/ID/attempt_N/`.

1. `qc.json` → `checks`: every item `ok`. frames (same count as the source), resolution, frame_rate, aspect, colour, codec (H.264 in MP4), bitrate (not below the source), duration, av_start (video start against the audio as in the source), audio, size, subtitles (valid SRT). Also read `probe_warnings` in the job state (`log/servers/<host>/jobs/ID.json`).
2. `sheet_detail.jpg`: subtitle band and logo band, source left, result right, at 8 spread points plus the darkest scenes. Look for: any trace of subtitles (strokes, outlines, shadows), any trace of the logo, blurred or smeared blocks, colour patches, damaged faces or bodies, damaged text that belongs to the film (signs, title, credits). `sheet_full.jpg`: the whole picture is intact and unshifted.
3. Decide:
   - all good: `approve ID --note ""` (a note goes into `qc_notes` of the delivery record; write one for anything the customer should know, e.g. a short soft area).
   - measured check failed but the result is acceptable for a stated reason: only with `--force` and that reason in `--note`.
   - fixable: `reject ID --note "..." --rerun "OPTS"`. Subtitles at the top: `--band 0,0.3`. Logo partly left: `--watermark <mask.png>` (a known mask). Bitrate below the source: `--crf 14`. Subtitles missed: `--min-score 0.5`.
   - not fixable by options, or the second attempt failed: `reject ID --note "..."` (needs_user) and tell the user.
   - `no_output` (no subtitles and no logo found): check the probe; if the film really has nothing to erase, ask the user how it should be delivered.

Approval moves the video (and the SRT) into `output/<batch_id>/` and verifies its MD5.

## Sol: batches

`ledger.py status` shows each batch's progress. When all films of a batch are delivered: `ledger.py batch BATCH_ID` writes `records_<batch_id>.csv` (the 16 fields of annex 2), `subtitles_<batch_id>.csv` if subtitles are delivered, and `README.md`, uploads them and verifies them. Then tell the user that the batch is complete, with its number, film count and total duration.

## Using the tool by hand (one-off tasks)

`clean-clear FILE -o OUT [options]`, `clean-clear --probe FILE -o DIR`, `clean-clear --check SOURCE RESULT -o DIR`. Options: `--subtitles off`, `--band 0,0.3`, `--watermark auto|off|mask.png`, `--crf 18`, `--jobs auto`. Not on a server whose worker is busy.

## Never

- Modify, move or overwrite source films or delivered results; approve without looking at the sheets.
- Use `--cut` for a delivery unless the work order allows removing footage (the contract requires the source's frames).
- Retime audio of a source with audio gaps; switch off checks to make a result pass.
- Put credentials into logs, deliveries or the repository.

## Known limits (tell the user honestly)

- Only burned-in subtitles and static corner logos. Captions elsewhere in the frame, QR codes, web addresses and scrolling text are not erased.
- Scene text on the subtitle line can be erased with the subtitles; fills over detailed backgrounds can blur; fast motion through the subtitle area can shimmer (ProPainter limit).
- No upscaling: a source below 720p stays below 720p.
- Speed on a healthy RTX 5090: a 90-minute 720p film about 1.3-2 hours with subtitles, 40-55 minutes for the logo only.
