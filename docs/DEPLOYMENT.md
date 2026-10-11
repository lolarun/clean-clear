# Deployment on a GPU server (mainland China)

Recipe for a freshly rented Linux GPU server (tested: RTX 5090 32 GB pod and NVIDIA L20 46 GB, Ubuntu 24.04, Python 3.12, root). Servers are usually temporary: download results and logs before releasing one.

## 1. Use mainland China mirrors

The default PyPI index and `download.pytorch.org` are slow or blocked in China, and the CUDA wheels pulled in by PyTorch and onnxruntime-gpu add up to 2 GB or more. Always pass a local index:

| Mirror | PyPI index URL |
|---|---|
| Tencent Cloud | `https://mirrors.cloud.tencent.com/pypi/simple/` |
| Huawei Cloud | `https://repo.huaweicloud.com/repository/pypi/simple/` |
| Aliyun | `https://mirrors.aliyun.com/pypi/simple/` |
| Tsinghua | `https://pypi.tuna.tsinghua.edu.cn/simple/` |

Speed varies by server and by time. On the 5090 pod the Aliyun index delivered the nvidia-* wheels at ~0.2 MB/s (hours for 2 GB); after switching to Tencent Cloud the same downloads ran at ~7 MB/s. A short `curl -r 0-60000000 -w '%{speed_download}'` test of a big wheel from each mirror gave 2.5–4 MB/s everywhere, so it is only a hint: if a real install crawls, kill it and switch mirror.

Other things that do not work or need care there:
- `git clone` from GitHub may hang; `git archive HEAD | ssh server "tar -x -C DIR"` copies the code. `codeload.github.com/.../tar.gz/refs/heads/main` usually works for ProPainter.
- `download.pytorch.org` answers 403 to scripts; get PyTorch from the PyPI mirror (the Linux wheels there are CUDA 12.x builds), or download the wheel files on another machine and `pip install` them from a folder with `--find-links`.
- Hugging Face: use `https://hf-mirror.com` for the LaMa model.
- On a fresh image `apt-get install ffmpeg` fails with "Unable to locate package" until `apt-get update` has run.

## 2. Install

```bash
MIRROR=https://mirrors.cloud.tencent.com/pypi/simple/
apt-get update && apt-get install -y ffmpeg python3-venv

# code (from the PC: git archive HEAD | ssh -p PORT root@HOST "mkdir -p /opt/clean-clear && tar -x -C /opt/clean-clear")
cd /opt/clean-clear
python3 -m venv .venv
.venv/bin/pip install -U pip -i $MIRROR
# PyTorch first (needed for ProPainter). RTX 50xx (Blackwell) needs a CUDA 12.8+ build: torch 2.7 or newer
.venv/bin/pip install torch torchvision -i $MIRROR
.venv/bin/pip install -e ".[propainter]" -i $MIRROR
# check
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/bin/python -c "import onnxruntime as o; o.preload_dlls(); print(o.get_available_providers())"   # CUDAExecutionProvider = ready
```

`scripts/install.sh` does the venv and the package install; `PIP_MIRROR=URL ./scripts/install.sh` selects the mirror (default Aliyun).

Run installs detached so an SSH disconnect does not abort them, and read the log:

```bash
setsid nohup ./setup.sh > setup.log 2>&1 < /dev/null &     # script with LF line endings
```

Never `pkill -f` / `pgrep -f` with a pattern that also occurs in your own ssh command line: it kills the shell you are in. Use `pgrep -x NAME` or kill by PID.

## 3. ProPainter and models

```bash
# ProPainter source
curl -L https://codeload.github.com/sczhou/ProPainter/tar.gz/refs/heads/main | tar -xz && mv ProPainter-main ProPainter
# weights go into ProPainter/weights/: ProPainter.pth, recurrent_flow_completion.pth, raft-things.pth
#   https://github.com/sczhou/ProPainter/releases/download/v0.1.0/<file>
```

If the server cannot reach the GitHub release files, download them on your PC and `scp` them up. The weights are pickled `.pth` files (they can execute code), so only take them from the official release. On a persistent disk they can be reused between servers.

The LaMa model (`models/lama_fp32.onnx`) and the RapidOCR models are downloaded on first run; for LaMa use the hf-mirror URL from the README if Hugging Face is unreachable. ProPainter is licensed for non-commercial use only.

## 4. Run

The inputs are video files or directories (any path), `-o` is the output directory:

```bash
.venv/bin/clean-clear /data/videos/film.mkv -o /data/out --propainter-dir /opt/ProPainter
```

### A batch of films

Films differ (no subtitles, bilingual subtitles, a logo in another corner, irregular time stamps), so check them first and give each its own options:

```bash
# 1. probe: a few minutes per film, nothing is erased; writes NAME.probe.json and probe-batch.txt
.venv/bin/clean-clear --probe /data/videos -o /data/probe --logo-library /data/masks
# 2. read the WARN comments in /data/probe/probe-batch.txt, adjust options if needed
# 3. run the batch file (one film after another, 4 processes per film, finished films are skipped)
.venv/bin/clean-clear --list /data/probe/probe-batch.txt -o /data/out --propainter-dir /opt/ProPainter
```

Keep the logo masks of finished jobs (`.cache/NAME.watermark.png` once checked) in the library folder: they are suggested when a new film's logo mask looks incomplete (dark films).

### Detached runs

Start long runs detached, write to a log, and read the progress lines (`[ocr] ...`, `[erase] ... ETA`, `part k/16 done`):

```bash
cat > run.sh <<'EOS'
#!/bin/bash
date +%s > /data/out/start.epoch
/opt/clean-clear/.venv/bin/clean-clear --list /data/probe/probe-batch.txt -o /data/out --propainter-dir /opt/ProPainter > /data/out/run.log 2>&1
echo "EXIT=$?" >> /data/out/run.log
date +%s > /data/out/end.epoch
EOS
chmod +x run.sh && setsid nohup ./run.sh > /dev/null 2>&1 < /dev/null &
tail -f /data/out/run.log
```

The batch file is read once when the run starts. To change it for the remaining films, wait until the run has just started the next film, stop it (the parent `clean-clear --list` first, then the part processes) and start it again: finished films are skipped, and only the minute or two of the film that had just started is lost.

## 5. Parallel runs, MPS and hung parts

- `--jobs auto` (the default) starts up to 4 processes per film: one per 7 GB of GPU memory and per 4 usable CPUs (the container's CPU quota counts, not the host's core count). A film is split at keyframes into 4 parts per process; the last part (the dense credits) starts first, and parts start 20 s apart.
- With more than one process on Linux as root, the NVIDIA MPS daemon is started for the run and stopped afterwards (`--mps auto`; `--mps off` disables it). While the MPS server holds the GPU, **a CUDA process started outside it gets no GPU** and silently runs on the CPU (seen: 0.3 frames/s instead of 15). Never start a second `clean-clear` next to a running batch; put the extra work into the batch file instead.
- A part process sometimes hangs right after loading its models (3 of ~35 parts on an RTX 5090 under MPS; it stays alive, idle, for hours). The parent kills and restarts a part that has printed no progress line 5 min after its start or no output for 15 min, at most twice. `py-spy` (if installed) saves the hung stack, but in containers without ptrace permission it cannot read the process.
- Measured on an RTX 5090 (32 GB, 16-core quota): a 110-min 1080p film 2 h 9 min; 84-97 min 720p films 78-89 min with subtitles, 37-54 min with the logo only.

## 6. Checking the results

- **Picture and audio in sync.** Compare the video and audio stream durations of every result (`ffprobe -show_entries stream=codec_type,duration`); more than ~0.3 s apart needs a closer look. For a precise check, compare the times of a few shot cuts (`select='gt(scene,0.35)',showinfo` with `-copyts`) in the source and in the result at the start, middle and end of the film.
- **Never retime audio that has gaps.** Some sources hold less sound than their time line (2829: 10.8 s of gaps); `--probe` warns about it. `atempo` closes the gaps and moves the sound against the picture. Fix the video instead, or re-run the film.
- Logo and subtitles: look at before/after crops at 6-8 points spread over the film, including dark scenes (a logo mask can be incomplete there).

## 7. Before releasing the server

- Copy the `*_clean.mp4`, `*.srt` and `clean-clear.log` files out (`scp`, then compare sizes or checksums), and keep the `.cache/` folder if a re-run with other settings may be needed.
- Trial jobs for a customer without a signed contract: leave only the code on the server (`src/`, `pyproject.toml`, `scripts/`), no project documents (README, MEMO, PLANNING, `docs/`), and the results stay on the server until the customer decides.
