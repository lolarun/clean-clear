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

For a long video start it detached, write to a log, and look at the progress lines (`[ocr] ...`, `[erase] ... ETA`):

```bash
cat > run.sh <<'EOS'
#!/bin/bash
date +%s > /data/out/start.epoch
/opt/clean-clear/.venv/bin/clean-clear /data/videos/film.mkv -o /data/out --propainter-dir /opt/ProPainter --jobs 4 > /data/out/run.log 2>&1
date +%s > /data/out/end.epoch
EOS
chmod +x run.sh && setsid nohup ./run.sh > /dev/null 2>&1 < /dev/null &
tail -f /data/out/run.log
```

## 5. Parallel runs and MPS

`--jobs N` splits a long video into 2N parts and runs N processes at once (one process leaves most of a big GPU idle). With `--jobs` greater than 1 on Linux as root, Clean Clear starts the NVIDIA MPS daemon itself for the run and stops it afterwards (`--mps auto`, the default; `--mps off` disables it; if MPS cannot start the run simply continues without it). If you already run MPS (`CUDA_MPS_PIPE_DIRECTORY` set), it is used as is. MPS made 4 processes about 14 % faster on an L20; see "Parallel runs and NVIDIA MPS" in the README.

## 6. Before releasing the server

Copy the `*_clean.mp4`, `*.srt` and `clean-clear.log` files out (`scp`, then compare checksums), and keep the `.cache/` folder if a customer may ask for a re-run with other settings.
