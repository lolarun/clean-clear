# Clean Clear

Removes hardcoded (burned-in) subtitles and static corner watermarks from videos and extracts the subtitles as SRT. Run it in a folder of videos and it batch-produces **clean videos** and **SRT subtitle files**.

- **Batch processing**: every video in the given directory is processed (mp4 / mkv / mov / avi / flv / ts / m4v / wmv / webm)
- **Subtitle extraction**: OCR recognition, merged into sentence-level SRT with frame-accurate timing
- **Automatic region detection**: the subtitle line is found from text positions across the whole video. Scene text such as signs, title calligraphy or costume patterns is **not** recognized or erased as subtitles
- **Stroke-only erasing**: by default only the pixels of the glyphs and their shadow are inpainted; the rest of the frame stays original, which is much sharper than erasing the whole text box
- **Per-sentence processing**: each subtitle uses one fixed mask, so the result does not flicker
- **Audio preserved**: the original audio stream is copied, not re-encoded
- **Watermark removal**: a static corner logo is detected automatically and erased on every frame
- **Cross-platform**: Windows / Linux with NVIDIA GPU acceleration (CUDA); CPU-only also works (very slow)

## Requirements

| Item | Requirement |
|---|---|
| OS | Windows 10/11 or Linux (tested on Ubuntu 22.04) |
| Python | 3.10 – 3.13 |
| GPU | NVIDIA with 4 GB+ VRAM; driver with CUDA 12 support (RTX 50 series needs a CUDA 12.8+ driver) |
| ffmpeg | `ffmpeg` and `ffprobe`, either on PATH or in an `ffmpeg/` folder in the repository root |

Getting ffmpeg:
- **Windows**: download a release build from https://www.gyan.dev/ffmpeg/builds/ and copy `ffmpeg.exe` and `ffprobe.exe` from `bin` into `ffmpeg\`
- **Linux**: `sudo apt install ffmpeg`

## Installation

```bash
# Windows
scripts\install.bat

# Linux
./scripts/install.sh
```

The install script creates a `.venv` virtual environment in the repository root and installs the package in editable mode (`pip install -e .`), which provides the `clean-clear` command in `.venv\Scripts` (Windows) or `.venv/bin` (Linux). On Linux it installs from the official PyPI (override with `PIP_MIRROR=...`), and also installs PyTorch when the folder contains the ProPainter weights; the Windows script uses the Aliyun mirror.

On a server in mainland China use a local PyPI mirror (Tencent Cloud, Huawei Cloud, ...); see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the full server recipe.
At the end it prints the available inference providers; `CUDAExecutionProvider` means the GPU is ready.

Python dependencies (declared in `pyproject.toml`, installed from PyPI):

| Package | Source |
|---|---|
| `onnxruntime-gpu[cuda,cudnn]` (includes the CUDA/cuDNN runtime) | https://pypi.org/project/onnxruntime-gpu/ |
| `rapidocr` | https://pypi.org/project/rapidocr/ |
| `opencv-python-headless` | https://pypi.org/project/opencv-python-headless/ |
| `numpy` | https://pypi.org/project/numpy/ |

### Models

Model files are not part of this repository.

| Model | Location | How to get it |
|---|---|---|
| LaMa inpainting (`lama_fp32.onnx`, ~200 MB) | `models/lama_fp32.onnx` | Downloaded automatically on first run. Manual download: https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx (mainland China mirror: https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx) |
| OCR (PP-OCRv6 detection + recognition, ~30 MB) | inside the `rapidocr` package | Downloaded automatically by RapidOCR on first run (from ModelScope) |

For offline machines, run the tool once on a connected machine, then copy `models/` and the `rapidocr/models/` folder from the virtual environment's `site-packages`.

## Usage

Go to the folder that contains the videos and run the command without arguments; it uses ProPainter, detects the watermark automatically and writes the results next to the videos:

```bash
# Windows
cd D:\videos
D:\clean-clear\.venv\Scripts\clean-clear

# Linux
cd /data/videos
/opt/clean-clear/.venv/bin/clean-clear

# explicit folders
clean-clear /data/videos -o /data/output
```

`python -m clean_clear` is equivalent. Output files of earlier runs (`*_clean.mp4`, `*_refined.mp4`) and temporary files are never taken as input, so reruns in the same folder are safe. A video whose `name_clean.mp4` and `name.srt` already exist is skipped, so an interrupted batch continues where it stopped (`--force` processes it again). The video is written to a hidden temporary file and renamed only when it is complete, so a `_clean.mp4` is never half-written. Two inputs with the same name but different extensions (`a.mp4`, `a.mkv`) would write the same output; the second one is reported as failed and must be renamed.

The console prints a progress line every 10 seconds for each stage (`[ocr] 1537/131326 1.2% 51.1 fps elapsed 0:00:30 ETA 0:42:18`), a `[k/N]` header per video, and the same log is appended to `clean-clear.log` in the output directory.

For each video the output directory contains:

| File | Description |
|---|---|
| `name_clean.mp4` | video with subtitles (and the watermark) removed |
| `name.srt` | extracted subtitles (UTF-8) |
| `.cache/` | OCR and watermark-mask cache. Reprocessing the same video skips OCR, so tuning erase settings is fast; use `--no-cache` to force it again. The cache is tied to the file (size and modification time) and to `--min-score`, `--min-height` and `--ocr-interval`, so changing those options or replacing the video runs OCR again |
| `clean-clear.log` | log of all runs, with timestamps |

### Common options

| Option | Default | Description |
|---|---|---|
| `INPUT ...` | current directory | video files or directories |
| `-o, --out` | current directory | output directory |
| `-m, --model` | `propainter` | inpainting model: `propainter` (video model, best quality, default; see below) or `lama` (about 5-7x faster, lower quality) |
| `--srt-only` | | extract subtitles only, do not erase |
| `--subtitles` | `on` | `off`: the video has no hardcoded subtitles; skip OCR, SRT and subtitle erasing and only erase the watermark (avoids erasing the credits of a film that has no subtitles). Without a watermark either, nothing is written |
| `--list FILE` | | batch file: one video per line, `"path" --option value ...`; see [Batch file](#batch-file) |
| `--probe` | | look at each INPUT video before a batch, no erasing (a few minutes per film): hardcoded subtitles (none, one or two lines, bottom or top), logo corner and whether its mask looks complete, black letterbox bars, irregular video time stamps and audio gaps. Writes `NAME.probe.json` and `probe-batch.txt`, a `--list` file with the suggested options per video and the warnings as comments |
| `--logo-library DIR` | | with `--probe`: folder of logo masks from earlier jobs (PNG, white = logo, the frame size of the videos they fit). When a detected logo mask looks incomplete (dark films), a library mask that contains it is suggested |
| `--band` | `0.70,1.0` | horizontal band containing subtitles, as fractions or pixels. The default is the bottom 30% of the frame; use `0,0.3` for subtitles at the top |
| `--device` | `auto` | `auto` / `cuda` / `cpu` / `dml` (`dml` requires onnxruntime-directml) |
| `--crf` | `18` | output quality; lower is sharper and larger. Use `23` for a file size close to the source |
| `--encoder` | `auto` | `h264_nvenc` when an NVIDIA GPU is present, otherwise `libx264` |
| `--mask` | `glyph` | `glyph` erases strokes only (recommended); `box` erases the whole text box (try it for non-white subtitles) |

### Batch file

For a set of videos that need different options (some without subtitles, a logo in another corner, ...), list them in a text file and run `clean-clear --list videos.txt -o /data/out`:

```
# a line starting with DEFAULT sets options for every video below it
DEFAULT --crf 18

"/data/videos/film one.mp4"
"/data/videos/film two.mp4"  --subtitles off
"/data/videos/film three.mp4" --band 0,0.3
```

The options are the ones of the command line (the call's own options apply to every video, a line overrides them). Videos run one after another, each with `--jobs auto` inside; a video that fails does not stop the others; one whose `NAME_clean.mp4` already exists in the output directory is skipped, so the same file can be run again after an interruption. At the end a summary is logged and written to `batch-status.txt`.

### Checking a batch first

```bash
clean-clear --probe /data/videos -o /data/probe --logo-library /data/masks
# read /data/probe/probe-batch.txt (WARN comments), adjust if needed, then
clean-clear --list /data/probe/probe-batch.txt -o /data/out
```

Rules behind the suggestions: subtitles count as present when text sits on one consistent line (or two, bilingual) in at least 12 % of ~200 sampled frames; otherwise `--subtitles off`. A logo mask is flagged as incomplete when a looser detection finds 1.8x as many pixels (the dark film 2771 gave 2060 of ~6000). Time stamps are flagged when the video has 0.1 % more or fewer frames than its duration x nominal rate, and audio gaps when the audio frames hold 0.5 s less sound than the audio duration (2829: 10.8 s).

### Tuning options (rarely needed)

| Option | Default | Description |
|---|---|---|
| `--grow` | `8` | glyph mask dilation in pixels, covers the outline |
| `--shadow` | `3` | extra dilation towards the lower right, covers the shadow |
| `--dilate` | `10` | text box dilation in pixels (`box` mode) |
| `--pad-frames` | `1` | extra frames erased before/after each subtitle, for fade in/out |
| `--ocr-interval` | `10` | max frames to skip OCR while the subtitle band is unchanged. `1` runs OCR on every frame: most accurate, slowest |
| `--min-dur` | `0.4` | subtitles shorter than this many seconds are treated as noise |
| `--watermark` | `auto` | erase a static corner logo on every frame: `auto` samples ~150 frames and finds pixels that never change in the four corners; `off` leaves it; or pass a mask image of the video's size (white = watermark). The detected mask is saved to `.cache/name.watermark.png` for inspection |
| `--ocr-fixed-shape` | `auto` | feed OCR recognition one fixed input shape; works around multi-second stalls per new input shape observed on an RTX 5090 with onnxruntime-gpu 1.23 (`auto` = on for compute capability 12+ with onnxruntime 1.23 or newer; with 1.22 it only made OCR 3x slower) |
| `--ffmpeg-threads` | `auto` | threads per ffmpeg decoder/encoder; `auto` = half of the process's CPU share (the CPU quota divided by the part processes), `0` = ffmpeg's own default (sized from the host's core count, which overran a container quota) |
| `--max-gap` | `5` | missed frames tolerated within one subtitle |
| `--bridge-sec` | `1.0` | OCR lost a subtitle's text for up to this long and then reads the very same text in the same place: one subtitle. Gap frames that show no subtitle are not erased (`0` = only `--max-gap`) |
| `--keep-ranges` | none | parts of the film left as they are: no OCR, no erasing, no cues in the SRT (opening title and credits, closing credits), e.g. `0-95,1:28:00-end`; per video in a `--list` file |
| `--qc-subtitles` | off | `clean-clear --qc-subtitles SOURCE RESULT [-o DIR] [--srt FILE]`: look for subtitles still on screen (calibration, five frames per SRT cue, a blind scan every second, logos); writes `report.md`, `findings.csv`, `evidence/` and contact sheets; exit code 1 = leftovers, 2 = a person must decide |
| `--qc-workers` | 0 | with `--qc-subtitles`: OCR processes (0 = a quarter of the CPUs, 2-6; 1 = one process). 1304 (17 min): 6 minutes with one process, 2.5 minutes with six |
| `--compose` | auto | `auto`: write the result on the source's own YUV so the picture outside the erased area stays as it was (needs 8-bit 4:2:0, limited range, a known matrix); `off`: decode and encode every pixel through RGB |
| `--min-score` | `0.6` | OCR confidence threshold |
| `--extend-sec` | `3` | when OCR loses a subtitle for part of its time, keep erasing up to this many seconds before/after it while its glyphs are still visible; `0` = off |
| `--verify` | `on` | after erasing, OCR the result again and erase any subtitle that is still readable (adds one cheap OCR pass; a second decode/encode only if something is found) |
| `--wm-guard` | `25` | repair a watermark fill whose mean brightness differs from its surroundings by more than this many grey levels (dark-scene flashes); `0` = off |
| `--stabilize` | `0.6` | blend every inpainted area with the previous frame by up to this weight where its surroundings are static; reduces the frame-to-frame shimmer of the fill that shows at 2x playback. `0` = off |
| `--jobs` | `auto` | process a long video in this many concurrent processes (split into up to 4x this many parts at keyframes, merged afterwards; OCR runs in parallel too). `auto` = up to 4, limited to one process per 7 GB of GPU memory and per 4 usable CPUs (1 with LaMa or without a GPU). On an L20 the GPU was saturated with 4 processes: the whole film took 2 h instead of 3 h+; on an RTX 5090 (16-core quota) 2 h 9 min. Videos shorter than 4 min per process run normally. See [Parallel runs and NVIDIA MPS](#parallel-runs-and-nvidia-mps) Each process loads its own models: about 6.5 GB of GPU memory per process was measured for 720p films on an RTX 5090, a 1080p subtitle band needs up to ~13 GB at `--pp-chunk 120`; a warning is logged when the processes may not fit. |
| `--keep-parts` | | with `--jobs`, keep the per-part work directory |
| `--force` | | process videos again even if their `_clean.mp4` and `.srt` already exist |
| `--refine-of ORIGINAL` | | INPUT is an already cleaned video of ORIGINAL: erase any subtitle still readable in it and fill logo pixels the first pass missed, in one decode and one encode (writes `NAME_refined.mp4`). About 30 min for a 90-minute film, instead of a full rerun |
| `--mps` | `auto` | with `--jobs` on Linux, start the NVIDIA MPS daemon for the run and stop it afterwards (needs root; `auto` skips it silently when unavailable, `off` never starts it) |
| `--find-ads` | | find advertisement pages spliced into a film and list them in `NAME.ads.txt`; nothing is changed. A film with letterbox bars is scanned for full-frame pages (picture in the bars) that are at least 10 s long; one whose picture contains a web address (`www.`, `.com`, ...) is `confirmed`, others are only `suspect` and never cut. Found the 45 s casino page of the test film to the frame |
| `--cut-ads` | | like `--find-ads`, then cut the confirmed ads out (`NAME_cut.mp4`, `NAME_cut.srt`) |
| `--cut START-END` | | remove a time range from the INPUT video, frame-exact (`--cut 0:40:01.667-0:40:46.958`, repeatable). Only the keyframe windows around the cut are re-encoded, the rest is copied; audio and SRT follow. Use it on the `*_clean.mp4` result (an mkv whose first frame is not at time 0 is refused) |
| `--min-height` | `0.015` | minimum text height as a fraction of frame height |

## How it works

1. **OCR**: only the subtitle band is decoded; RapidOCR (PP-OCRv6) detects and recognizes text. Frames whose subtitle band has not changed reuse the previous result, so only about a quarter of the frames actually go through OCR
2. **Region detection**: text boxes from the whole video are analysed to find the usual line position and text height; text that does not match is discarded
3. **Sentence segmentation**: consecutive frames with the same or similar text are merged into one subtitle with start/end times and written to SRT
4. **Masks**: for each subtitle, pixels that are white in more than half of its frames are treated as strokes, then dilated to cover the outline and shadow. The mask is fixed within a subtitle, so there is no flicker
5. **Erasing**:
   - `lama`: the subtitle line is cut into strips and packed into one 512×512 image so LaMa inpaints the whole line in a single call
   - `propainter`: frames are streamed in chunks; only the frame ranges with subtitles and only the subtitle band are inpainted using neighbouring frames

   In both cases only pixels inside the mask are replaced
6. **Watermark** (`--watermark`, on by default): a static logo is found by sampling frames across the video, and the same erase step runs on it for every frame (before the subtitle step, chained on the same frame stream, so the video is still decoded and encoded once)
7. **Encoding**: ffmpeg encodes the video and copies the original audio. Colours are converted with the source's own matrix (BT.709 / BT.601, as tagged; untagged videos from 720 lines up are treated as BT.709) and the output is tagged accordingly. A variable-frame-rate source is resampled to its average frame rate so that video, audio and subtitles stay in sync; a warning is logged if the output video is not as long as the source

## Performance

Test clip: 1920×1080, 25 fps, 5 minutes, ~100 subtitles.

| GPU | OCR | Erase + encode | Total |
|---|---|---|---|
| NVIDIA A10 | 154 s | 363 s | ~8.6 min |
| RTX 3050 (estimated) | | | ~20–25 min |

### Parallel runs and NVIDIA MPS

On a big machine one process leaves most of it idle, so long videos run faster with `--jobs`. Measured on an NVIDIA L20 (16 vCPU, 46 GB) with a 90-minute 720p film: 4 processes finished in 2 h 8 min, a single process took over 3 h. More than 4 processes did not help, because the GPU is then the limit.

Several processes on one GPU take turns by default: their kernels never run at the same time, and the small crops Clean Clear works on leave much of the GPU unused (utilisation shows ~100 %, power stays at ~65 %). NVIDIA's Multi-Process Service (MPS) lets them run concurrently. With more than one process on Linux as root, Clean Clear starts the MPS daemon for the run and stops it afterwards (`--mps auto`, the default; `--mps off` never starts it; an MPS daemon that is already running, `CUDA_MPS_PIPE_DIRECTORY` set, is used as is).

While the MPS server holds the GPU, a CUDA process started outside it gets no GPU and silently runs on the CPU. Do not start a second `clean-clear` next to a running parallel run; put the work into the batch file instead.

A part process occasionally hangs right after loading its models (seen on an RTX 5090 under MPS); the parent restarts it, see below.

Measured on the same L20 with 90-second clips (other work was running on the machine at the time, so treat the numbers as indicative):

| Setup | Time | Note |
|---|---|---|
| 4 processes, no MPS | 491 s | |
| 4 processes, MPS | 420 s | about 14 % faster |
| 8 processes, MPS | slower per clip than 4 | no gain, 30 GB GPU memory |
| 4 processes, MPS, `--pp-chunk 240` | 451 s | larger chunks were slower |

A part process that hangs (seen 3 times in ~35 parts on the 5090: alive but idle right after loading the models) is detected by the parent, which kills it and restarts that part: after 5 min without a first progress line, or 15 min without any output; at most twice per part. The Python stack of the hung process is saved to `hang_dumps/` if `py-spy` is installed (`pip install py-spy`).

Recommendation: `--jobs 4` with MPS on a 16-core machine with one GPU; this is what `--jobs auto` (the default) picks there.

RTX 5090 (32 GB, 16-core quota), 110-minute 1080p film, 4 processes + MPS, 8 parts: 2 h 9 min. A single process uses ~31 % of the GPU while erasing (182 W of the card's power budget, 6.5 GB, ~6 cores). The last part, the dense credits, took 73 min against 46-56 for the others, so the other 3 processes idled for 29 min at the end (18 % of the process time); parts are therefore now 4 per process (16 for 4 processes) and the last one is started first. If MPS cannot be started (e.g. no root in a container), `--jobs 4` alone still helps.

## ProPainter model (optional)

`--model propainter` uses the [ProPainter](https://github.com/sczhou/ProPainter) video inpainting model. It fills the masked area with information from neighbouring frames and handles moving people or cameras better than LaMa, but it is **much slower** (the test clip takes about 30–45 minutes on an A10) and needs PyTorch.

> ⚠️ **ProPainter is licensed under the NTU S-Lab License 1.0, non-commercial use only.** Do not use this model in commercial projects.

Setup (in the same `.venv`):

```bash
# 1. PyTorch matching your CUDA version (RTX 50 series: a cu128 or newer build): https://pytorch.org/get-started/locally/
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e ".[propainter]"

# 2. ProPainter code, cloned into ./ProPainter (or anywhere, see --propainter-dir)
git clone https://github.com/sczhou/ProPainter.git

# 3. Weights into ProPainter/weights/
#   https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth
#   https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth
#   https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth
```

Run (ProPainter is the default model):

```bash
clean-clear /data/videos -o /data/output --propainter-dir /opt/ProPainter
```

| Option | Default | Description |
|---|---|---|
| `--propainter-dir` | `$PROPAINTER_DIR` or `./ProPainter` | ProPainter checkout containing `weights/` |
| `--pp-chunk` | `120` | frames per chunk. GPU memory grows with it: ~13 GB at 120 frames for a 1080p subtitle band. Use `80` or lower on 8 GB cards |
| `--pp-raft-iter` | `12` | optical flow iterations; fewer is faster but less accurate |
| `--pp-ctx` | `10` | context frames added on each side of a chunk for flow estimation; lower it for less redundant work between chunks |
| `--pp-pad` | `8` | extra frames inpainted before/after each run of subtitle frames |
| `--pp-margin` | `80` | horizontal margin (px) kept around the subtitle when cropping columns; only this cropped region is processed instead of the full frame width |

The OCR cache is shared between models, so you can switch between `lama` and `propainter` on the same output directory without running OCR again.

## Tests

```bash
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m pytest
```

The tests need `ffmpeg` but no GPU and no models: they encode small synthetic videos and use fake OCR and inpainting models to check decoding/encoding (colours, frame counts, variable frame rate, encoder failures), masks, segmentation, caching, the backends' streaming logic, `--jobs` merging and the full `process()` flow.

## Project layout

```
src/clean_clear/
  cli.py             command-line options, batch loop
  pipeline.py        per-video flow: OCR -> SRT -> masks -> watermark -> erase -> encode
  subtitles.py       OCR, subtitle-line detection, segmentation, SRT
  masks.py           glyph / box masks (stored as crops, MaskSet)
  watermark.py       static watermark detection
  stabilize.py       temporal smoothing of filled areas
  rewrite.py         re-encode only the changed windows of a finished video
  refine.py          --refine-of
  parallel.py        --jobs: split, run parts, merge
  video.py           ffmpeg decoding and encoding (colour, variable frame rate), cut detection
  device.py          ONNX Runtime device selection, GPU capability and memory
  common.py          version, paths, logging, progress
  backends/lama.py        LaMa backend (ONNX Runtime)
  backends/propainter.py  ProPainter backend (PyTorch)
tests/               pytest suite (needs ffmpeg, no GPU)
scripts/             install.sh, install.bat
docs/spec/           functional and technical design
pyproject.toml       dependencies and the clean-clear command
```

Both backends implement `erase(frames, frame_seg, masks, cuts)`, which takes the frame stream plus one mask per subtitle and yields the erased frames in order.
