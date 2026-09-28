# Clean Frame

Removes hardcoded (burned-in) subtitles from videos and extracts them as SRT. Point it at a folder and it batch-produces **subtitle-free videos** and **SRT subtitle files**.

- **Batch processing**: every video in the given directory is processed (mp4 / mkv / mov / avi / flv / ts / m4v / wmv / webm)
- **Subtitle extraction**: OCR recognition, merged into sentence-level SRT with frame-accurate timing
- **Automatic region detection**: the subtitle line is found from text positions across the whole video. Scene text such as signs, title calligraphy or costume patterns is **not** recognized or erased as subtitles
- **Stroke-only erasing**: by default only the pixels of the glyphs and their shadow are inpainted; the rest of the frame stays original, which is much sharper than erasing the whole text box
- **Per-sentence processing**: each subtitle uses one fixed mask, so the result does not flicker
- **Audio preserved**: the original audio stream is copied, not re-encoded
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
install.bat

# Linux
./install.sh
```

The install script creates a `.venv` virtual environment in this directory and installs the dependencies. It uses the Aliyun PyPI mirror by default (on Linux, override with `PIP_MIRROR=...`).
At the end it prints the available inference providers; `CUDAExecutionProvider` means the GPU is ready.

Python dependencies (`requirements.txt`, installed from PyPI):

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

```bash
# Windows
run.bat D:\videos -o D:\output

# Linux
./run.sh /data/videos -o /data/output
```

For each video the output directory contains:

| File | Description |
|---|---|
| `name_clean.mp4` | video with subtitles removed |
| `name.srt` | extracted subtitles (UTF-8) |
| `.cache/` | OCR cache. Reprocessing the same video skips OCR, so tuning erase settings is fast; use `--no-cache` to force OCR again |

### Common options

| Option | Default | Description |
|---|---|---|
| `-o, --out` | `output` | output directory |
| `-m, --model` | `lama` | inpainting model: `lama` (fast, default) or `propainter` (video model, slower; see below) |
| `--srt-only` | | extract subtitles only, do not erase |
| `--band` | `0.70,1.0` | horizontal band containing subtitles, as fractions or pixels. The default is the bottom 30% of the frame; use `0,0.3` for subtitles at the top |
| `--device` | `auto` | `auto` / `cuda` / `cpu` / `dml` (`dml` requires onnxruntime-directml) |
| `--crf` | `18` | output quality; lower is sharper and larger. Use `23` for a file size close to the source |
| `--encoder` | `auto` | `h264_nvenc` when an NVIDIA GPU is present, otherwise `libx264` |
| `--mask` | `glyph` | `glyph` erases strokes only (recommended); `box` erases the whole text box (try it for non-white subtitles) |

### Tuning options (rarely needed)

| Option | Default | Description |
|---|---|---|
| `--grow` | `8` | glyph mask dilation in pixels, covers the outline |
| `--shadow` | `3` | extra dilation towards the lower right, covers the shadow |
| `--dilate` | `10` | text box dilation in pixels (`box` mode) |
| `--pad-frames` | `1` | extra frames erased before/after each subtitle, for fade in/out |
| `--ocr-interval` | `10` | max frames to skip OCR while the subtitle band is unchanged. `1` runs OCR on every frame: most accurate, slowest |
| `--min-dur` | `0.4` | subtitles shorter than this many seconds are treated as noise |
| `--watermark` | `off` | also erase a static corner logo on every frame: `auto` samples ~150 frames and finds pixels that never change in the four corners; or pass a mask image of the video's size (white = watermark). The detected mask is saved to `.cache/name.watermark.png` for inspection |
| `--max-gap` | `5` | missed frames tolerated within one subtitle |
| `--min-score` | `0.6` | OCR confidence threshold |
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
6. **Watermark** (optional, `--watermark`): a static logo is found by sampling frames across the video, and the same erase step runs on it for every frame (before the subtitle step, chained on the same frame stream, so the video is still decoded and encoded once)
7. **Encoding**: ffmpeg encodes the video and copies the original audio

## Performance

Test clip: 1920×1080, 25 fps, 5 minutes, ~100 subtitles.

| GPU | OCR | Erase + encode | Total |
|---|---|---|---|
| NVIDIA A10 | 154 s | 363 s | ~8.6 min |
| RTX 3050 (estimated) | | | ~20–25 min |

## ProPainter model (optional)

`--model propainter` uses the [ProPainter](https://github.com/sczhou/ProPainter) video inpainting model. It fills the masked area with information from neighbouring frames and handles moving people or cameras better than LaMa, but it is **much slower** (the test clip takes about 30–45 minutes on an A10) and needs PyTorch.

> ⚠️ **ProPainter is licensed under the NTU S-Lab License 1.0, non-commercial use only.** Do not use this model in commercial projects.

Setup (in the same `.venv`):

```bash
# 1. PyTorch matching your CUDA version: https://pytorch.org/get-started/locally/
.venv/bin/python -m pip install -r requirements-propainter.txt

# 2. ProPainter code, cloned into ./ProPainter (or anywhere, see --propainter-dir)
git clone https://github.com/sczhou/ProPainter.git
.venv/bin/python -m pip install -r ProPainter/requirements.txt

# 3. Weights into ProPainter/weights/
#   https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth
#   https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth
#   https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth
```

Run:

```bash
run.bat D:\videos -o D:\output -m propainter
./run.sh /data/videos -o /data/output -m propainter --propainter-dir /opt/ProPainter
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

## Project layout

```
main.py            entry point
cli.py             command-line options, batch loop
pipeline.py        per-video flow: OCR -> SRT -> masks -> erase -> encode
subtitles.py       OCR, subtitle-line detection, segmentation, SRT
masks.py           glyph / box masks
watermark.py       static watermark detection
video.py           ffmpeg decoding and encoding
device.py          ONNX Runtime device selection
common.py          shared constants and helpers
backends/lama.py        LaMa backend (ONNX Runtime)
backends/propainter.py  ProPainter backend (PyTorch)
```

Both backends implement `erase(frames, frame_seg, masks)`, which takes the frame stream plus one mask per subtitle and yields the erased frames in order.

## Troubleshooting

**Only `CPUExecutionProvider` is listed; the GPU is not used**
- Check that the NVIDIA driver is installed and `nvidia-smi` works
- Make sure `onnxruntime-gpu` is installed, not `onnxruntime`. Having both installed causes conflicts; uninstall `onnxruntime` first
- RTX 50 series cards need a recent driver (CUDA 12.8+)

**Subtitles are missed, or other text is picked up**
- Narrow the subtitle region with `--band`
- For small subtitle text, lower `--min-height`

**Outlines of the text remain after erasing**
- Increase `--grow` (e.g. `10`) and `--shadow` (e.g. `5`)
- For non-white subtitles, use `--mask box`

**Model download fails (e.g. in mainland China)**
- The LaMa model falls back to hf-mirror.com automatically; you can also download `lama_fp32.onnx` manually into `models/`

## Third-party components and licenses

| Component | Purpose | License |
|---|---|---|
| [RapidOCR](https://github.com/RapidAI/RapidOCR) / PaddleOCR models | text detection and recognition | Apache-2.0 |
| [LaMa](https://github.com/advimman/lama) (ONNX export: Carve/LaMa-ONNX) | image inpainting | Apache-2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | inference | MIT |
| [FFmpeg](https://ffmpeg.org/) | encoding/decoding | LGPL / GPL |
| [ProPainter](https://github.com/sczhou/ProPainter) (optional) | video inpainting | NTU S-Lab License 1.0 (**non-commercial**) |
