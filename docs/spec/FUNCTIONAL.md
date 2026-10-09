# Clean Clear — Functional Design

Version 0.3.x. Companion document: [TECHNICAL.md](TECHNICAL.md).

## 1. Purpose

Clean Clear is a command-line tool that takes a folder of videos with **hardcoded (burned-in) subtitles** and produces, for every video:

1. a **subtitle-free video** in which the subtitles have been inpainted with plausible background, and
2. an **SRT file** with the recognized subtitle text and frame-accurate timing.

Optionally it also removes a **static corner watermark** (a station or site logo) from every frame.

It was built for a customer job: batch-process long videos on their own Windows and Linux machines (NVIDIA RTX 3050 and RTX 5090), with output that the customer accepts by visual inspection.

## 2. Users and environment

| Item | Description |
|---|---|
| User | An operator who runs the tool on a folder and reviews the output. No programming knowledge is required. |
| OS | Windows 10/11, Linux (tested on Ubuntu 22.04) |
| GPU | NVIDIA, 4 GB+ VRAM for OCR and LaMa; 8 GB+ for ProPainter (with a reduced chunk size); RTX 50 series needs a CUDA 12.8+ driver and PyTorch build (not yet tested) |
| Software | Python 3.10–3.13, ffmpeg/ffprobe, ONNX Runtime GPU, PyTorch (ProPainter only) |
| Network | Only needed once, to download models and packages. Processing is fully offline. |

## 3. Scope

### In scope

- Batch processing of every video in one or more folders (or individual files).
- Subtitle recognition (Chinese and other text supported by PP-OCR) and export to SRT.
- Subtitle erasing with two selectable inpainting models: **LaMa** (fast, single-image) and **ProPainter** (slower, uses neighbouring frames, better on moving shots).
- Optional static watermark removal (`--watermark`), detected automatically or supplied as a mask image.
- Audio is preserved bit-exact (stream copy).
- Restart-friendly caching of the expensive OCR and watermark-detection results.

### Out of scope

- Soft subtitles (separate subtitle tracks) — they can simply be dropped with ffmpeg.
- Translation, subtitle styling, re-burning subtitles.
- Moving, animated or semi-transparent watermarks (see §8).
- Subtitles at arbitrary positions in the same video (one subtitle line position per video is assumed).
- A graphical user interface.
- Commercial use of the ProPainter model (see §9).

## 4. Inputs and outputs

### Input

`clean-clear [<path> ...] [-o OUT]`, where each path is a video file or a directory; without arguments the current directory is processed and the results are written to the current directory. Directories are scanned (non-recursively) for `.mp4 .mkv .mov .avi .flv .ts .m4v .wmv .webm`, sorted by name. Our own results (`*_clean`, `*_refined`) and temporary files (hidden files, `*.tmp`, `*.fix`) are skipped, so input and output may be the same folder. Missing paths are skipped with a log line. Inputs that differ only in extension (`a.mp4`, `a.mkv`) would overwrite each other's results: the second one is reported as failed.

A video whose `<name>_clean.mp4` and `<name>.srt` already exist is skipped (`--force` reprocesses it), so a long batch can be restarted after an interruption. The video is written under a temporary name and renamed when complete.

### Output (in `OUT`, default: the current directory)

| File | Description |
|---|---|
| `<name>_clean.mp4` | H.264 video, subtitles (and optionally the watermark) removed, original audio copied |
| `<name>.srt` | UTF-8 SRT, one entry per recognized subtitle sentence |
| `.cache/<name>.<y0>-<y1>.<key>.ocr.json` | OCR results for the subtitle band (reused on reruns; the key covers the file's size and modification time and the OCR options) |
| `.cache/<name>.<file key>.watermark.png` | Detected watermark mask (white = watermark); all black = none found. Also useful for visual inspection |

The process exit code is `0` if every file succeeded and `1` if at least one failed. A failure in one file is logged (`!! failed <file>: <reason>`) and does not stop the batch.

## 5. Functional requirements

### 5.1 Subtitle extraction

| ID | Requirement |
|---|---|
| FR-1 | Only a horizontal band of the frame is analysed (default: bottom 30%, `--band`). Fractions or pixel rows are accepted. |
| FR-2 | The subtitle line position and text height are inferred from the whole video, so that scene text (signs, calligraphy, credits, logos, costume patterns) is **not** treated as a subtitle. |
| FR-3 | Frames with the same or very similar text are merged into one subtitle with start and end time. OCR jitter is tolerated (`--max-gap`); two long, clearly different sentences stay separate even if similar. |
| FR-4 | Subtitles shorter than `--min-dur` (default 0.4 s) are treated as noise. |
| FR-5 | Two-line subtitles are supported and joined with a line break. |
| FR-6 | The SRT is written even in `--srt-only` mode, without loading any inpainting model. |
| FR-7 | Reruns on the same video and band reuse the OCR cache; `--no-cache` forces recognition again. The cache is invalidated when the file changes (size or modification time) or when `--min-score`, `--min-height` or `--ocr-interval` change. |

### 5.2 Subtitle erasing

| ID | Requirement |
|---|---|
| FR-8 | Each subtitle uses **one fixed mask** for its whole duration, so the filled area does not flicker. |
| FR-9 | By default only the **glyph strokes** (plus outline and a shadow offset) are erased (`--mask glyph`); the rest of the frame keeps its original pixels. `--mask box` erases whole text boxes. Subtitles without white text fall back to boxes. |
| FR-10 | Frames before and after each subtitle (`--pad-frames`, default 1) are erased too, to cover fade in/out. Back-to-back subtitles get the union of both masks on shared frames. |
| FR-11 | Pixels outside the mask are never changed. |
| FR-12 | The output has the same resolution, frame rate and frame count as the input. A variable-frame-rate input is converted to a constant rate (its average), so the output keeps the input's duration and stays in sync with the audio; a warning is logged if the output video's duration differs from the source's by more than 0.5 s. |
| FR-13 | With ProPainter, inpainting never mixes frames from different shots (shot-cut aware). |

### 5.3 Watermark removal

| ID | Requirement |
|---|---|
| FR-14 | `--watermark off` leaves the watermark untouched; the default is `auto`. |
| FR-15 | `--watermark auto` finds a static, opaque logo in the four corners by sampling frames across the video, and removes it on every frame. The mask is saved to `.cache` for inspection. |
| FR-16 | `--watermark <mask.png>` uses a user-supplied mask instead (must have the video's exact size; white = watermark). This is the manual override when detection is wrong. |
| FR-17 | If nothing is found, or the detected area is unreasonably large (> 5% of the frame), the video is processed without watermark removal and a log line says so. |

### 5.4 Encoding

| ID | Requirement |
|---|---|
| FR-18 | NVENC H.264 is used when available, otherwise libx264 (`--encoder` overrides). Quality is set with `--crf` (default 18). |
| FR-19 | The audio stream is copied without re-encoding; video is `yuv420p` with `faststart`. |
| FR-20 | Colours are preserved: video is converted with the source's own YUV matrix and range (BT.709 / BT.601 as tagged; untagged video BT.709 from 720 lines, BT.601 below) and the output is tagged accordingly. |
| FR-21 | A `_clean.mp4` is either complete or absent: it is written under a temporary name and renamed when finished. A video whose `_clean.mp4` and `.srt` already exist is skipped unless `--force` is given, so an interrupted batch can be restarted. |

## 6. Command-line interface

```
clean-clear [INPUT ...] [-o OUT] [-m propainter|lama] [options]
```

| Option | Default | Meaning |
|---|---|---|
| `INPUT ...` | current directory | Video files or directories |
| `-o, --out` | current directory | Output directory |
| `-m, --model` | `propainter` | Inpainting model: `propainter` or `lama` |
| `--band` | `0.70,1.0` | Subtitle band (fractions or pixels), e.g. `0,0.3` for top subtitles |
| `--device` | `auto` | `auto`, `cuda`, `dml`, `cpu` for OCR and LaMa (ProPainter uses CUDA if available) |
| `--srt-only` | off | Extract subtitles only |
| `--no-cache` | off | Ignore OCR and watermark caches |
| `--force` | off | Process videos again even if their `_clean.mp4` and `.srt` already exist |
| `--watermark` | `auto` | `auto`, `off`, or a mask image path |
| `--ocr-fixed-shape` | `auto` | Feed OCR recognition a fixed input shape (workaround for multi-second stalls per new shape seen on an RTX 5090); `auto` = on for compute capability 12+ |
| `--ocr-interval` | `10` | Max frames to skip OCR while the band is unchanged (`1` = every frame) |
| `--encoder`, `--crf` | `auto`, `18` | Encoder and quality |
| `--mask` | `glyph` | `glyph` or `box` |
| `--dilate`, `--grow`, `--shadow` | `10`, `8`, `3` | Box dilation, glyph dilation, shadow offset (px) |
| `--pad-frames` | `1` | Extra frames erased around each subtitle |
| `--max-gap`, `--min-dur` | `5`, `0.4` | Merging tolerance (frames), minimum duration (s) |
| `--min-score`, `--min-height` | `0.6`, `0.015` | OCR confidence, minimum text height (fraction of frame height) |
| `--propainter-dir` | `$PROPAINTER_DIR` or `./ProPainter` | ProPainter checkout with `weights/` |
| `--pp-chunk` | `120` | Frames per ProPainter chunk; lower on small GPUs (60–80 for 8 GB) |
| `--pp-raft-iter` | `12` | Optical-flow iterations (lower = faster, less accurate) |
| `--pp-ctx`, `--pp-pad` | `10`, `8` | Context frames per chunk side, extra frames around subtitle runs |
| `--pp-margin` | `80` | Horizontal margin (px) around the subtitle when cropping |
| `--pp-ref-stride` | `10` | Interval between ProPainter's global reference frames |
| `--extend-sec` | `3` | Keep erasing up to this long before/after a subtitle while its glyphs are still visible (`0` = off) |
| `--verify` | `on` | OCR the result again and erase any subtitle still readable |
| `--wm-guard` | `25` | Repair watermark fills brighter than their dark surroundings by more than this (`0` = off) |
| `--stabilize` | `0.6` | Blend filled areas with the motion-compensated previous frame against shimmer (`0` = off) |
| `--jobs` | `1` | Split a long video into 2× this many parts and process them in this many processes; each needs its own GPU memory (~13 GB with ProPainter at chunk 120, a warning is logged if they will not fit) |
| `--keep-parts` | off | With `--jobs`, keep the per-part work directory |
| `--refine-of ORIGINAL` | | Post-fix an already cleaned INPUT of ORIGINAL (residual subtitles, missed logo pixels), writing `NAME_refined.mp4` |

`scripts/install.bat` / `scripts/install.sh` create the virtual environment and install the package in editable mode, which provides the `clean-clear` command (`python -m clean_clear` is equivalent).

**Progress and logs.** Every 10 s each stage prints `[stage] done/total  pct  fps  elapsed  ETA` (the erase ETA is weighted, because subtitle frames cost about 15x more than pass-through frames); each video gets a `[k/N]` header and a summary with its elapsed time, the batch ends with the total time and the list of failed files. The same lines, with timestamps, are appended to `clean-clear.log` in the output directory. Library deprecation warnings are suppressed.

## 7. Model choice

| | LaMa | ProPainter |
|---|---|---|
| Method | Single-image inpainting, each frame independent | Video inpainting using optical flow and neighbouring frames |
| Speed (A10) | About 5–7× faster in the erase step (1080p test clip, before the speed optimizations: 341 s vs 30–40 min) | About 2.3 h per hour of 720p video including OCR (§10) |
| Static shots | Can flicker slightly frame to frame | Stable, and can copy real background from other frames of the shot |
| Moving shots | Can smear | Best available quality |
| Never-exposed background | Hallucinated, can show colour casts (seen on dark walls) | Hallucinated, but smoother and temporally stable |
| Licence | Apache-2.0 | NTU S-Lab 1.0, **non-commercial only** |
| GPU memory | ~2 GB | ~13 GB at chunk 120 (1080p band); lower chunk for 8 GB cards |

The customer decision so far is to use **ProPainter** for delivery.

## 8. Quality characteristics and known limitations

- **Where results are best:** any background that is visible somewhere else in the same shot (moving people, camera moves) is restored almost perfectly. Static-camera shots with a background that stays covered for the whole shot can only be filled plausibly, not exactly.
- **Blur:** mask-only inpainting keeps unmasked pixels sharp, but the filled strokes can look softer than the original.
- **Shot cuts:** rapid montages make very short shots (down to one frame); these have little temporal context and are the most likely place for visible artefacts.
- **Fast motion and motion blur** reduce optical-flow accuracy and can leave smears.
- **Watermark detection** assumes an opaque, static logo in a corner. Semi-transparent, moving, fading or position-changing logos need a manual mask or are not supported.
- **One subtitle position per video.** Subtitles that move between top and bottom are only handled if `--band` covers both, and scene text in that band may be mistaken for subtitles.
- **Subtitle colour:** the tight glyph mask relies on white text; other colours fall back to text boxes (blurrier fill).
- **OCR accuracy** is not guaranteed; SRT text may contain recognition errors and should be proofread.
- Detection results and quality on 1080p, on the 3050 and on the 5090 have **not been measured** (see §10).

## 9. Legal and compliance notes

- ProPainter (code and weights) is released under the NTU S-Lab License 1.0 for **non-commercial use only**. The customer must confirm that this is acceptable for their use; otherwise the LaMa backend (Apache-2.0) must be used. The planned quality improvements that matter most for static shots (clean plate, inpaint-once-and-propagate; TECHNICAL §13) can be built on LaMa as well. Alternative models and their licences are surveyed in TECHNICAL §14.
- Removing a third-party watermark or subtitles from a video may infringe rights of the content owner. The tool does not check ownership; this is the operator's responsibility.
- Model and package licences are listed in the README.

## 10. Performance expectations

Measured on an NVIDIA A10 (24 GB), 1280×714 25 fps, ProPainter, current code:

| Stage | 87.6 min video (131,326 frames) |
|---|---|
| Subtitle OCR (36% of frames actually recognized) | 1683 s |
| Glyph masks | 171 s |
| Shot-cut detection | 130 s |
| Erase + encode (subtitles, ~13 fps overall) | 9999 s |
| **Total** | **~3.3 h (about 2.3 h per hour of video)** |

Watermark removal on a 5-minute 1280×714 clip: detection 13 s, plus about 291 s of additional erase time (≈ 39 ms per frame, ≈ 1 h per hour of video on an A10).

Planning estimate for 200 hours (300 videos) of **1080p** video on one RTX 5090, derived from the numbers above (not measured on that hardware or resolution):

| Scope | Estimate |
|---|---|
| Extract + erase subtitles + remove watermark | ≈ 420 h (range 350–500 h), about 18 days non-stop |
| Extract + erase subtitles only | ≈ 320 h, about 13 days |

Running two processes on the 5090 and splitting the batch across the customer's machines both reduce elapsed time.

## 11. Acceptance criteria

1. For each input video a `_clean.mp4` and an `.srt` are produced; resolution, frame rate, frame count, duration, colours and audio are unchanged (variable-frame-rate inputs: constant average rate, same duration).
2. At a set of spot-checked timestamps (static dialogue shots, moving shots, back-to-back subtitles, shot cuts; the fixed check points are listed in TECHNICAL §9) no subtitle remnants, dark blobs or flicker are visible at normal playback speed.
3. Running OCR on the output (`--srt-only --ocr-interval 1`) finds no subtitles (residual-text check).
4. With `--watermark auto`, the logo is invisible in the output and the frame content around it is unchanged.
5. Subtitle text and timing in the SRT match what is shown in the source video, apart from occasional OCR character errors.
6. A batch with a corrupt or unsupported file completes for the other files and returns a non-zero exit code; no partial `_clean.mp4` is left behind, and a rerun skips the finished videos.
7. `pytest` passes (see README §Tests).

## 12. Change history (functional)

| Change | Reason |
|---|---|
| Glyph masks instead of box masks | Box masks blurred the background |
| Per-segment fixed mask | Removed flicker |
| Union masks on shared frames | A subtitle was left visible on the frame shared by two subtitles |
| Shot-cut-aware chunking | Dark blobs where the fill was copied from another shot |
| Optional watermark removal | Customer request |
| Defaults: ProPainter, watermark `auto`, current directory in and out | One command in the video folder |
| Renamed to Clean Clear | Owner's decision |
| Skip finished videos, atomic output, cache keyed by file and options, colour-exact and VFR-safe encoding (v0.3.4) | Review findings: batch restarts, stale caches, colour shift on BT.709 sources, audio drift on variable-frame-rate sources |
