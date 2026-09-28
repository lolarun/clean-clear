# Clean Frame — Technical Design

Version 0.2.x. Companion document: [FUNCTIONAL.md](FUNCTIONAL.md). Defect analyses: [MEMO.md](../../MEMO.md). Roadmap: [PLANNING.md](../../PLANNING.md).

## 1. Overview

Clean Frame is a single-process Python CLI. For each video it runs a fixed sequence of stages:

```
                 ┌───────────────────────── pass 1 ─────────────────────────┐
  video ──ffmpeg crop (subtitle band)──► OCR ──► segmentation ──► SRT
                 └──────────────────────────────────────────────────────────┘
                 ┌───────────────────────── pass 2 ─────────────────────────┐
  video ──► glyph masks ─► shot cuts ─► watermark mask
        ──► decode ─► [erase watermark] ─► [erase subtitles] ─► encode ─► _clean.mp4
                 └──────────────────────────────────────────────────────────┘
```

Design principles:

- **Only touch what must change.** Only the subtitle band is decoded for OCR; only the masked pixels of a small crop are inpainted and pasted back; everything else is copied through unchanged.
- **Stream, never load a video.** Frames flow through generators; memory is bounded by about one ProPainter chunk.
- **One decode, one encode.** Watermark and subtitle erasing are chained generators over the same frame stream.
- **Backends are interchangeable** behind a one-method interface (`erase`).
- **Expensive, deterministic results are cached** (OCR, watermark mask) so tuning and reruns are cheap.

## 2. Repository layout

```
main.py             entry point: sys.exit(cli.main())
cli.py              argument parsing, input discovery, model/encoder setup, per-file loop with error isolation
pipeline.py         per-video orchestration (both passes), OCR loop, watermark mask resolution
subtitles.py        OCR wrapper, subtitle-line detection, segmentation, SRT writer
masks.py            glyph / box masks, per-frame mask assignment
watermark.py        static watermark detection
video.py            ffmpeg wrappers: probe, decode, cut detection, encoder selection, threaded I/O helpers
device.py           ONNX Runtime provider selection
common.py           ROOT, version, log()
backends/
  __init__.py       backend interface and factory
  lama.py           LaMa (ONNX Runtime)
  propainter.py     ProPainter (PyTorch)
docs/spec/          this document and FUNCTIONAL.md
install.*, run.*    environment setup and launcher scripts
```

`ROOT` (repository root) is where `models/`, an optional `ffmpeg/` folder and the default `ProPainter/` checkout are looked up.

## 3. Runtime dependencies

| Component | Used for | Notes |
|---|---|---|
| ffmpeg / ffprobe | decode, encode, probing | on PATH or in `ROOT/ffmpeg`; NVENC optional |
| onnxruntime-gpu 1.22+ | OCR and LaMa | `ort.preload_dlls()` loads the pip-installed CUDA/cuDNN |
| RapidOCR 3.4+ | text detection and recognition | PP-OCRv6 small det + rec models; the direction classifier is disabled (`use_cls=False`) |
| OpenCV, NumPy | mask morphology, image ops | headless OpenCV |
| PyTorch (CUDA) | ProPainter only | torch 2.14 + cu130 on the test server |
| ProPainter checkout + 3 weight files | ProPainter only | `RAFT`, `RecurrentFlowCompleteNet`, `InpaintGenerator`; not part of this repository |

Models are never committed to git; the README lists every download URL.

## 4. Pipeline in detail

### 4.1 Probe and setup (`pipeline.process`)

`video.probe` runs `ffprobe` and returns `(W, H, fps, n_frames)`. The frame count comes from `nb_frames`, falling back to `duration × fps`; it is therefore **approximate** and the code never relies on it being exact (see §5.5). `parse_band` converts `--band` into even-aligned pixel rows `(y0, y1)`.

### 4.2 Pass 1 — OCR (`pipeline.run_ocr`)

- `video.read_frames(src, W, H, crop=(y0, y1))` decodes only the band through ffmpeg's `crop` filter (raw BGR over a pipe).
- **OCR skipping.** For each frame a cheap signature is computed: the boolean map of bright, low-saturation pixels (`min > 170 and max − min < 50`) on the half-resolution band. If fewer than `max(30, 8% of reference white pixels)` pixels differ from the last OCR'd frame and fewer than `--ocr-interval` frames have passed, the previous result is reused. On the test videos OCR ran on 24–36% of frames.
- **Recognition.** RapidOCR with `Det.limit_type=max`, `Det.limit_side_len=960` (an earlier `min` setting upscaled the band and was very slow), on the ONNX CUDA provider.
- **First-pass filter** (`subtitles.frame_boxes`): drop boxes with score below `--min-score`, height below `max(8, H × --min-height)`, or a centre outside 15–85% of the frame width. Boxes are converted to full-frame coordinates.
- Raw per-frame boxes are cached as JSON, keyed by video stem and band.

### 4.3 Subtitle-line detection and segmentation (`subtitles.py`)

1. **Subtitle line** (`subtitle_line`): histogram of all box y-centres in 4 px bins; the mode is the line position, the median height of boxes within 8 px of the mode is the text height. This uses the whole video, so recurring subtitle rows dominate over scene text.
2. **Line filter** (`filter_line`): keep boxes whose height is 0.75–1.33× the text height and whose centre is between `mode − 2.5h` and `mode + 0.5h`. A box above the main line is only accepted if it lies **entirely above** the main line (boxes overlapping the line vertically are scene content — this removed costume text being read as a second subtitle line).
3. **Text joining** (`join_text`): boxes are grouped into lines by y-centre proximity, sorted left to right, joined with spaces and newlines.
4. **Segmentation** (`build_segments`), two stages:
   - consecutive frames with identical text (spaces ignored, gaps up to `--max-gap`) form *runs*;
   - a run is merged into the previous one if their horizontal extents overlap by more than 40% and the text similarity (`difflib` ratio) is at least 0.9, or at least 0.6 when the shorter run is very short (OCR jitter). Two long, different sentences are kept apart.
   - segments shorter than `max(2, round(min_dur × fps))` frames are dropped.
5. **SRT**: start = `start / fps`, end = `(end + 1) / fps`, text = the most frequent OCR text of the segment (majority vote).

### 4.4 Masks (`masks.py`)

- **Glyph mask** (`glyph_masks`, default). One extra decode of the band. For each frame that belongs to a segment, white pixels inside the segment's box rectangle are counted per pixel. Pixels white in **at least 50%** of the segment's frames are strokes. The stroke map is dilated with an ellipse of radius `--grow` (default 8) to cover the outline, OR-ed with a copy shifted `--shadow` px (default 3) down and right for the drop shadow, and intersected with the box dilated by the same kernel. If a segment has too few white pixels (< 3% of its box area) it falls back to the box mask.
- **Box mask** (`box_masks`): union of a segment's OCR boxes expanded by `--dilate`.
- **Frame assignment** (`frame_masks`): every frame from `start − pad` to `end + pad` maps to the segment. A frame covered by several segments (back-to-back subtitles) maps to a tuple key and receives the **union** of their masks. Result: `frame_seg: {frame → key}`, `masks: {key → HxW bool}`. Masks are full-frame arrays that are zero outside the band; identical keys share one array.

### 4.5 Shot-cut detection (`video.detect_cuts`)

Only computed when the backend sets `uses_cuts = True`. The video is decoded as 320×180 grayscale; `d[j]` is the mean absolute difference between frames `j` and `j+1`. A cut is reported at `j+1` if `d[j] > 12` **and** `d[j] > 3 × max(median of the 5 differences on each side, 2)`. The spike rule matters: a plain threshold found 345 "cuts" in a 5-minute clip (fire, flashes, fast motion) versus 123 with the rule. Cost: about 130 s for 87 minutes of 720p.

### 4.6 Watermark mask (`watermark.detect`, `pipeline.watermark_mask`)

1. Grab 150 full frames evenly spaced between 3% and 97% of the duration (one `ffmpeg -ss` per sample, four parallel workers).
2. For each corner window (`min(W/3, 480) × min(H/4, 200)`), compute the per-pixel median colour of the stack.
3. A pixel is a watermark candidate if its colour stays within ±14 of the median in more than 85% of samples **and** the median image has an edge there (morphological gradient > 25, dilated) — this excludes flat regions such as black bars or uniformly dark scenes.
4. Close small gaps (7×7), keep connected components of at least 20 px, dilate by a 9×9 ellipse for anti-aliased edges.
5. Reject the result if it covers more than 5% of the frame.

The mask is cached as `.cache/<stem>.watermark.png` (an all-black image means "none found" and prevents redetection). `--watermark <png>` bypasses detection.

Measured: 13 s per video, independent of video length, IoU 0.99 between detection on the original and on an already-cleaned video.

### 4.7 Pass 2 — erase and encode

```python
stream = prefetch(read_frames(src, W, H))                    # decode thread
if wm is not None:
    stream = backend.erase(stream, {i: 0 for i in range(n + 50)}, {0: wm}, cuts)   # watermark on every frame
for frame in backend.erase(stream, frame_seg, masks, cuts):  # subtitles
    tw.write(frame.tobytes())                                # encode thread
```

- The two `erase` calls are lazy generators, so the video is decoded once and encoded once; the watermark stage buffers at most one chunk, the subtitle stage another.
- The watermark frame map deliberately covers `n + 50` frames because `n` is approximate; the ProPainter backend tolerates entries beyond the real end of the video.
- **Why the watermark is a separate erase call.** ProPainter processes one rectangle per call. The logo (top-left) and the subtitles (bottom-centre) are far apart, so a single rectangle would span most of the frame height. Two small crops are far cheaper. They also cover different frame sets (all frames vs. ~50%).
- **Threading (P4).** `video.prefetch` runs the ffmpeg reader in a background thread (bounded queue of 8); `video.ThreadedWriter` writes to the encoder's stdin from another thread (queue of 8) and re-raises worker errors on the next `write`/`close`. GPU inference stays on the main thread. Order is preserved because each side is a single-consumer queue.
- **Encoding** (`video.open_writer`): raw BGR on stdin, audio copied from the source (`-map 1:a?`, `-c:a copy`), NVENC `p5` VBR with `-cq` (`--crf`) and `-b:v 0` or libx264 `medium`, `yuv420p`, `+faststart`. `pick_encoder("auto")` tests NVENC with a 0.2 s synthetic clip and falls back to libx264.
- A non-zero ffmpeg exit code raises, which the CLI loop records as a failed file.

## 5. Backends

### 5.1 Interface (`backends/__init__.py`)

```python
backend.erase(frames, frame_seg, masks, cuts=()) -> iterator of frames
```

- `frames`: iterator of `H×W×3` BGR `uint8` frames, from frame 0.
- `frame_seg`: `{frame index: mask key}` for frames that need erasing.
- `masks`: `{mask key: H×W bool}`.
- `cuts`: shot-start frame indices, only passed if `backend.uses_cuts`.
- Must yield exactly one frame per input frame, in order.

`backends.create(model, ...)` imports lazily, so LaMa users never import PyTorch.

### 5.2 LaMa (`backends/lama.py`)

ONNX export `Carve/LaMa-ONNX` (`lama_fp32.onnx`), fixed 512×512 input, output in the 0–255 range, run through ONNX Runtime with the selected providers. A subtitle line is a long thin band, so `inpaint_packed` cuts the mask's bounding box into 512-wide strips at the columns with the fewest mask pixels (gaps between characters), each with 64 px of side context, and stacks them vertically into one 512×512 image: one model call inpaints a whole line (2–4× faster than sliding square windows). Layout is cached per mask object. Wide or tall masks fall back to `inpaint_frame` (sliding square windows). Only masked pixels are replaced. Frames without a mask pass through.

### 5.3 ProPainter (`backends/propainter.py`)

**Engine (`ProPainterEngine`).** Loads the three networks once: `RAFT_bi` (bidirectional optical flow), `RecurrentFlowCompleteNet` (fills flow inside the mask) and `InpaintGenerator` (image propagation, then feature propagation plus a sparse temporal transformer). Flow completion and the generator run in fp16, RAFT in fp32. A call processes an in-memory clip in stages, releasing the CUDA cache between stages, with sub-video windows so memory stays bounded. A one-frame clip is duplicated because flow needs two frames. `RAFT` iterations are configurable (`--pp-raft-iter`, default 12).

**Backend (`ProPainterBackend.erase`)** streams frames and decides *what* the engine sees:

1. **Runs and chunks (`_chunks`).** Frames to erase are grouped into runs (gaps ≤ 10 frames, same shot). Each run is extended by `PAD` frames (default 8) on both sides, clamped to the shot, and split into chunks of `--pp-chunk` frames (default 120). Each chunk is processed with `CTX` (default 10) context frames on each side, also clamped to the shot. **No chunk or context ever crosses a shot cut**: ProPainter assumes one continuous shot and would otherwise propagate pixels from another shot into the hole (root cause of the dark blobs analysed in MEMO.md §1).
2. **Vertical band.** Rows are the vertical extent of all masks ± 40 px, aligned to multiples of 8 and clamped to `H // 8 × 8` (RAFT requires multiples of 8; a 714-px-tall video otherwise fails inside RAFT's correlation lookup). Minimum height 128 px (`MIN_SIDE`) so tiny regions such as a corner logo still get context.
3. **Horizontal crop per chunk (P1).** Columns are the horizontal extent of the masks used in that chunk ± `--pp-margin` (80 px), rounded to multiples of 8, minimum width 128 px. Bands are cached full-width and cropped fresh for each chunk, so two chunks sharing a frame can use different crops. Frames are edge-padded and masks zero-padded to multiples of 8 if needed. This makes engine work proportional to text width instead of frame width.
4. **Streaming buffer.** A dictionary holds frames not yet emitted (`buf`) and original bands still needed as chunk input (`bands`). A chunk runs as soon as its last context frame has arrived; frames before the next pending chunk are emitted immediately. If the video ends before the last chunk's context is complete, the remaining chunks run on what exists. `need` is computed up front so only required bands are copied.
5. **Compositing.** Engine output is pasted back only where the mask dilated by 4 px is set, and only for the chunk's own frame range (`s..e`), not for context frames.

Peak VRAM: about 13 GB at chunk 120 on the 1080p test clip; 7.0 / 15.6 / 20.3 GB for 208 / 464 / 624-row bands at 71 frames. A chunk of 240 (+20 context) frames needed more than 24 GB, hence the default of 120.

## 6. Data structures and invariants

| Name | Shape | Notes |
|---|---|---|
| `per_frame` | `[[ (x0,y0,x1,y1,text) ]]` | one entry per frame, full-frame coordinates, JSON-cacheable |
| `Seg` | `start, end, texts: Counter, boxes` | inclusive frame range |
| `frame_seg` | `{int: key}` | key is an `int` (watermark) or a tuple of segment indices |
| `masks` | `{key: H×W bool}` | full-frame arrays, shared between equal keys; treated as read-only |
| `cuts` | sorted `[int]` | first frame of each new shot |
| chunk | `(s, e, cs, ce)` | output range `s..e`, input range `cs..ce`, inclusive |

Invariants: output frame `i` equals input frame `i` outside the masks; the number of yielded frames equals the number of decoded frames; chunks never span a cut; engine inputs have height and width divisible by 8.

## 7. Performance model

Measured on an NVIDIA A10 (24 GB), 1280×714, ProPainter, chunk 120:

| Stage | 87.6 min video | Notes |
|---|---|---|
| OCR | 1683 s | 47,872 of 131,345 frames recognized, ≈ 78 fps effective |
| Glyph masks | 171 s | one extra decode of the band |
| Cut detection | 130 s | ≈ 40× real time, CPU/decode bound |
| Erase + encode | 9999 s | ≈ 13 fps overall; 47.6% of frames contain subtitles; pass-through frames run at ≈ 150 fps |
| Watermark (5-min clip) | +291 s / 7,499 frames | ≈ 39 ms per frame |

Cost drivers and levers:

| Factor | Effect | Lever |
|---|---|---|
| Frames containing subtitles (+ ≈ 27% padding and context) | linear | `--pp-pad`, `--pp-ctx` |
| Crop area (text width + margin) × band height | roughly linear in GPU time | `--pp-margin`, `MIN_SIDE` |
| RAFT iterations (≈ 30–40% of engine time) | linear | `--pp-raft-iter` |
| Resolution | decode/OCR/mask cost ≈ ×1.7–2.4 at 1080p; erase ≈ ×1.9 (estimate) | — |
| Small crops under-fill the GPU | speed-up on faster GPUs is smaller than for large crops | run 2 processes per GPU |

Implemented optimizations: horizontal crop (P1), fewer RAFT iterations (P2, 20 → 12, **quality not yet validated on a large sample**), tunable context and padding (P3, partial — flow is **not** reused across chunk overlaps), decode/inference/encode threading (P4). Remaining ideas are in PLANNING.md (clean plate for static shots, shot-wide reference frames, half-resolution mode, `torch.compile`).

## 8. Error handling

| Situation | Behaviour |
|---|---|
| Missing path, no videos found | Logged / `sys.exit("No video files found")` |
| ffmpeg missing | `sys.exit` with install hint |
| Device unavailable | `sys.exit` listing available ONNX providers |
| ProPainter weights or checkout missing | `sys.exit` with the missing path |
| Exception while processing one video | Logged as `!! failed`, batch continues, exit code 1 |
| ffmpeg encoder failure / broken pipe | `RuntimeError`; encoder thread errors are re-raised in the main thread |
| Watermark mask of wrong size | `ValueError` with the expected size |
| Watermark detection finds nothing / > 5% area | Logged, video processed without it |
| CUDA out of memory | Not caught; lower `--pp-chunk` |
| Video ends earlier than `nb_frames` | Handled: trailing chunks run on available frames |

## 9. Testing and verification

There is no automated test suite in the repository; verification so far was done as follows.

**Offline, with fakes** (no GPU, no models): chunk layout and cut clamping; streaming order and frame count with a fake engine; horizontal crop widths and pasting; chained watermark + subtitle `erase` at 1280×714 with a shot cut, asserting the engine sees dimensions divisible by 8 and that logo and subtitle pixels change only where intended; `prefetch` / `ThreadedWriter` ordering with 2000 items.

**On the A10 server, real data:**

| What | Result |
|---|---|
| 5-minute 1080p clip, full pipeline | comparison against the customer's reference output: on par for static shots |
| Full 87.6-minute 720p film | completes, output 3.1 GB, ≈ 3.3 h |
| 0:25 dark blobs (MEMO.md §1) | reproduced, root-caused to shot cuts, fixed and verified on a full rerun |
| 2:56 residual text (MEMO.md §2) | fixed by union masks |
| Watermark on 5-minute clip | logo removed in five spot checks incl. static dark and textured backgrounds |

**Not yet verified:** RAFT at 12 iterations across the full test set; 1080p timings; RTX 3050 and RTX 5090 runs (VRAM at chunk 60–80, PyTorch build for Blackwell); Windows end-to-end run of the ProPainter path; multi-video batches with mixed resolutions.

Suggested regression checks: the fixed timestamps listed in PLANNING.md §Validation, the residual-text OCR check on the output, and a frame-count/duration comparison between input and output.

## 10. Known risks and open issues

- **ProPainter licence** is non-commercial; the LaMa backend is the licence-clean fallback.
- **Approximate frame count** from ffprobe; any new code must not assume an exact `n`.
- **Whole-frame copies:** the watermark stage copies every frame it buffers, and the subtitle stage copies frames it edits; at 1080p this is a few MB per frame and is small next to inference, but it adds up on very fast GPUs.
- **Chunk seams:** neighbouring chunks are solved independently (PLANNING Q5, cross-fade not implemented).
- **Very short shots** (≥ 1 frame) get the duplicate-frame fallback and little context.
- **Detection heuristics** (subtitle line, white-glyph masks, watermark) are tuned on Chinese dialogue films with white subtitles and one opaque logo; other styles need parameter changes or manual masks.
- **CPU-bound stages** (decode, OCR pre-processing, cut detection, mask counting) do not speed up with a faster GPU; a weak CPU on the customer's machine will limit throughput.
- `ROOT` was resolved one directory too high after the files were moved to the repository root; fixed by using the directory of `common.py`. Servers must have `models/` (and `ProPainter/` or `--propainter-dir`) next to `main.py`.

## 11. Extension points

- **New inpainting model:** add `backends/<name>.py` with a class implementing `erase(frames, frame_seg, masks, cuts=())`, set `uses_cuts` if it is temporal, register it in `backends.MODELS` and `create()`.
- **Second watermark or moving logo:** call `erase` again in `pipeline.process` with another mask (per-frame masks are already supported by `frame_seg`).
- **Different OCR engine:** replace `subtitles.OCR`; the rest only needs `[(box, text, score)]` per frame.
- **Per-video parameters:** `pipeline.process(src, out_dir, args, ...)` reads everything from `args`, so a config-file layer can be added in `cli.py`.
- **Parallel processing:** run several `main.py` instances on disjoint folders, or several per GPU for small-crop workloads; there is no shared state except the output directory.
