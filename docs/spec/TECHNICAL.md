# Clean Clear — Technical Design

Version 0.3.x. Companion document: [FUNCTIONAL.md](FUNCTIONAL.md). Defect analyses: [MEMO.md](../../MEMO.md). Roadmap: [PLANNING.md](../../PLANNING.md).

## 1. Overview

Clean Clear is a single-process Python CLI. For each video it runs a fixed sequence of stages:

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
src/clean_clear/                    the package (src layout, installed with `pip install -e .`)
  __main__.py                       `python -m clean_clear`
  cli.py                            argument parsing, input discovery, model/encoder setup, per-file loop with error isolation
  pipeline.py                       per-video orchestration (both passes, verify), OCR loop, watermark mask resolution,
                                    cache paths, temporary/final output paths
  subtitles.py                      OCR wrapper, subtitle-line detection, segmentation, SRT writer
  masks.py                          MaskSet (crop-based mask storage), glyph / box masks, per-frame mask assignment
  stabilize.py                      motion-compensated temporal smoothing of filled areas
  rewrite.py                        re-encode only keyframe windows of a finished video, stream-copy the rest
  refine.py                         --refine-of: post-fix an already cleaned video
  parallel.py                       --jobs: split at keyframes, run part processes, merge video and SRT
  watermark.py                      static watermark detection
  video.py                          ffmpeg wrappers: probe, colour/VFR handling, decode, cut detection, encoder selection,
                                    threaded I/O helpers, duration check
  device.py                         ONNX Runtime provider selection, GPU compute capability and memory
  common.py                         ROOT, version, log(), Progress, usable_cpus()
  backends/__init__.py              backend interface and factory
  backends/lama.py                  LaMa (ONNX Runtime)
  backends/propainter.py            ProPainter (PyTorch)
scripts/install.sh, install.bat     environment setup
tests/                              pytest suite (synthetic videos, fake OCR and models; needs ffmpeg, no GPU)
docs/spec/                          this document and FUNCTIONAL.md
pyproject.toml                      dependencies (extras `propainter`, `test`), version, the `clean-clear` console script
```

`ROOT` (repository root, three levels above `common.py` in a source checkout or editable install) is where `models/`, an optional `ffmpeg/` folder and the default `ProPainter/` checkout are looked up. A non-editable install would not find them, which is why the install scripts use `pip install -e .`.

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

`video.probe` runs `ffprobe` and returns `(W, H, fps, n_frames)`. `fps` is the average frame rate. The frame count comes from `nb_frames`, falling back to `duration × fps`; it is therefore **approximate** and the code never relies on it being exact (see §5.5). A source whose average and nominal rates differ by more than 0.5% is treated as variable frame rate: `n = duration × fps`, and every decoder (`read_frames`, `detect_cuts`) resamples it to `fps` so that all passes share one frame numbering (§4.8a). `video.colour` returns the YUV matrix, range and output colour tags used by `read_frames` and `open_writer`. `parse_band` converts `--band` into even-aligned pixel rows `(y0, y1)`.

### 4.2 Pass 1 — OCR (`pipeline.run_ocr`)

- `video.read_frames(src, W, H, crop=(y0, y1))` decodes only the band through ffmpeg's `crop` filter (raw BGR over a pipe).
- **OCR skipping.** For each frame a cheap signature is computed: the boolean map of bright, low-saturation pixels (`min > 170 and max − min < 50`) on the half-resolution band. If fewer than `max(30, 8% of reference white pixels)` pixels differ from the last OCR'd frame and fewer than `--ocr-interval` frames have passed, the previous result is reused. On the test videos OCR ran on 24–36% of frames.
- **Recognition.** RapidOCR with `Det.limit_type=max`, `Det.limit_side_len=960` (an earlier `min` setting upscaled the band and was very slow), on the ONNX CUDA provider.
- **First-pass filter** (`subtitles.frame_boxes`): drop boxes with score below `--min-score`, height below `max(8, H × --min-height)`, or a centre outside 15–85% of the frame width. Boxes are converted to full-frame coordinates.
- Per-frame boxes are cached as JSON (`pipeline.ocr_cache_path`), keyed by video stem, band, the file's size and modification time, and the options applied during recognition (`--min-score`, `--min-height`, `--ocr-interval`): a replaced file or changed option never reuses a stale result.

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
- **Frame assignment** (`frame_masks`): every frame from `start − pad` to `end + pad` maps to the segment. A frame covered by several segments (back-to-back subtitles) maps to a tuple key and receives the **union** of their masks. Result: `frame_seg: {frame → key}`, `masks: {key → HxW bool}`. Masks are kept in a `MaskSet`: each mask is stored as the crop around its pixels and the full-frame (read-only) array is built on access, with the 16 most recently used arrays cached so that a backend asking for the same key on consecutive frames gets the same object. Full-frame storage, plus full-band per-pixel counts in `glyph_masks`, took ~7.6 GB per 1,500 subtitles at 1080p; crops take ~0.3 GB.

### 4.5 Shot-cut detection (`video.detect_cuts`)

Only computed when the backend sets `uses_cuts = True`. The video is decoded as 320×180 grayscale; `d[j]` is the mean absolute difference between frames `j` and `j+1`. A cut is reported at `j+1` if `d[j] > 12` **and** `d[j] > 3 × max(median of the 5 differences on each side, 2)`. The spike rule matters: a plain threshold found 345 "cuts" in a 5-minute clip (fire, flashes, fast motion) versus 123 with the rule. Cost: about 130 s for 87 minutes of 720p.

### 4.6 Watermark mask (`watermark.detect`, `pipeline.watermark_mask`)

1. Grab 150 full frames evenly spaced between 3% and 97% of the duration (one `ffmpeg -ss` per sample, four parallel workers).
2. For each corner window (`min(W/3, 480) × min(H/4, 200)`), compute the per-pixel median colour of the stack.
3. A pixel is a watermark candidate if its colour stays within ±14 of the median in more than 85% of samples **and** the median image has an edge there (morphological gradient > 25, dilated) — this excludes flat regions such as black bars or uniformly dark scenes.
4. Close small gaps (7×7), keep connected components of at least 20 px, dilate by a 9×9 ellipse for anti-aliased edges.
5. Reject the result if it covers more than 5% of the frame.

The mask is cached as `.cache/<stem>.<file key>.watermark.png` (an all-black image means "none found" and prevents redetection). `--refine-of` reads this cache to know exactly what the first pass erased (falling back to the pre-0.3.4 name, then to a re-detection with the 0.3.2 rules). `--watermark <png>` bypasses detection.

Measured: 13 s per video, independent of video length, IoU 0.99 between detection on the original and on an already-cleaned video.

### 4.7 Pass 2 — erase and encode

```python
stream = prefetch(read_frames(src, W, H))                    # decode thread
if wm is not None:
    stream = backend.erase(stream, all_frames(n), {0: wm}, cuts)   # watermark on every frame
for frame in backend.erase(stream, frame_seg, masks, cuts):  # subtitles
    tw.write(frame.tobytes())                                # encode thread
```

- The two `erase` calls are lazy generators, so the video is decoded once and encoded once; the watermark stage buffers at most one chunk, the subtitle stage another.
- The watermark frame map (`all_frames`) deliberately covers `n + max(250, 2% of n)` frames because `n` is approximate; the backends tolerate entries beyond the real end of the video.
- **Why the watermark is a separate erase call.** ProPainter processes one rectangle per call. The logo (top-left) and the subtitles (bottom-centre) are far apart, so a single rectangle would span most of the frame height. Two small crops are far cheaper. They also cover different frame sets (all frames vs. ~50%).
- **Threading (P4).** `video.prefetch` runs the ffmpeg reader in a background thread (bounded queue of 8); `video.ThreadedWriter` writes to the encoder's stdin from another thread (queue of 8) and re-raises worker errors on the next `write`/`close`. GPU inference stays on the main thread. Order is preserved because each side is a single-consumer queue.
- **Encoding** (`video.open_writer`): raw BGR on stdin, audio copied from the source (`-map 1:a?`, `-c:a copy`), NVENC `p5` VBR with `-cq` (`--crf`) and `-b:v 0` or libx264 `medium`, `yuv420p`, `+faststart`. `pick_encoder("auto")` tests NVENC with a 0.2 s synthetic clip and falls back to libx264.
- A non-zero ffmpeg exit code raises, which the CLI loop records as a failed file.

### 4.8 Safety nets added after customer review (v0.3.1, not yet measured on a GPU server)

A customer spot-check of the first delivery found a subtitle left on screen and flicker in the subtitle and watermark areas. Frame statistics over the delivered video (white-pixel counts in the subtitle line, frame-to-frame change in the logo area against an untouched neighbouring patch) pointed to three causes, each with a countermeasure:

1. **Subtitle partly erased.** In all 5 leftovers found in the first 63 minutes (e.g. 14:27.3, 42 frames) the first part of a sentence was erased and the rest was not: OCR stopped seeing the text but the same glyphs stayed on screen. `masks.glyph_masks(extend=...)` checks up to `--extend-sec` (3 s) of frames before/after every segment against that segment's own glyph pixels (>= 60 % of the strokes white, little white outside them, 2 bad frames tolerated) and widens the erase range (`Seg.ext_lo/ext_hi`, used by `frame_masks`). It runs inside the existing pass over the subtitle band, so it adds no decoding. The SRT keeps the OCR timing.
2. **Anything still missed.** `pipeline.verify_and_fix` OCRs the finished video again with the first pass's subtitle line, and erases whatever is still readable in a second pass over the result (written to `*.fix.mp4`, then replacing the output; the first result is kept if the pass fails). It costs one more OCR pass (cheap, because a cleaned band rarely has white text, so most frames are skipped) and, only when something is found, one more decode and encode. `--verify off` disables it.
3. **Bright flashes in the logo area.** In dark scenes the watermark fill came out as a bright logo-shaped blob for ~4 frames (seen at 42:34-42:36); in the first 63 minutes there were 75 isolated jumps above 5 grey levels in the output against 10 in the source. `watermark.guard_fill` compares the mean brightness inside the mask with a ring of untouched pixels around it and re-fills a frame with a spatial inpaint (`cv2.inpaint`) when the ring is dark (mean < 60) and the fill is brighter by more than `--wm-guard` (25) grey levels. The first version also repaired darker or brighter fills and smeared the model's good output on bright lattice windows (362 of 2250 frames in one clip), hence the restriction; in a dark scene it cut the flash jumps from 42 to 4. This treats the symptom; the cause inside the model is not established. Chunk boundaries (every 120 frames) showed about 5x more jumps than other frames, but they account for only ~10 % of the jumps.

**Root cause of the leftover at 14:28 (found afterwards).** OCR read the sentence with confidence 0.89, but the detector box was 74 px tall instead of the usual ~52 px because it also enclosed a light streak on the floor (read as a trailing `\`). `subtitles.filter_line` accepted only 0.75-1.33x the median text height, so the box was discarded as scene text, and the verify pass uses the same filter and could not see it either. The upper bound is now 1.6x (the position test still keeps scene text out) and stray trailing symbols are stripped from the text. A statistic based on "an erased run followed by the same glyphs" cannot see such whole-sentence misses; the OCR check on the result can.

**Logo tick.** Two small semi-transparent green ticks belong to the logo; their colour depends on the background, so the static-colour test never selects them and a few pixels stayed visible in dark scenes. `watermark.leftover_accents` adds saturated, bright pixels in a ring around the mask that occur in >= 5 % of the sampled frames; the first pass now includes them, and `--refine-of` can fill them on an existing result.

**Parallel run (`--jobs N`, `parallel.py`) and post-fix (`--refine-of`, `refine.py`).** `--jobs` splits the video at keyframes into 2N parts, runs N independent Clean Clear processes pinned to disjoint CPU sets (OCR, masks, erase, verify each), takes the watermark mask from one detection on the whole film, and merges the parts (SRT time-shifted, video stream-copied, original audio). On a 16-vCPU L20 four processes reached 100 % GPU utilisation at about 19 frames/s in total (single process 12-13), so more processes do not help; 87.5 min of 720p took 2 h including OCR. `--refine-of` OCRs a finished result with the subtitle line taken from the original, erases what is still readable and fills missed logo pixels, writing one new file.

Not supported by the data and therefore not changed: bridging OCR gaps inside a sentence (0 cases found), larger `--pad-frames`, and lowering `--pp-ref-stride` (no measurable effect on the flicker; it only makes the transformer step slower).

### 4.8a Robustness fixes after code review (v0.3.4)

- **Colours.** `video.colour()` determines the YUV matrix and range of a video (its tags; untagged: BT.709 from 720 lines, BT.601 below). `read_frames` converts with `scale=in_color_matrix=…`, `open_writer` with `out_color_matrix=…` and writes the colour tags. Before, a tagged BT.709 film was decoded as BT.709 but encoded as BT.601 without tags (a test bar moved from YCbCr 84/154/158 to 93/150/156).
- **Variable frame rate.** `probe` flags a source whose average and nominal rates differ by more than 0.5%; `read_frames` and `detect_cuts` then resample it to the average rate (`-fps_mode cfr`), and the frame count is `duration × fps`. Before, a 6 s VFR clip came out as 9.2 s of video against 6 s of audio. `duration_check` logs a warning when an output's video length differs from the source's by more than 0.5 s.
- **Encoder failure.** `ThreadedWriter` keeps draining its queue after a write error, so `write()`/`close()` raise instead of blocking forever when ffmpeg exits (seen as a hang with an unknown encoder; NVENC session limits or a full disk do the same).
- **Early stop.** `prefetch` stops its thread and closes the reader when the consumer stops; `read_frames` kills its ffmpeg if closed early. A failed video no longer leaves a decoder process behind.
- **Atomic output and restart.** `process` writes `.<name>_clean.tmp.mp4`, runs verify on it and renames it when everything succeeded; on failure it is deleted. The CLI skips videos whose `_clean.mp4` and `.srt` exist (`--force` to redo) and logs a traceback for failures.
- **`--jobs`.** Part processes get `--split-part`: segments touching a part's first or last frames are kept even if shorter than `--min-dur` (a subtitle cut by the split was dropped and stayed on screen). `parallel.merge_cues` joins the two halves of such a subtitle into one cue. SRT offsets come from the frame counts of the cleaned parts (what the merged video is made of), the merged frame count and duration are checked, and a GPU-memory warning is logged when the processes will not fit.
- **Watermark frame map** (`pipeline.all_frames`) reaches `n + max(250, 2% of n)`; if the video still has more frames, the run fails instead of silently leaving the logo on the last frames.

### 4.9 Third customer review (v0.3.3)

"Hardly visible, but at 2x speed it flickers at 45:08, 51:48-52:00, 54:30." Frame-exact comparison of the delivery with the source (time-based decoding; OpenCV frame seeking was off by several frames on these MP4s), then A/B runs of 30 s clips on an L20:

- **Shimmer of the subtitle fill.** No repeated or dropped frames and no brightness steps; the logo corner changed only slightly. In the erased subtitle area the result changed 1.3-2.4x as much from frame to frame as the source does there. `stabilize.stabilize` blends each filled pixel with the previous output frame warped along Farneback optical flow (flow computed with the fill blurred, so it follows the surroundings), weighted by how well the warped frame matches an untouched ring around the area (`--stabilize` = 0.6 at a perfect match, 0 at a mean ring mismatch of 6 grey levels), restarting at shot cuts and mask changes. Clip results (result/source change in the erased area, off -> on): 51:48 1.35 -> 1.01, 45:05 0.90 -> 0.82, 54:30 2.44 -> 2.34. The rest at 54:30 is fast motion right through the subtitle area (a hand and sleeve), where the hidden content is guessed anew every frame; smoothing cannot recover it.
- **Missed shot cut.** At 54:41 a hard cut right after fast motion (grey change 39.4 against 15-23 around it) failed the "3x the local level" test, so ProPainter treated both shots as one and carried dark content of the first into the second (a dark smear for several frames). `video.detect_cuts` also accepts a spike of the HSV colour-histogram distance (> 0.2 and 3x its local level): 0.33 at that cut, 0.06-0.09 during the motion before it. On the three test clips it added exactly that cut and nothing else.
- **The film was 4.3 luma levels darker than the source.** Every YUV->RGB->YUV round trip with ffmpeg's default swscale flags lowered luma by ~1.5 levels (-1.54, -2.96, -4.37 after 1-3 round trips); with `accurate_rnd+full_chroma_int+bitexact` it is -0.05 and does not accumulate. The flags must follow `-i`: placed before it, ffmpeg 6.1 ignores them silently (first attempt: still -1.7). Clips now measure +0.03 against the source. Values outside 16-235 are still clipped by the RGB conversion.
- **Fewer generations.** The verify pass and `--refine-of` re-encoded the whole file for a few subtitles. `rewrite.rewrite` re-encodes only keyframe-aligned windows around the changed frames and stream-copies the rest. Three details were needed for clean joins: pieces are cut by frame count (a time cut at a keyframe carried packets of the next GOP); copies start at the keyframe's exact pts (seeking a quarter frame past it shifted the piece by one frame); and the concat list carries each piece's exact duration (an encoded window's last frame was stored with a 1-tick duration). Inputs with hidden pre-roll frames (stream-copy cuts, detected by negative or discarded packets) are refused, and `--refine-of` falls back to a full re-encode for them. Disabling B-frames was tried and dropped: +27 % file size.
- **A blinking logo dot.** The logo's yellow-green tick is semi-transparent and lights up only now and then (its tip was bright in 4 of 136 dark samples), reaching 2-3 px past the erased area; on dark scenes it showed as a blinking green dot (e.g. 42:34-42:42). Static-colour detection cannot see a part that is usually absent, and an after-the-fact check was tried and dropped: "bright next to the logo on a dark background" flagged 18 % of the frames (bright window lattices, lamps) and smeared them; adding a colour test learned from the logo still matched foliage. The fix is a wider margin: `watermark.detect` now dilates the logo mask by 7 px (`grow=15`) instead of 4 px, which covers the tip in the first pass. Not yet measured on a full run. The delivery of this film was patched once by hand: the tip area was filled only in the 146 frames where it showed (positions learned from the stretch checked by eye), re-encoding 4,500 frames in 10 windows.
- **Measured on the full film (L20, 16 vCPU, `--jobs 4`, 2 h 8 min):** luma -0.05..+0.03 against the source at five points; erased-area steadiness (result/source frame-to-frame change) 0.99 at 51:48 (was 1.35), 1.21 at 45:05; 54:30 and 54:41 stay at 2.4-2.7: fast motion through the subtitle area and a short shot where the background under the text is never revealed, so the model guesses it anew every frame. The cut at 54:41 was detected; the dark smear there is that guess, not content from the previous shot. Verify windows re-encoded 250-1000 frames per part instead of whole parts; the file is 2.74 GB instead of 3.08 GB.
- **NVIDIA MPS** (concurrent kernels from several processes; 90 s clips, other jobs were running on the machine at the same time, so the numbers are indicative): 4 processes 491 s without MPS, 420 s with MPS (-14 %, GPU power 256 -> 294 W); 8 processes with MPS were slower per clip than 4; `--pp-chunk 240` with MPS was 7 % slower than 120. GPU memory peaked at 15.5 GB with 4 processes and 30 GB with 8.
- Audio/video sync is fine: the source has 17 one-frame gaps (0.72 s in total) and the result is constant-rate, which puts the picture at most 41 ms off the audio.

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
| `masks` | `MaskSet` `{key: H×W bool}` (or a plain dict for the watermark) | stored as crops; indexing returns a read-only full-frame array, the same object while it stays in the 16-entry cache |
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

### 7.1 Measurement on an RTX 5090 container (2026-09-28)

Full run of the 87.6-minute 1280×714 film with all defaults (ProPainter, watermark `auto`, subtitles extracted and erased, libx264 because the container has no NVENC access, 16-core CPU quota):

| Stage | Time |
|---|---|
| OCR (47,872 of 131,345 frames recognized) | 3452 s |
| Glyph masks / shot cuts / watermark detection | 198 s / 84 s / 12 s |
| Erase subtitles + erase watermark + encode | 7398 s |
| **Total** | **11,152 s (3 h 06 min), about 2.1 h per hour of video** |

The estimate made before the run (about 1.7 h) was scaled from A10 timings with GPU spec ratios and was too optimistic. During the erase stage the GPU was busy only about 55% of the time and the Python main process ran at about 1.1 cores while the container used about 2.3 of its 16 cores: the stage is bound by single-threaded Python work between GPU calls (cropping, padding, copying results back, compositing, whole-frame copies), which a faster GPU does not speed up. OCR was slower than on the A10 (3452 s vs 1683 s).

### 7.2 Container quirks found on that machine and their workarounds

- **CPU quota.** `os.cpu_count()` reported 128 but the cgroup quota was 16 cores (`/sys/fs/cgroup/cpu.max`). Thread pools sized from the host (onnxruntime, OpenBLAS/OpenMP, PyTorch, OpenCV) were throttled hard; OCR on CPU took 1.28 s per call with default threads and 0.15 s with 4. `common.usable_cpus()` (affinity capped by the quota) now sizes them: at most 4 onnxruntime threads for OCR, at most 8 for OpenMP/BLAS, PyTorch and OpenCV.
- **Shape changes in onnxruntime-gpu 1.23.2 on Blackwell.** The recognition model ran at 2–8 ms per call for a constant input shape, but each change of shape cost 2–3 s and alternating between many shapes kept every call slow (OCR 205 s for a 40 s clip). `subtitles._FixedShape` pads every recognition batch to `(6, 3, 48, 1280)` (zero padding, the same way RapidOCR pads within a batch), which brought the same clip to 35 s. It is enabled automatically for compute capability 12+ (`--ocr-fixed-shape`), and `cudnn_conv_algo_search=HEURISTIC` is set for the CUDA provider. Two English credit lines differed by one character or space from the variable-shape output. Similar symptoms are reported for RTX 5090 by others ([onnxruntime issue 28305](https://github.com/microsoft/onnxruntime/issues/28305)); the cause was not identified and the container's GPU virtualization may contribute, so this is a workaround, not a documented fix.
- **No NVENC.** `NVIDIA_DRIVER_CAPABILITIES=compute,utility`: the encoder libraries exist but opening a session fails (`unsupported device`); `pick_encoder("auto")` falls back to libx264.

## 8. Error handling

| Situation | Behaviour |
|---|---|
| Missing path, no videos found | Logged / `sys.exit("No video files found")` |
| ffmpeg missing | `sys.exit` with install hint |
| Device unavailable | `sys.exit` listing available ONNX providers |
| ProPainter weights or checkout missing | `sys.exit` with the missing path |
| Exception while processing one video | Logged as `!! failed` with a traceback, the temporary output is deleted, batch continues, exit code 1 |
| ffmpeg encoder failure / broken pipe | `RuntimeError` on the next `write`/`close` (the writer thread keeps draining, so nothing blocks) |
| Two inputs with the same name, different extension | The second is reported as failed (they would share outputs) |
| Output video shorter/longer than the source | Warning in the log |
| Watermark mask of wrong size | `ValueError` with the expected size |
| Watermark detection finds nothing / > 5% area | Logged, video processed without it |
| CUDA out of memory | Not caught; lower `--pp-chunk` |
| Video ends earlier than `nb_frames` | Handled: trailing chunks run on available frames |

## 9. Testing and verification

`pytest` (see README §Tests) runs on synthetic videos with ffmpeg and fake OCR/inpainting models, no GPU needed: colour round trips (tagged BT.709, untagged HD and SD), grey level, variable frame rate, encoder failure, early stop of readers, frame-exact windows; MaskSet, union masks, glyph masks and glyph-matched extension; segmentation incl. split edges; cache keys; input filtering, name clashes, skipping finished videos; `--jobs` cue merging and child options; refine's first-pass mask; LaMa and ProPainter streaming with fake models (order, frame count, only masked pixels change, chunks never cross a cut); and `process()` end to end (SRT text and timing, subtitle erased, verify pass, failed run leaves no output, OCR cache reused). Earlier verification, still relevant for model quality:

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
- **Approximate frame count** from ffprobe; any new code must not assume an exact `n`. Maps that must cover every frame use `pipeline.all_frames(n)`.
- **Outputs of 0.3.3 and earlier** are untagged and were encoded with BT.601; a partial re-encode of such a file (`--refine-of` windows) can show a slight colour difference between rewritten windows and copied pieces. Reprocess from the original if this matters.
- **`--jobs` part boundaries:** parts are stream-copy cuts at keyframes; ProPainter context and stabilization restart at each boundary, and a subtitle across a boundary is processed as two halves (joined again in the SRT).
- **Whole-frame copies:** the watermark stage copies every frame it buffers, and the subtitle stage copies frames it edits; at 1080p this is a few MB per frame and is small next to inference, but it adds up on very fast GPUs.
- **Chunk seams:** neighbouring chunks are solved independently (PLANNING Q5, cross-fade not implemented).
- **Very short shots** (≥ 1 frame) get the duplicate-frame fallback and little context.
- **Detection heuristics** (subtitle line, white-glyph masks, watermark) are tuned on Chinese dialogue films with white subtitles and one opaque logo; other styles need parameter changes or manual masks.
- **CPU-bound stages** (decode, OCR pre-processing, cut detection, mask counting) do not speed up with a faster GPU; a weak CPU on the customer's machine will limit throughput.
- `ROOT` is derived from the location of `common.py` (repository root = `parents[2]`); it broke once when files were moved, so keep it in sync with the layout. Servers must have `models/` (and `ProPainter/` or `--propainter-dir`) in the repository root.

## 11. Extension points

- **New inpainting model:** add `backends/<name>.py` with a class implementing `erase(frames, frame_seg, masks, cuts=())`, set `uses_cuts` if it is temporal, register it in `backends.MODELS` and `create()`.
- **Second watermark or moving logo:** call `erase` again in `pipeline.process` with another mask (per-frame masks are already supported by `frame_seg`).
- **Different OCR engine:** replace `subtitles.OCR`; the rest only needs `[(box, text, score)]` per frame.
- **Per-video parameters:** `pipeline.process(src, out_dir, args, ...)` reads everything from `args`, so a config-file layer can be added in `cli.py`.
- **Parallel processing:** `--jobs` for one long video, or several `clean-clear` processes on disjoint folders. Processes writing to the same output directory share `.cache/` (keys include the file fingerprint, so they do not collide) and the skip-if-finished check.
- **Tests:** add a pytest under `tests/`; `conftest.py` has helpers for synthetic videos, and `tests/test_pipeline.py` shows how to drive `process()` with a fake OCR and backend.
