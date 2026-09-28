# Planning: quality and performance

Roadmap for improving Clean Frame, mainly the ProPainter model (`backends/propainter.py`). Effort estimates are rough and speed-ups are untested. See [MEMO.md](MEMO.md) for defect analyses.

## Baseline

Test clip: 1920×1080, 25 fps, 5 min, 100 subtitles, 7515 frames (3785 with subtitles). GPU: NVIDIA A10 24 GB.

| Model | OCR | Masks | Erase + encode | Total | Peak VRAM |
|---|---|---|---|---|---|
| LaMa | 154 s | 22 s | 341 s | ~8.6 min | ~2 GB |
| ProPainter (chunk 120) | 154 s | 22 s | ~30–40 min | ~35–45 min | ~13 GB |

ProPainter processes 4792 frames to erase 3785 (≈27% overhead from chunk context and padding).

## Quality

Smearing appears where the background behind the subtitle has to be invented rather than copied from another frame. Background that is visible anywhere in the same shot can be restored almost perfectly. Background that stays covered for the whole shot can only be made plausible and temporally stable, not exact.

### Current causes of smearing

1. **Short temporal reach** – each chunk sees only ~5 s (120 frames + 10 context frames per side). Background exposed earlier or later in the shot is never used.
2. **Wrong completed flow** – optical flow inside the mask is completed by the model; errors drag pixels along (the classic smear). The inpainted band is only ~40 px taller than the text on each side, which gives RAFT little context.
3. **Chunks cross shot cuts** – pixels propagate from one shot into another. Confirmed as the cause of the dark blobs at 0:25 (see MEMO.md). *Fixed by Q1.*
4. **Chunk seams** – neighbouring chunks are solved independently, so results can jump at chunk boundaries.
5. **Never-exposed regions** – ProPainter's transformer hallucinates these per frame window; results are soft and may drift between windows.

### Planned work

| # | Item | Fixes | Effort | Priority |
|---|---|---|---|---|
| Q1 | **Shot detection**; chunks never cross a cut | 3 | 0.5 d | ✅ done, verified on the test clip |
| Q2 | **Clean plate for static shots**: per shot, estimate camera motion; if static, fill every masked pixel from the nearest frame in the shot where it is unmasked (temporal median of unmasked observations) | 1, 2 for static shots | 1 d | P0 |
| Q3 | **Shot-wide reference frames**: pick ProPainter's non-local references from the whole shot, preferring frames where the masked area is exposed, instead of a fixed stride inside the chunk | 1 | 0.5–1 d | P1 |
| Q4 | **Taller band** (e.g. 2–3× text height of context above) for flow estimation; only the masked pixels are pasted back. Test at 0:25: VRAM 7.0 → 15.6 → 20.3 GB for 208 → 464 → 624 px; no visible gain there once the shot-cut problem was removed | 2 | 0.25 d | P2 |
| Q5 | **Overlapping chunks with cross-fade** of the overlapping frames | 4 | 0.5 d | P2 |
| Q6 | **Inpaint once, propagate**: for pixels never exposed in a shot, inpaint a single keyframe (LaMa, or a diffusion model such as DiffuEraser) and propagate it with the completed flow, so the fill is consistent across the shot | 5 | 1–2 d | P2 |

Q1 + Q2 are expected to remove most visible smearing (static dialogue shots are the most common and the most noticeable case).

## Performance

| # | Item | Expected gain | Effort | Priority |
|---|---|---|---|---|
| P1 | **Horizontal crop**: process only the columns around the subtitle (mask bbox + margin, rounded to 8) instead of the full 1920 px width | ~2× | 0.25 d | ✅ done, offline-verified with a fake engine |
| P2 | **Fewer RAFT iterations** (20 → 10–12); flow is ~30–40% of the time | 20–30% | 0.1 d + quality check | ✅ done (default lowered to 12), **not yet verified on GPU** |
| P3 | **Less redundant work**: smaller context/padding, reuse flow of overlapping context frames | 15–20% | 0.5 d | ⚠️ partial: `CTX`/`PAD` are now CLI-tunable (`--pp-ctx`, `--pp-pad`), defaults unchanged; true flow-reuse across chunks not implemented (needs an `ProPainterEngine` refactor to cache flow between calls) |
| P4 | **Pipelining**: decode, GPU inference and encode in separate threads | 10–20% | 0.5 d | ✅ done (`video.prefetch` / `video.ThreadedWriter`), offline-verified for order/correctness |
| P5 | **Half-resolution mode** (optional flag): inpaint the band at 0.5×, upscale only the inpainted pixels | 2–4× | 0.5 d | P2, trades sharpness |
| P6 | Batch small chunks together; `torch.compile` / TensorRT for the inpainting network | 10–30% | 1 d | P3 |

P1–P4 together should bring ProPainter from ~35–45 min to roughly 10–15 min on an A10 for the test clip. Q2 also saves time: pixels filled from a clean plate need no network inference.

**Status (2026-09-27):** P1, P2 and P4 implemented in `backends/propainter.py` / `video.py` / `pipeline.py`; P3 only partially (tunable knobs, no flow caching). Verified offline with fake engines/generators (chunking, cropping, streaming order all correct); **no GPU run yet** to confirm speed-up or that lowering RAFT iterations to 12 doesn't visibly hurt quality — needs a real A10/3050/5090 run before delivery.

## Suggested order

1. **P1, P2** – quick wins, make every later experiment cheaper
2. **Q2** – biggest remaining visible quality gain (Q1 done)
3. **Q3, Q4, P3, P4**
4. **Q5, Q6, P5, P6** – only if still needed

## Validation

- **Fixed check points**: 0:03, 0:20, 1:53, 2:30, 2:56, 3:20, 4:05, 4:35 of the test clip, compared side by side with the original, the customer's reference output and the previous version. 2:30 (static table) and 1:53 / 4:05 (moving people) are the hardest cases.
- **Subtitle transitions**: all back-to-back subtitle changes (14 in the test clip), since they were the source of a residual-text bug.
- **Residual text**: run OCR (`--srt-only --ocr-interval 1`) on the output; any subtitle found is a miss.
- **Speed and VRAM**: total time and peak VRAM on the A10; VRAM must stay within 8 GB with a reduced `--pp-chunk` for RTX 3050 cards.

## Risks and constraints

- **License**: ProPainter (NTU S-Lab License 1.0) is non-commercial only. For commercial delivery, Q2 and Q6 (with LaMa) can also be built on the LaMa backend, which is Apache-2.0.
- **VRAM**: longer reference windows (Q3) and taller bands (Q4) increase memory; chunk size must be adjusted per GPU.
- **RTX 50 series**: requires a PyTorch build with CUDA 12.8+; not yet tested.
