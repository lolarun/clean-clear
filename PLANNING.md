# Planning: quality and performance

Roadmap for improving Clean Clear, mainly the ProPainter model (`src/clean_clear/backends/propainter.py`). Effort estimates are rough and speed-ups are untested. See [MEMO.md](MEMO.md) for defect analyses.

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

**Status (2026-09-27):** P1, P2 and P4 implemented in `backends/propainter.py` / `video.py` / `pipeline.py` (all under `src/clean_clear/`); P3 only partially (tunable knobs, no flow caching). Verified offline with fake engines/generators (chunking, cropping, streaming order all correct); **no GPU run yet** to confirm speed-up or that lowering RAFT iterations to 12 doesn't visibly hurt quality — needs a real A10/3050/5090 run before delivery.

## Alternative models (survey, 2026-09-28)

Desk research only; none of these has been run on our clips. Official links were checked on 2026-09-28.

### ProPainter (current)

ICCV 2023, flow-guided propagation + transformer, not a diffusion model. Three networks: **RAFT** (optical flow between frames), **RecurrentFlowCompleteNet** (completes the flow inside the mask), **InpaintGenerator** (propagates real pixels from other frames along the flow, then fills what was never visible with a sparse spatio-temporal transformer). Strength: copies real background when it is exposed elsewhere in the shot, no hallucination risk, relatively fast. Weakness: never-exposed background is soft; fast motion gives bad flow and smears.

### Candidates

| Model | Type | Quality | Efficiency | Availability / licence |
|---|---|---|---|---|
| **ProPainter** (current) | flow + transformer | very good when background is exposed elsewhere; soft otherwise | fast relative to the others; cropped per chunk (P1) | open source, NTU S-Lab 1.0, **non-commercial** |
| **DiffuEraser** (2025) | diffusion (SD-based), uses ProPainter output as prior and refines it | sharper in never-exposed areas | much slower; runs ProPainter first anyway | Apache-2.0, but the bundled ProPainter parts keep ProPainter's licence |
| **MiniMax-Remover** (2025) | video diffusion (6 steps, no CFG), general object removal | strong on object removal | reported ~24 s for 81 frames at 480p, ~14 GB peak VRAM; costly at 1080p over long films | open source |
| **EraserDiT** (2025) | diffusion transformer video inpainting | reported good | "fast" relative to other diffusion methods | paper |
| **SEDiT** (Baidu, arXiv 2605.14894, May 2026) | subtitle-specific, **mask-free**, one-step DiT on LTX-Video-2B + LoRA (381 M trainable) | vs MiniMax-Remover on VSR-Bench-400: PSNR 31.59 vs 28.31, FVD 24.06 vs 39.35, MOS 4.5 vs 2.5 | 1080p, 65 frames in ≈ 4 s on an A800 80 GB (≈ 16 fps; MiniMax-Remover 150 s, DiffuEraser 166 s) | [project page](https://zheng222.github.io/SEDiT_project/) only; **no code or weights**; © Baidu |
| **CLEAR** (ICML 2026, arXiv 2603.21901) | subtitle-specific, **mask-free**, LoRA (rank 64) on Wan2.1-Fun-1.3B, 5 steps, 81-frame sliding window | +6.77 dB PSNR, −74.7% VFID vs ProPainter / DiffuEraser / MiniMax-Remover on Chinese subtitles | **≈ 4.86 s per frame** (reported) — about 1,000 GPU-days for 200 h of video | [code](https://github.com/silent-commit/CLEAR) and [weights](https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal) released, Apache-2.0 but model card says **research purposes only** |

### Cost

Numbers are from the papers / READMEs, not measured by us.

| Model | Speed | VRAM | RTX 3050 (8 GB) | Basis |
|---|---|---|---|---|
| ProPainter (current) | baseline: ≈ 13 fps overall on an A10 for a 720p film (subtitle crop only) | ≈ 13 GB at chunk 120; fits 8 GB with chunk 60–80 | yes | measured |
| MiniMax-Remover | ≈ 3.4 fps at **480p** on an A800 (81 frames in ~24 s); 1080p has 4.5× the pixels and attention cost grows faster than linearly | ≈ 14 GB at 480p; likely > 24 GB at 1080p full frame | practically no | paper |
| SEDiT | ≈ 16 fps at 1080p on an A800 (65 frames in ≈ 4 s), one step; on par with or faster than ProPainter on a comparable GPU | tested on 80 GB; 2 B backbone at 1080p full frame likely > 24 GB | practically no | paper |
| CLEAR | ≈ 4.86 s per frame (5 steps) — two orders of magnitude slower than ProPainter | not reported (1.3 B base model) | unclear | paper / README |

The newer models win on sharpness of never-exposed areas and on not needing masks; they lose on speed and hardware. Realistic use: only on the 5090, or only for shots where ProPainter fails.

### How to apply the subtitle-specific models

**SEDiT** — not usable yet.
- Official: [project page](https://zheng222.github.io/SEDiT_project/), [paper](https://arxiv.org/abs/2605.14894). Authors: Zheng Hui, Yunlong Bai (Baidu Inc.); contact zheng_hui@aliyun.com, baiyunlong@baidu.com.
- No repository, weights or demo are linked (checked 2026-09-28). The only route is to ask the authors about an open-source release or a commercial licence.
- If released: it needs no masks, so it would replace OCR-based masking and the erase step (OCR would still be needed for the SRT). Expected to need a large-VRAM GPU (tested on 80 GB), i.e. the 5090 at best, not the 3050.

**CLEAR** — usable for experiments, not for delivery.
- Official: [code](https://github.com/silent-commit/CLEAR), [weights](https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal), [paper](https://arxiv.org/abs/2603.21901).
- Setup: install [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (torch ≥ 2.4), download the Wan2.1 1.3B base model and the LoRA checkpoint `CLEAR-mask-free-subtitle-removal.pt`, then:

  ```bash
  python inference.py --model_base_path <Wan2.1-Fun-V1.1-1.3B-Control> \
      --lora_checkpoint ./checkpoints/CLEAR-mask-free-subtitle-removal.pt --lora_rank 64 \
      --input_video input.mp4 --output_dir ./results --num_steps 5 --cfg_scale 1.0 --use_sliding_window
  ```

  Sliding window of 81 frames with 16 frames overlap (`--chunk_size`, `--chunk_overlap`).
- Caveats: the README downloads `Wan2.1-T2V-1.3B` but the inference command points at `Wan2.1-Fun-V1.1-1.3B-Control` — check which base model is actually required. Code is Apache-2.0, but the model card says **research purposes only** and the upstream Wan2.1 licence applies. At ≈ 4.86 s per frame, 200 h (≈ 18 M frames) would take ≈ 1,000 GPU-days.

**MiniMax-Remover** — dropped (2026-09-28). It is a general **object** remover trained on object masks, processes at 480×832 (thin subtitle strokes get lost when downscaled and dilated) and still needs our masks, so it has no advantage over ProPainter for subtitles.

### Not applicable

- **NVIDIA DLSS 5** (shipped 2026-09-03, RTX 50 only): a one-step pixel-space diffusion model that adds photoreal lighting and material detail to game frames. It needs engine motion vectors and runs inside game pipelines; there is no SDK for arbitrary video or inpainting. It enhances existing pixels and does not reconstruct occluded content.
- **RTX Video Super Resolution / HDR, Maxine Video Effects**: upscaling, HDR, denoising, background replacement — no inpainting.
- **TensorRT** is not a model but can accelerate ProPainter (see P6).

### Decision

- Keep **ProPainter**: our masks are small, the videos are long and the total volume (≈ 200 h) is throughput-bound; the diffusion models cost several times more compute per frame.
- **Watch SEDiT**: subtitle-specific, mask-free (our glyph-mask and OCR-filter heuristics are the weakest parts) and fast at 1080p, but no code or weights. If released, benchmark it on the fixed check points of both test clips.
- **CLEAR is released but too slow** for 200 h (≈ 4.86 s per frame) and its weights are marked research-only; at most useful as a quality reference on a few shots.
- Speed comes from P6 (TensorRT / `torch.compile`) and Q2 (clean plate for static shots) rather than a model change.

Sources: [ProPainter](https://github.com/sczhou/ProPainter), [DiffuEraser](https://github.com/lixiaowen-xw/diffueraser), [MiniMax-Remover](https://arxiv.org/pdf/2505.24873), [EraserDiT](https://arxiv.org/html/2506.12853v2), [SEDiT](https://arxiv.org/abs/2605.14894) ([project](https://zheng222.github.io/SEDiT_project/)), [CLEAR](https://arxiv.org/abs/2603.21901) ([code](https://github.com/silent-commit/CLEAR), [weights](https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal)), [DLSS 5 (NVIDIA ADLR)](https://research.nvidia.com/labs/adlr/DLSS5/).

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
- **Automated tests**: `pytest` (README §Tests) before every delivery; it checks decoding/encoding (colours, frame counts, variable frame rate), masks, segmentation, caching, `--jobs` merging and the backends' streaming logic with fake models. It does not judge inpainting quality, which still needs the check points above.
- **Duration and colour**: compare output and source duration (logged as a warning when they differ) and spot-check colours on a saturated scene; outputs before 0.3.4 shifted colours on BT.709-tagged sources.

## Risks and constraints

- **License**: ProPainter (NTU S-Lab License 1.0) is non-commercial only. For commercial delivery, Q2 and Q6 (with LaMa) can also be built on the LaMa backend, which is Apache-2.0.
- **VRAM**: longer reference windows (Q3) and taller bands (Q4) increase memory; chunk size must be adjusted per GPU.
- **RTX 50 series**: requires a PyTorch build with CUDA 12.8+; not yet tested.
