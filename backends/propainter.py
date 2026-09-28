"""ProPainter backend (PyTorch): flow-guided video inpainting using neighbouring frames.

Higher quality on moving shots, much slower than LaMa. Requires PyTorch and a checkout of
https://github.com/sczhou/ProPainter with its weights in <repo>/weights/.
ProPainter is released under the NTU S-Lab License 1.0 - non-commercial use only.

Only the frames that contain subtitles (plus context) and only the horizontal band where the
subtitles appear are inpainted; the result is composited back into the original frames.
Frames are streamed: at most about one chunk of full frames is held in memory.
"""
import bisect
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from common import log


def _ref_index(mid, neighbor_ids, length, ref_stride=10, ref_num=-1):
    """Indices of the non-local reference frames for the transformer (as in ProPainter's inference script)"""
    refs = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                refs.append(i)
    else:
        start = max(0, mid - ref_stride * (ref_num // 2))
        end = min(length, mid + ref_stride * (ref_num // 2))
        for i in range(start, end, ref_stride):
            if i not in neighbor_ids:
                if len(refs) > ref_num:
                    break
                refs.append(i)
    return refs


class ProPainterEngine:
    """ProPainter inference on an in-memory clip; the three models are loaded once."""

    def __init__(self, repo, fp16=True, raft_iter=20, neighbor_length=10, ref_stride=10, subvideo_length=80):
        import torch
        repo = Path(repo).resolve()
        weights = repo / "weights"
        for f in ("ProPainter.pth", "recurrent_flow_completion.pth", "raft-things.pth"):
            if not (weights / f).exists():
                sys.exit(f"ProPainter weight missing: {weights / f} (see README for download links)")
        sys.path.insert(0, str(repo))
        from model.modules.flow_comp_raft import RAFT_bi
        from model.recurrent_flow_completion import RecurrentFlowCompleteNet
        from model.propainter import InpaintGenerator

        if not torch.cuda.is_available():
            log("  warning: CUDA not available, ProPainter will run on CPU (extremely slow)")
            fp16 = False
        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16
        self.raft_iter = raft_iter
        self.neighbor_length = neighbor_length
        self.ref_stride = ref_stride
        self.subvideo_length = subvideo_length
        self.raft = RAFT_bi(str(weights / "raft-things.pth"), self.dev)
        fc = RecurrentFlowCompleteNet(str(weights / "recurrent_flow_completion.pth"))
        for p in fc.parameters():
            p.requires_grad = False
        self.fc = fc.to(self.dev).eval()
        self.model = InpaintGenerator(model_path=str(weights / "ProPainter.pth")).to(self.dev).eval()
        if fp16:
            self.fc = self.fc.half()
            self.model = self.model.half()

    def __call__(self, frames, masks, dilation=2):
        """frames: [HxWx3 BGR uint8] (H and W multiples of 8); masks: [HxW bool] -> [HxWx3 BGR uint8]"""
        with self.torch.no_grad():
            if len(frames) == 1:  # flow needs two frames (e.g. a one-frame shot): duplicate it
                return self._run(frames * 2, masks * 2, dilation)[:1]
            return self._run(frames, masks, dilation)

    def _run(self, frames, masks, dilation):
        torch, dev = self.torch, self.dev
        T = len(frames)
        h, w = frames[0].shape[:2]
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        m = np.stack([cv2.dilate(mk.astype(np.uint8), cross, iterations=dilation) if dilation else mk.astype(np.uint8)
                      for mk in masks])
        ori = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]  # RGB
        x = torch.from_numpy(np.stack(ori)).to(dev).permute(0, 3, 1, 2).float().div_(127.5).sub_(1).unsqueeze(0)
        masks_dilated = torch.from_numpy(m).to(dev).float()[None, :, None]
        flow_masks = masks_dilated

        # ---- optical flow (RAFT, fp32) ----
        short = 12 if w <= 640 else 8 if w <= 720 else 4 if w <= 1280 else 2
        if T > short:
            ff, fb = [], []
            for f in range(0, T, short):
                e = min(T, f + short)
                a, b = self.raft(x[:, f:e] if f == 0 else x[:, f - 1:e], iters=self.raft_iter)
                ff.append(a)
                fb.append(b)
            gt = (torch.cat(ff, 1), torch.cat(fb, 1))
        else:
            gt = self.raft(x, iters=self.raft_iter)
        self._empty_cache()

        if self.fp16:
            x, flow_masks, masks_dilated = x.half(), flow_masks.half(), masks_dilated.half()
            gt = (gt[0].half(), gt[1].half())

        # ---- flow completion ----
        L = gt[0].size(1)
        sv = self.subvideo_length
        if L > sv:
            pf, pb = [], []
            pad = 5
            for f in range(0, L, sv):
                s, e = max(0, f - pad), min(L, f + sv + pad)
                ps, pe = f - s, e - min(L, f + sv)
                sub = (gt[0][:, s:e], gt[1][:, s:e])
                pred, _ = self.fc.forward_bidirect_flow(sub, flow_masks[:, s:e + 1])
                pred = self.fc.combine_flow(sub, pred, flow_masks[:, s:e + 1])
                pf.append(pred[0][:, ps:e - s - pe])
                pb.append(pred[1][:, ps:e - s - pe])
            pred_flows = (torch.cat(pf, 1), torch.cat(pb, 1))
        else:
            pred, _ = self.fc.forward_bidirect_flow(gt, flow_masks)
            pred_flows = self.fc.combine_flow(gt, pred, flow_masks)
        del gt
        self._empty_cache()

        # ---- image propagation ----
        masked = x * (1 - masks_dilated)
        svp = min(100, sv)
        if T > svp:
            uf, um = [], []
            pad = 10
            for f in range(0, T, svp):
                s, e = max(0, f - pad), min(T, f + svp + pad)
                ps, pe = f - s, e - min(T, f + svp)
                b, t = masks_dilated[:, s:e].shape[:2]
                sub = (pred_flows[0][:, s:e - 1], pred_flows[1][:, s:e - 1])
                prop, upd = self.model.img_propagation(masked[:, s:e], sub, masks_dilated[:, s:e], "nearest")
                fr = x[:, s:e] * (1 - masks_dilated[:, s:e]) + prop.view(b, t, 3, h, w) * masks_dilated[:, s:e]
                uf.append(fr[:, ps:e - s - pe])
                um.append(upd.view(b, t, 1, h, w)[:, ps:e - s - pe])
            updated_frames, updated_masks = torch.cat(uf, 1), torch.cat(um, 1)
        else:
            b, t = masks_dilated.shape[:2]
            prop, upd = self.model.img_propagation(masked, pred_flows, masks_dilated, "nearest")
            updated_frames = x * (1 - masks_dilated) + prop.view(b, t, 3, h, w) * masks_dilated
            updated_masks = upd.view(b, t, 1, h, w)
        self._empty_cache()

        # ---- feature propagation + transformer ----
        comp = [None] * T
        stride = self.neighbor_length // 2
        ref_num = sv // self.ref_stride if T > sv else -1
        bm_all = masks_dilated[0, :, 0].bool()
        for f in range(0, T, stride):
            nb = list(range(max(0, f - stride), min(T, f + stride + 1)))
            ids = nb + _ref_index(f, nb, T, self.ref_stride, ref_num)
            pflow = (pred_flows[0][:, nb[:-1]], pred_flows[1][:, nb[:-1]])
            pred = self.model(updated_frames[:, ids], pflow, masks_dilated[:, ids], updated_masks[:, ids], len(nb))
            pred = ((pred.view(-1, 3, h, w) + 1) / 2 * 255).float().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            bms = bm_all[nb].cpu().numpy()
            for j, idx in enumerate(nb):
                img = ori[idx].copy()
                img[bms[j]] = pred[j][bms[j]]
                if comp[idx] is None:
                    comp[idx] = img
                else:
                    comp[idx] = ((comp[idx].astype(np.float32) + img.astype(np.float32)) * 0.5).astype(np.uint8)
        self._empty_cache()
        return [np.ascontiguousarray(c[:, :, ::-1]) for c in comp]

    def _empty_cache(self):
        if self.dev.type == "cuda":
            self.torch.cuda.empty_cache()


class ProPainterBackend:
    name = "propainter"
    uses_cuts = True  # the pipeline passes shot cuts to erase()
    MIN_SIDE = 128    # smallest band height / crop width (px) handed to the models

    def __init__(self, repo, chunk=120, raft_iter=20, ctx=10, pad=8, margin=80):
        if not repo or not Path(repo).is_dir():
            sys.exit("ProPainter needs --propainter-dir pointing to a ProPainter checkout (see README)")
        self.chunk = chunk
        self.CTX = ctx    # context frames added on each side of a chunk
        self.PAD = pad    # extra frames inpainted around each run of subtitle frames
        self.margin = margin  # horizontal margin (px) kept around the subtitle when cropping columns
        self.engine = ProPainterEngine(repo, raft_iter=raft_iter, subvideo_length=chunk + 2 * self.CTX)

    def _chunks(self, frame_seg, cuts=()):
        """Frames to inpaint -> runs (gaps <= 10 frames merged, PAD frames added at both ends)
        -> chunks (s, e, cs, ce): output frames s..e, processed with context frames cs..ce.

        Nothing crosses a shot cut: ProPainter treats a chunk as one continuous shot and would copy
        pixels from the other shot into the masked area (seen as dark, flickering blobs)."""
        cuts = sorted(cuts)
        end = 1 << 60  # open end of the last shot (the real frame count is only known while streaming)

        def shot(i):
            return bisect.bisect_right(cuts, i)

        def bounds(sh):
            return (cuts[sh - 1] if sh > 0 else 0), (cuts[sh] - 1 if sh < len(cuts) else end)

        runs = []
        for i in sorted(frame_seg):
            if runs and i - runs[-1][1] <= 10 and shot(i) == runs[-1][2]:
                runs[-1][1] = i
            else:
                runs.append([i, i, shot(i)])
        chunks = []
        for a, b, sh in runs:
            lo, hi = bounds(sh)
            a = max(lo, a - self.PAD)
            b = min(hi, b + self.PAD)
            for s in range(a, b + 1, self.chunk):
                e = min(b, s + self.chunk - 1)
                chunks.append((s, e, max(lo, s - self.CTX), min(hi, e + self.CTX)))
        return chunks

    def erase(self, frames, frame_seg, masks, cuts=()):
        if not frame_seg:
            yield from frames
            return
        H, W = next(iter(masks.values())).shape
        rows = np.nonzero(np.any([m.any(1) for m in masks.values()], 0))[0]
        # RAFT's feature extractor needs both crop dimensions to be exact multiples of 8; clamping
        # r1 to H (frame height) instead of a multiple of 8 breaks that when H itself isn't one
        # (e.g. 714px), which shows up as a batch-size mismatch deep inside RAFT's correlation lookup.
        H8 = H // 8 * 8
        lo = max(0, int(rows.min()) - 40)
        hi = min(H, int(rows.max()) + 40)
        height = (hi - lo + 7) // 8 * 8
        r0 = lo // 8 * 8
        if r0 + height > H8:
            r0 = max(0, H8 - height)
        r1 = min(H8, r0 + height)
        if r1 - r0 < self.MIN_SIDE:  # tiny regions (e.g. a corner logo) still need some context
            r1 = min(H8, r0 + self.MIN_SIDE)
            r0 = max(0, r1 - self.MIN_SIDE) // 8 * 8
        kern = np.ones((9, 9), np.uint8)
        # composite only within the mask dilated by 4 px
        paste = {k: cv2.dilate(m[r0:r1].astype(np.uint8), kern).astype(bool) for k, m in masks.items()}

        pending = deque(self._chunks(frame_seg, cuts))
        n_chunks = len(pending)
        need = set()
        for _, _, cs, ce in pending:
            need.update(range(cs, ce + 1))
        log(f"  propainter: band y={r0}-{r1}, {n_chunks} chunks")
        buf = {}    # frames not yet emitted
        bands = {}  # original bands (full row-band width) still needed as chunk input
        nxt = 0

        def band_of(f):
            return f[r0:r1].copy()

        def crop_cols(ids):
            """Horizontal crop (columns around the subtitle, padded to a multiple of 8): a chunk's
            subtitle is usually much narrower than the frame, so this keeps the engine's per-frame
            work proportional to the text width instead of the full video width."""
            keys = {frame_seg[i] for i in ids if frame_seg.get(i) is not None}
            if not keys:
                return 0, W
            cols = np.zeros(W, bool)
            for k in keys:
                cols |= masks[k][r0:r1].any(0)
            xs = np.nonzero(cols)[0]
            c0 = int(max(0, (xs.min() - self.margin) // 8 * 8))
            c1 = int(min(W, c0 + ((xs.max() + self.margin - c0 + 7) // 8) * 8))
            if c1 - c0 < self.MIN_SIDE:
                c1 = min(W, c0 + self.MIN_SIDE)
                c0 = max(0, c1 - self.MIN_SIDE) // 8 * 8
            return c0, c1

        def run(chunk, last):
            s, e, cs, ce = chunk
            ids = [i for i in range(cs, min(ce, last) + 1) if i in bands]
            if not ids:
                return
            c0, c1 = crop_cols(ids)
            w, h = c1 - c0, r1 - r0
            w8, h8 = (w + 7) // 8 * 8, (h + 7) // 8 * 8  # the models need multiples of 8
            empty = np.zeros((h8, w8), bool)

            def crop_frame(a):
                a = a[:, c0:c1]
                return np.pad(a, ((0, h8 - h), (0, w8 - w), (0, 0)), mode="edge") if (h8, w8) != (h, w) else a.copy()

            def crop_mask(a):
                a = a[:, c0:c1]
                return np.pad(a, ((0, h8 - h), (0, w8 - w))) if (h8, w8) != (h, w) else a.copy()

            ms = []
            for i in ids:
                k = frame_seg.get(i)
                ms.append(empty if k is None else crop_mask(masks[k][r0:r1]))
            res = self.engine([crop_frame(bands[i]) for i in ids], ms)
            for j, i in enumerate(ids):
                k = frame_seg.get(i)
                if s <= i <= e and k is not None and i in buf:
                    b = buf[i][r0:r1, c0:c1]
                    b[paste[k][:, c0:c1]] = res[j][:h, :w][paste[k][:, c0:c1]]
            log(f"  [propainter] chunk {n_chunks - len(pending)}/{n_chunks}")

        last = -1
        for last, f in enumerate(frames):
            i = last
            buf[i] = f.copy() if i in frame_seg else f
            if i in need:
                bands[i] = band_of(f)
            while pending and i >= pending[0][3]:
                run(pending.popleft(), i)
                lo = pending[0][2] if pending else i + 1
                for j in [j for j in bands if j < lo]:
                    del bands[j]
            limit = pending[0][0] - 1 if pending else i
            while nxt <= min(limit, i):
                yield buf.pop(nxt)
                nxt += 1
        while pending:  # the video ended before the last chunk's context was complete
            run(pending.popleft(), last)
        while nxt <= last:
            yield buf.pop(nxt)
            nxt += 1
