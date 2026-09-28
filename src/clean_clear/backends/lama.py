"""LaMa backend (ONNX Runtime): single-image inpainting, frame by frame. Fast, Apache-2.0."""
import sys
from pathlib import Path

import cv2
import numpy as np

from ..common import ROOT, log

LAMA_NAME = "lama_fp32.onnx"
LAMA_URLS = [
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
    "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
]


def lama_path():
    for p in [ROOT / "models" / LAMA_NAME, Path.home() / ".cache" / "clean-clear" / LAMA_NAME]:
        if p.exists():
            return p
    dst = ROOT / "models" / LAMA_NAME
    dst.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    for url in LAMA_URLS:
        try:
            log(f"Downloading LaMa model: {url}")
            tmp = dst.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dst)
            return dst
        except Exception as e:
            log(f"  failed: {e}")
    sys.exit(f"Could not download the LaMa model; download {LAMA_NAME} manually into {dst.parent}")


class Lama:
    SIZE = 512

    def __init__(self, providers):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(lama_path()), so, providers=providers)

    def __call__(self, img, mask):
        """img: HxWx3 BGR uint8 (square), mask: HxW bool -> inpainted BGR uint8"""
        h, w = mask.shape
        S = self.SIZE
        a = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA if h > S else cv2.INTER_CUBIC)
        a = a[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        m = cv2.resize(mask.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
        m = m.astype(np.float32)[None, None]
        out = self.sess.run(None, {"image": np.ascontiguousarray(a), "mask": m})[0][0]
        out = np.clip(out.transpose(1, 2, 0), 0, 255).astype(np.uint8)[:, :, ::-1]
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


def inpaint_frame(frame, mask, lama):
    """Inpaint the mask with square windows from left to right; each window uses the previous window's result as context."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return frame
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    T = int(np.clip((y1 - y0) * 2.5, 512, min(H, W)))
    cy = (y0 + y1) // 2
    Y0 = int(np.clip(cy - T // 2, 0, H - T))
    out = frame.copy()
    remain = mask.copy()
    X = max(0, x0 - T // 4)
    while True:
        X = min(X, W - T)
        last = X + T >= min(W, x1 + T // 4)
        region = out[Y0:Y0 + T, X:X + T]
        m = remain[Y0:Y0 + T, X:X + T].copy()
        if not last:
            m[:, 3 * T // 4:] = False
        if m.any():
            res = lama(region, remain[Y0:Y0 + T, X:X + T])
            region[m] = res[m]
            remain[Y0:Y0 + T, X:X + T][m] = False
        if last:
            break
        X += T // 2
    return out


_plan_cache = {}


def _pack_plan(mask, S, ctx, side):
    """Compute the strip layout; the mask is constant within a segment, so cache it per mask object"""
    key = id(mask)
    hit = _plan_cache.get(key)
    if hit is not None and hit[0] is mask:
        return hit[1]
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        plan = None
    else:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        k = S // (y1 - y0 + 2 * ctx)
        if k < 2 or W < S or H < S:
            plan = "square"
        else:
            hs = S // k
            Y0 = int(np.clip((y0 + y1) // 2 - hs // 2, 0, H - hs))
            col = mask[Y0:Y0 + hs].sum(0)
            F = S - 2 * side
            cuts = [x0]
            while x1 - cuts[-1] > F:
                lo = cuts[-1] + F - 96
                cuts.append(lo + int(np.argmin(col[lo:cuts[-1] + F + 1])))
            cuts.append(x1)
            pieces = [(a, b, int(np.clip(a - side, 0, W - S))) for a, b in zip(cuts[:-1], cuts[1:])]
            plan = (k, hs, Y0, pieces)
    _plan_cache.clear()
    _plan_cache[key] = (mask, plan)
    return plan


def inpaint_packed(frame, mask, lama, ctx=24, side=64):
    """A subtitle is a long, thin band: cut it into 512-wide strips and stack them into one 512x512
    image so a single LaMa call inpaints the whole line - 2-4x faster than square windows.
    Cuts are placed at the columns with the fewest mask pixels (gaps between characters), and
    every strip carries `side` pixels of context on each side."""
    plan = _pack_plan(mask, S=lama.SIZE, ctx=ctx, side=side)
    if plan is None:
        return frame
    if plan == "square":
        return inpaint_frame(frame, mask, lama)
    k, hs, Y0, pieces = plan
    S = lama.SIZE
    out = frame.copy()
    band = out[Y0:Y0 + hs]
    mband = mask[Y0:Y0 + hs]
    for j in range(0, len(pieces), k):
        chunk = pieces[j:j + k]
        img = np.zeros((S, S, 3), np.uint8)
        m = np.zeros((S, S), bool)
        for r, (a, b, ws) in enumerate(chunk):
            img[r * hs:(r + 1) * hs] = band[:, ws:ws + S]
            m[r * hs:(r + 1) * hs] = mband[:, ws:ws + S]
        used = len(chunk) * hs
        if used < S:  # pad unused rows with edge pixels
            img[used:] = img[used - 1]
        res = lama(img, m)
        for r, (a, b, ws) in enumerate(chunk):
            sl = slice(a - ws, b - ws)
            mm = mband[:, a:b]
            band[:, a:b][mm] = res[r * hs:(r + 1) * hs, sl][mm]
    return out


class LamaBackend:
    name = "lama"

    def __init__(self, providers):
        self.lama = Lama(providers)

    def erase(self, frames, frame_seg, masks, cuts=()):
        for i, frame in enumerate(frames):
            k = frame_seg.get(i)
            yield inpaint_packed(frame, masks[k], self.lama) if k is not None else frame
