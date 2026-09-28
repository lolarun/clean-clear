"""Static watermark detection: pixels that keep the same colour in (almost) every sampled frame."""
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from common import log
from video import _exe


def _grab(src, t, W, H):
    r = subprocess.run([_exe("ffmpeg"), "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
    if len(r.stdout) != W * H * 3:
        return None
    return np.frombuffer(r.stdout, np.uint8).reshape(H, W, 3)


def detect(src, W, H, duration, samples=150, tol=14, consistency=0.85, min_area=20, grow=9):
    """-> HxW bool mask of the watermark(s) found in the four corners, or None.

    Real footage changes from frame to frame while an overlaid logo does not. Frames are sampled across
    the whole video; a pixel belongs to the watermark if it is within `tol` of its median colour in at
    least `consistency` of the samples and sits on an edge of the median image (which excludes flat
    black bars). `grow` is the dilation kernel size covering the anti-aliased outline."""
    cw, ch = min(W // 3, 480), min(H // 4, 200)
    corners = [(0, 0), (W - cw, 0), (0, H - ch), (W - cw, H - ch)]  # top-left, top-right, bottom-left, bottom-right
    times = np.linspace(duration * 0.03, duration * 0.97, samples)
    with ThreadPoolExecutor(4) as ex:
        frames = list(ex.map(lambda t: _grab(src, t, W, H), times))
    frames = [f for f in frames if f is not None]
    if len(frames) < 20:
        log("  watermark: too few frames could be sampled")
        return None

    mask = np.zeros((H, W), np.uint8)
    for x, y in corners:
        stack = np.stack([f[y:y + ch, x:x + cw] for f in frames])
        k = len(stack) // 2
        med = np.partition(stack, k, axis=0)[k]
        near = (np.abs(stack.astype(np.int16) - med.astype(np.int16)).max(-1) < tol).mean(0)
        gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
        edge = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        cand = (near > consistency) & (cv2.dilate(edge, np.ones((3, 3), np.uint8)) > 25)
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        for i in range(1, n):
            if st[i, 4] >= min_area:
                mask[y:y + ch, x:x + cw][lab == i] = 1
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    log(f"  watermark found: x={xs.min()}-{xs.max()} y={ys.min()}-{ys.max()}")
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))).astype(bool)
    if mask.mean() > 0.05:
        log("  watermark: detected area is more than 5% of the frame, ignoring it (probably not a logo)")
        return None
    return mask
