"""Compose the result in the source's own YUV: pixels that were not erased are never converted through RGB.

The erasing works on BGR frames. Decoding to BGR and encoding back changes every pixel, also the ones that were not
touched: many YCbCr values have no RGB counterpart (they lie outside the RGB cube) and are clipped on the way, which
does not come back. On 25 s of 1304 that was 43.2 dB against the source for the top 60 % of the picture (38.3 dB at
the worst frame, with or without erasing, CRF 18 or 16; a plain re-encode of the source: 49.5 dB), while the
delivery requirement is 40 dB or more outside the erased area. Here a frame without erased pixels is written as the
decoder gave it, and a frame with erased pixels gets only those pixels (and the 2x2 chroma blocks they touch) from the
erased BGR frame, converted to YUV with the source's matrix.
"""
import cv2
import numpy as np

# (Kr, Kb) of the matrices video.colour() names
COEFFS = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114), "bt2020": (0.2627, 0.0593), "smpte240m": (0.212, 0.087),
          "fcc": (0.30, 0.11)}
GROW = 5   # pixels around an erase mask that the erasing may change (ProPainter pastes within 4 px, plus one for rounding)


def supported(matrix, pix_fmt, range_, w, h):
    """Can the result be composed in YUV: 8-bit 4:2:0 in the limited range, a matrix we know, even picture size"""
    return pix_fmt == "yuv420p" and range_ in ("tv", "limited", None) and matrix in COEFFS and w % 2 == 0 and h % 2 == 0


def region_of(mask_crop, grow=GROW):
    """(y0, x0, bool crop) of an erase mask -> the same grown by `grow` pixels: where the erasing may have changed pixels"""
    y0, x0, c = mask_crop
    if not c.size:
        return y0, x0, c
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    pad = grow
    padded = np.pad(c.astype(np.uint8), pad)
    return y0 - pad, x0 - pad, cv2.dilate(padded, k).astype(bool)


def bgr_to_yuv420(bgr, matrix):
    """BGR uint8 (even height and width) -> Y (h, w), U and V (h/2, w/2) uint8, limited range, 2x2 mean for the chroma"""
    kr, kb = COEFFS[matrix]
    kg = 1.0 - kr - kb
    f = bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    y = kr * r + kg * g + kb * b
    cb = (b - y) / (2 * (1 - kb))
    cr = (r - y) / (2 * (1 - kr))
    h, w = y.shape
    sub = lambda a: a.reshape(h // 2, 2, w // 2, 2).mean((1, 3))
    yy = np.clip(np.rint(16 + 219 / 255 * y), 0, 255).astype(np.uint8)
    uu = np.clip(np.rint(128 + 224 / 255 * sub(cb)), 0, 255).astype(np.uint8)
    vv = np.clip(np.rint(128 + 224 / 255 * sub(cr)), 0, 255).astype(np.uint8)
    return yy, uu, vv


def compose(yuv, bgr, regions, matrix, W, H):
    """yuv: the source frame as the decoder gave it (flat uint8, W*H*3/2); bgr: the same frame after erasing;
    regions: [(y0, x0, bool crop)] where pixels may differ. -> bytes of the result frame in yuv420p"""
    if not regions or all(not r[2].any() for r in regions):
        return yuv.tobytes()
    full = np.zeros((H, W), bool)
    for y0, x0, c in regions:
        ya, xa = max(0, y0), max(0, x0)
        yb, xb = min(H, y0 + c.shape[0]), min(W, x0 + c.shape[1])
        if yb > ya and xb > xa:
            full[ya:yb, xa:xb] |= c[ya - y0:yb - y0, xa - x0:xb - x0]
    ys, xs = np.nonzero(full.any(1))[0], np.nonzero(full.any(0))[0]
    if not len(ys):
        return yuv.tobytes()
    y0, y1 = ys[0] // 2 * 2, (ys[-1] // 2 + 1) * 2          # the box, on the chroma grid
    x0, x1 = xs[0] // 2 * 2, (xs[-1] // 2 + 1) * 2
    out = yuv.copy()
    Y = out[:W * H].reshape(H, W)
    U = out[W * H:W * H + W * H // 4].reshape(H // 2, W // 2)
    V = out[W * H + W * H // 4:].reshape(H // 2, W // 2)
    ny, nu, nv = bgr_to_yuv420(bgr[y0:y1, x0:x1], matrix)
    box = full[y0:y1, x0:x1]
    Yb = Y[y0:y1, x0:x1]
    Yb[box] = ny[box]
    blocks = box.reshape((y1 - y0) // 2, 2, (x1 - x0) // 2, 2).any((1, 3))   # chroma samples that an erased pixel touches
    Ub, Vb = U[y0 // 2:y1 // 2, x0 // 2:x1 // 2], V[y0 // 2:y1 // 2, x0 // 2:x1 // 2]
    Ub[blocks] = nu[blocks]
    Vb[blocks] = nv[blocks]
    return out.tobytes()
