"""Measure an eraser against the truth of a pilot film (scripts/pilot_make.py).

    python scripts/pilot_eval.py pilot.json RESULT.mp4 [-o report.json]

PSNR of the luma between the result and the clean picture the subtitles were burned into, on every 3rd frame:
  outside   the top 60 % of the picture, every frame (the customer's limit: 40 dB or more outside the erased area)
  clean_band the subtitle band (bottom 40 %) in frames without a subtitle
  band_rest  the subtitle band outside the burned text's box (+ margin), in frames with a subtitle
  box        inside the box of the burned text (+ margin): how good the fill is
and the frames in which the burned text can still be read (the share of white pixels with a dark ring that the result
has inside the box and the truth has not).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clean_clear.masks import outline_score  # noqa: E402
from clean_clear.video import read_frames  # noqa: E402

MARGIN = 14


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def stats(v):
    v = np.array(v) if len(v) else np.array([float("nan")])
    return {"n": len(v), "mean": round(float(np.mean(v)), 2), "min": round(float(np.min(v)), 2),
            "p5": round(float(np.percentile(v, 5)), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meta")
    ap.add_argument("result")
    ap.add_argument("-o", "--out")
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    W, H = meta["size"]
    fps, first, count = meta["fps"], meta["first_frame"], meta["frames"]
    cues = meta["cues"]
    top, band0 = int(H * 0.6), int(H * 0.6)
    res = {"outside": [], "clean_band": [], "band_rest": [], "box": []}
    readable, with_cue = 0, 0
    truth = read_frames(meta["clean"], W, H, start=first, count=count)
    for j, (t, r) in enumerate(zip(truth, read_frames(args.result, W, H, count=count))):
        if j % args.stride:
            continue
        res["outside"].append(psnr(t[:top], r[:top]))
        time = j / fps
        cue = next((c for c in cues if c["start"] <= time < c["end"]), None)
        if cue is None:
            res["clean_band"].append(psnr(t[band0:], r[band0:]))
            continue
        with_cue += 1
        x0, y0, x1, y1 = cue["box"]
        x0, y0, x1, y1 = max(0, x0 - MARGIN), max(band0, y0 - MARGIN), min(W, x1 + MARGIN), min(H, y1 + MARGIN)
        res["box"].append(psnr(t[y0:y1, x0:x1], r[y0:y1, x0:x1]))
        keep = np.ones((H - band0, W), bool)
        keep[y0 - band0:y1 - band0, x0:x1] = False
        a, b = t[band0:][keep], r[band0:][keep]
        res["band_rest"].append(psnr(a, b))
        if outline_score(r[y0:y1, x0:x1]) > 0.25 and outline_score(t[y0:y1, x0:x1]) <= 0.25:
            readable += 1
    out = {k: stats(v) for k, v in res.items()}
    out["subtitle_frames_sampled"] = with_cue
    out["frames_where_text_is_still_readable"] = readable
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
