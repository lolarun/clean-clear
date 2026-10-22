"""Make a pilot film: burn the subtitles of an SRT into a clean video, to measure an eraser against the truth.

    python scripts/pilot_make.py CLEAN.mp4 SUBS.srt -o pilot.mp4 --start 300 --duration 180

The customer's pilot (acceptance step 1): a film whose own picture has no subtitles, a few minutes of dialogue, the
subtitles burned in - the picture without them is then the right answer. Writes pilot.mp4 (video only, the part of
CLEAN from --start for --duration seconds with the subtitles on), pilot.srt (times relative to the part) and
pilot.json (the cues and the box of every burned text, for scripts/pilot_eval.py). White text with a black outline,
centred at the bottom, one line.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clean_clear.qc import read_srt  # noqa: E402
from clean_clear.subtitle_qc import find_font  # noqa: E402
from clean_clear.video import ThreadedWriter, open_writer, probe, read_frames  # noqa: E402


def fit_font(path, text, size, max_width):
    while size > 12:
        font = ImageFont.truetype(path, size)
        if ImageDraw.Draw(Image.new("RGB", (4, 4))).textlength(text, font=font) <= max_width:
            return font
        size -= 2
    return ImageFont.truetype(path, 12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clean")
    ap.add_argument("srt")
    ap.add_argument("-o", "--out", default="pilot.mp4")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into CLEAN where the pilot starts")
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--size", type=float, default=0.06, help="text height as a fraction of the picture height")
    ap.add_argument("--crf", type=int, default=16)
    args = ap.parse_args()
    font_path = find_font()
    if not font_path:
        sys.exit("no CJK font found")
    W, H, fps, n = probe(args.clean)
    first, count = int(args.start * fps), int(args.duration * fps)
    cues = [(a - args.start, b - args.start, t.replace("\n", " ")) for a, b, t in read_srt(args.srt)
            if b > args.start and a < args.start + args.duration]
    size = int(H * args.size)
    stroke = max(2, size // 14)
    boxes = []
    for a, b, text in cues:
        font = fit_font(font_path, text, size, 0.88 * W)
        d = ImageDraw.Draw(Image.new("RGB", (4, 4)))
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font, stroke_width=stroke)
        x = (W - (x1 - x0)) // 2 - x0
        y = int(H * 0.93) - (y1 - y0) - y0
        boxes.append({"start": a, "end": b, "text": text, "font": font, "xy": (x, y),
                      "box": [x + x0, y + y0, x + x1, y + y1]})
    out = Path(args.out)
    wr = open_writer(args.clean, out, W, H, fps, "libx264", args.crf, audio=False)
    tw = ThreadedWriter(wr)
    for j, frame in enumerate(read_frames(args.clean, W, H, start=first, count=count)):
        t = j / fps
        cue = next((c for c in boxes if c["start"] <= t < c["end"]), None)
        if cue:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(img).text(cue["xy"], cue["text"], font=cue["font"], fill=(255, 255, 255), stroke_width=stroke,
                                     stroke_fill=(0, 0, 0))
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        tw.write(frame)
    tw.close()
    with open(out.with_suffix(".srt"), "w", encoding="utf-8") as fh:
        for k, c in enumerate(boxes, 1):
            ts = lambda s: f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(round((s % 1) * 1000)):03d}"
            fh.write(f"{k}\n{ts(c['start'])} --> {ts(c['end'])}\n{c['text']}\n\n")
    meta = {"clean": str(args.clean), "first_frame": first, "frames": count, "fps": fps, "size": [W, H],
            "cues": [{k: v for k, v in c.items() if k not in ("font", "xy")} for c in boxes]}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out}: {count} frames, {len(boxes)} cues burned in")


if __name__ == "__main__":
    main()
