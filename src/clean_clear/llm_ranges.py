"""--suggest-keep: where do the opening and closing credits of a film end and begin? A suggestion for --keep-ranges.

The subtitles are erased where the dialogue is; the credits before the first and after the last line of dialogue (names,
logos, titles, scrolling lists, lyrics in the picture) must stay as they are. The tool cannot tell them from subtitles, a
person can. This asks a language model that speaks the OpenAI chat-completions protocol, in two steps:
  1. the first and last cues of the SRT (the tool's own, or any other) -> the numbers of the first and the last cue that is
     spoken dialogue;
  2. frames of the source around both boundaries -> does the picture agree (dialogue subtitle after, none before)?
It only suggests: the ranges, the frames it looked at and its reasons are written next to the video for a person to check,
and nothing is erased. The model sees the SRT text of the head and the tail of the film and a few of its frames: the
configuration (.env) decides which service gets them.
"""
import base64
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import ROOT, log
from .qc import read_srt
from .video import _exe, probe

WINDOW = 15 * 60.0    # s: the model reads the cues of the first and of the last WINDOW seconds of the film
MARGIN = 5.0          # s between a protected range and the dialogue. The OCR end of a subtitle can be several seconds early
#                       (Tianyan: 3.7 s, a bright background made the outline test drop the last frames), and a range that
#                       eats the last words of the dialogue leaves a subtitle on screen; credits start much later
HEAD_SHOTS = (-20, -10, -5, -2.5, -1, 1, 2.5, 5, 10)                 # s around the start of the first dialogue cue
TAIL_SHOTS = (-10, -5, -2.5, -1, 1, 2.5, 5, 7.5, 10, 15, 20, 30)     # s around the end of the last one
IMAGE_WIDTH = 960
KEYS = {"base_url": ("LLM_BASE_URL", "OPENAI_BASE_URL"), "key": ("LLM_API_KEY", "OPENAI_API_KEY"),
        "model": ("LLM_MODEL", "OPENAI_MODEL"), "vision_model": ("LLM_VISION_MODEL",), "timeout": ("LLM_TIMEOUT",)}


def parse_env(text):
    """KEY=VALUE lines (# comments, 'export ', quotes) -> dict"""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.removeprefix("export ").partition("=")
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        elif " #" in v:
            v = v.split(" #", 1)[0].rstrip()
        out[k.strip()] = v
    return out


def load_config(env_file=None):
    """The service to ask: from env_file (default: .env in the working directory, else next to the tool), a real
    environment variable of the same name wins. -> {base_url, key, model, vision_model, timeout}"""
    if env_file and not Path(env_file).is_file():
        raise SystemExit(f"--llm-env: {env_file} does not exist")
    values = {}
    for f in [Path(env_file)] if env_file else [Path(".env"), ROOT / ".env"]:
        if f.is_file():
            values = parse_env(f.read_text(encoding="utf-8-sig"))
            break
    cfg = {}
    for name, aliases in KEYS.items():
        cfg[name] = next((os.environ[a] for a in aliases if os.environ.get(a)), None) or \
            next((values[a] for a in aliases if values.get(a)), "")
    missing = [KEYS[k][0] for k in ("base_url", "key", "model") if not cfg[k]]
    if missing:
        raise SystemExit("--suggest-keep needs " + ", ".join(missing) + " in the .env file (see .env.example) or the environment")
    cfg["vision_model"] = cfg["vision_model"] or cfg["model"]
    cfg["timeout"] = float(cfg["timeout"] or 120)
    return cfg


def _redact(text, cfg):
    return text.replace(cfg["key"], "[key]") if cfg.get("key") else text


def _post(cfg, payload):
    """One chat-completions request -> the response as a dict; retries rate limits, server errors and timeouts"""
    url = cfg["base_url"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json",
                                                                       "Authorization": "Bearer " + cfg["key"]})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=cfg["timeout"]) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {_redact(e.read().decode('utf-8', 'replace')[:300], cfg)}"
            if e.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = _redact(str(e), cfg)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"the language model service did not answer: {last}")


def extract_json(text):
    """The JSON object of a model reply, also when it is wrapped in a code fence or talks around it"""
    text = (text or "").strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < a:
        raise ValueError(f"no JSON in the reply: {text[:200]!r}")
    return json.loads(text[a:b + 1])


def chat_json(cfg, content, model):
    """content: the parts of one user message (text and image_url parts) -> the JSON object the model answered with"""
    body = _post(cfg, {"model": model, "temperature": 0, "messages": [{"role": "user", "content": content}]})
    return extract_json(body["choices"][0]["message"]["content"])


def hms(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def cue_lines(cues, numbers):
    return "\n".join(f"#{k} [{hms(cues[k - 1][0])} - {hms(cues[k - 1][1])}] {cues[k - 1][2].replace(chr(10), ' / ')}"
                     for k in numbers)


TEXT_PROMPT = """You help a film-processing team. A tool read the hardcoded subtitles of a film with OCR and wrote them as an SRT; below are the cues of its first {window} minutes and of its last {window} minutes (the middle is left out), numbered. The film is {duration} long; the SRT has {count} cues.

Subtitles are the ordinary lines at the bottom centre that go with what is heard: spoken dialogue, narration or voice-over (also at the very start of a film), sung lyrics shown as subtitles. These are NOT subtitles: names and job titles of the opening and closing credits, company names and logos, title cards, English translations of credits, scrolling lists, lyrics that are part of the picture, OCR noise (one or two odd letters, a line that flickers for less than a second).

Find the FIRST cue and the LAST cue that are subtitles in this sense (the JSON keys keep the word dialogue). If the film has no such cue at all, answer null.

{lines}

Answer only with JSON: {{"first_dialogue_cue": <number or null>, "last_dialogue_cue": <number or null>, "reason": "<one or two sentences>"}}"""

VISION_PROMPT = """Frames of a film, with their times. We believe the {which} spoken-dialogue subtitle {rule} at {t}. A dialogue subtitle is an ordinary line at the bottom centre of the picture that goes with what is heard: spoken words, narration, voice-over, sung lyrics. Credits (names, job titles, company names), logos, title cards, scrolling lists and lyrics that are part of the picture are other text, not dialogue.

List EVERY frame, in the order given, with its time t exactly as given, and say for each whether a dialogue subtitle is visible in it (a subtitle that is on screen in the frame counts even if the boundary says otherwise) and what other text is visible. Then say whether the frames agree with the boundary: {agree_rule}. If they do not, say where the boundary is instead (seconds), if the frames show it.

Answer only with JSON: {{"frames": [{{"t": <seconds>, "dialogue_subtitle": <true|false>, "other_text": "<none|credits|title|logo|picture text|lyrics|other>"}}], "boundary_ok": <true|false>, "better_time": <seconds or null>, "reason": "<one or two sentences>"}}"""


def grab(video, t, width=IMAGE_WIDTH):
    """JPEG bytes of the frame at t seconds"""
    r = subprocess.run([_exe("ffmpeg"), "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(video), "-frames:v", "1",
                        "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"],
                       capture_output=True)
    if not r.stdout:
        raise RuntimeError(f"cannot read a frame of {video} at {t:.1f} s")
    return r.stdout


def confirm(cfg, video, out_dir, side, boundary, offsets, duration):
    """The frames around `boundary` -> {frames: [files], boundary_ok, better_time, reason}; the frames are kept in out_dir"""
    times = sorted({min(max(0.0, boundary + o), duration - 0.1) for o in offsets})
    stem = Path(video).stem
    content, files = [], []
    first = side == "head"
    content.append({"type": "text", "text": VISION_PROMPT.format(
        which="first" if first else "last", t=hms(boundary), rule="starts" if first else "ends",
        agree_rule=("no dialogue subtitle before it, only credits, logos or titles, and a dialogue subtitle right after it"
                    if first else "a dialogue subtitle just before it, and after it only credits, logos, lyrics or titles"))})
    for t in times:
        jpg = grab(video, t)
        name = f"{stem}.keep-{side}-{t:07.1f}s.jpg"
        (out_dir / name).write_bytes(jpg)
        files.append(name)
        content.append({"type": "text", "text": f"Frame at {hms(t)} ({t:.2f} s):"})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}})
    answer = chat_json(cfg, content, cfg["vision_model"])
    return {"frames": files, "times": times, "dialogue": flags_of(times, answer.get("frames")),
            "boundary_ok": answer.get("boundary_ok") is True, "better_time": answer.get("better_time"),
            "reason": str(answer.get("reason", ""))}


def flags_of(times, seen):
    """What the model said per frame: [True | False | None (not said)] for each of `times`, matched by the frame's time"""
    out = [None] * len(times)
    for x in seen if isinstance(seen, list) else []:
        t, v = (x.get("t"), x.get("dialogue_subtitle")) if isinstance(x, dict) else (None, None)
        if isinstance(t, (int, float)) and isinstance(v, bool):
            i = min(range(len(times)), key=lambda k: abs(times[k] - t))
            if abs(times[i] - t) <= 0.6:
                out[i] = v
    return out


def refine(side, boundary, times, dialogue):
    """The boundary the frames show. The SRT end of the last dialogue can be early: if a frame after it still shows a
    dialogue subtitle, the end is the first sampled frame after the last such frame (the subtitle is gone by then). The same
    for the start of the first dialogue, which can be late: the sampled frame before the first frame that shows one."""
    shows = [t for t, d in zip(times, dialogue) if d is True]
    if side == "tail":
        late = [t for t in shows if t > boundary]
        if not late:
            return boundary
        after = [t for t in times if t > max(late)]
        return min(after) if after else max(late)
    early = [t for t in shows if t < boundary]
    if not early:
        return boundary
    before = [t for t in times if t < min(early)]
    return max(before) if before else max(0.0, min(early) - 1.0)


def windowed(cues, duration, window):
    """Numbers (1-based) of the cues that start in the first `window` seconds or end in the last `window` seconds"""
    if duration <= 2 * window:
        return list(range(1, len(cues) + 1))
    return [k for k, (a, b, _) in enumerate(cues, 1) if a < window or b > duration - window]


def suggest_keep(video, srt, out_dir, cfg, window=WINDOW, margin=MARGIN):
    """-> dict(keep_ranges, head, tail, warnings, needs_review); writes NAME.keep.json and the frames into out_dir"""
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cues = read_srt(srt)
    W, H, fps, n = probe(video)
    duration = n / fps
    if not cues:
        raise SystemExit(f"{srt} has no cues: nothing to judge")
    shown = windowed(cues, duration, window)
    log(f"  asking {cfg['model']} about the first and the last {window / 60:g} minutes: {len(shown)} of {len(cues)} cues")
    answer = chat_json(cfg, [{"type": "text", "text": TEXT_PROMPT.format(
        duration=hms(duration), window=f"{window / 60:g}", count=len(cues), lines=cue_lines(cues, shown))}], cfg["model"])
    res = {"video": str(video), "srt": str(srt), "duration": round(duration, 3), "model": cfg["model"],
           "text_answer": answer, "warnings": [], "head": None, "tail": None}
    pick = {}
    for side, key in (("head", "first_dialogue_cue"), ("tail", "last_dialogue_cue")):
        k = answer.get(key)
        if k is None:
            continue
        if not isinstance(k, int) or k not in shown:
            res["warnings"].append(f"{key} = {k!r} is not one of the cues that were sent: ignored")
            continue
        pick[side] = k
    if "head" in pick:
        t = cues[pick["head"] - 1][0]
        res["head"] = {"cue": pick["head"], "text": cues[pick["head"] - 1][2], "boundary": round(t, 3)}
        if t > 0.25 * duration:
            res["warnings"].append(f"the first dialogue is at {hms(t)}, more than a quarter into the film: check")
    if "tail" in pick:
        t = cues[pick["tail"] - 1][1]
        res["tail"] = {"cue": pick["tail"], "text": cues[pick["tail"] - 1][2], "boundary": round(t, 3)}
        if t < 0.75 * duration:
            res["warnings"].append(f"the last dialogue is at {hms(t)}, less than three quarters into the film: check")
    if not pick:
        res["warnings"].append(f"the model found no dialogue subtitle in the first and the last {window / 60:g} minutes: "
                               "a film without subtitles, a dialogue that lies outside them, or a bad SRT")
    else:
        for side in ("head", "tail"):
            if side not in pick:
                res["warnings"].append(f"no dialogue found at the {side} of the film (within {window / 60:g} minutes): "
                                       "no range is suggested there")
    ranges = []
    for side, offsets in (("head", HEAD_SHOTS), ("tail", TAIL_SHOTS)):
        s = res[side]
        if not s:
            continue
        for scale in (1, 2, 4, 8):
            s["check"] = confirm(cfg, video, out_dir, side, s["boundary"], [o * scale for o in offsets], duration)
            s["check"]["scale"] = scale
            times, said = s["check"]["times"], s["check"]["dialogue"]
            edge = 0 if side == "head" else -1     # the outermost frame still shows a subtitle: the boundary lies further out
            if said[edge] is not True or (times[edge] < 0.5 if side == "head" else times[edge] > duration - 0.5):
                break
        b = refine(side, s["boundary"], s["check"]["times"], s["check"]["dialogue"])
        if abs(b - s["boundary"]) > 0.01:
            res["warnings"].append(f"the SRT time of the {side} boundary ({hms(s['boundary'])}) is wrong: the frames show a "
                                   f"dialogue subtitle {'until' if side == 'tail' else 'from'} {hms(b)}; that is used")
            s["srt_boundary"], s["boundary"] = s["boundary"], round(b, 3)
        elif not s["check"]["boundary_ok"]:
            res["warnings"].append(f"the frames do not confirm the {side} boundary: {s['check']['reason']}"
                                   + (f" (the frames suggest {s['check']['better_time']} s)"
                                      if s["check"]["better_time"] is not None else ""))
        if side == "head" and s["boundary"] - margin > 1.0:
            ranges.append(f"0-{s['boundary'] - margin:.3f}")
            s["keep"] = [0.0, round(s["boundary"] - margin, 3)]
        if side == "tail" and s["boundary"] + margin < duration - 1.0:
            ranges.append(f"{s['boundary'] + margin:.3f}-end")
            s["keep"] = [round(s["boundary"] + margin, 3), round(duration, 3)]
    res["margin"] = margin
    res["keep_ranges"] = ",".join(ranges)
    res["needs_review"] = bool(res["warnings"])
    (out_dir / f"{video.stem}.keep.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def report(res, out_dir):
    h, t = res["head"], res["tail"]
    log("  head: " + (f"first dialogue #{h['cue']} at {hms(h['boundary'])} {h['text']!r}" if h else "none"))
    log("  tail: " + (f"last dialogue #{t['cue']} ends {hms(t['boundary'])} {t['text']!r}" if t else "none"))
    for side in ("head", "tail"):
        if res[side] and "check" in res[side]:
            c = res[side]["check"]
            log(f"  {side} frames: {'confirm' if c['boundary_ok'] else 'DO NOT confirm'} - {c['reason']}")
    for w in res["warnings"]:
        log(f"  CHECK: {w}")
    log(f"  suggestion: --keep-ranges \"{res['keep_ranges']}\""
        + (f"   (look at the frames in {out_dir} first)" if res["needs_review"] else ""))
