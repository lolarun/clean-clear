import io
import json
import re
import urllib.error

import pytest

from clean_clear import cli, llm_ranges as L
from conftest import ffmpeg

KEY = "sk-test-0123456789"


@pytest.fixture
def film(tmp_path):
    """30 s, 25 fps, 160x90; and its SRT: credits (cue 1, 2.0-3.5 s), dialogue (cues 2-4), credits again (cue 5, 27-28 s)"""
    p = tmp_path / "film.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=s=160x90:r=25:d=30", "-c:v", "libx264", "-pix_fmt", "yuv420p", p)
    cues = [(2.0, 3.5, "录音 张磊 / Sound Designers Zhang Lei"), (6.0, 8.0, "你好吗"), (12.0, 14.0, "我很好"),
            (23.0, 25.0, "再见"), (27.0, 28.0, "音乐总监 方勃焱")]
    ts = lambda s: f"00:00:{int(s):02d},{int(round((s % 1) * 1000)):03d}"
    p.with_suffix(".srt").write_text("\n".join(f"{k}\n{ts(a)} --> {ts(b)}\n{t}\n" for k, (a, b, t) in enumerate(cues, 1)),
                                     encoding="utf-8")
    return p


def fake_model(monkeypatch, first=2, last=4, boundary_ok=True, better=None, shows=None):
    """Stands for the service: answers the text question and the frame question, remembers what it was asked"""
    asked = []

    def chat(cfg, content, model):
        asked.append((model, content))
        if any(part["type"] == "image_url" for part in content):
            times = [float(m) for p in content if p["type"] == "text" for m in re.findall(r"\(([\d.]+) s\):", p["text"])]
            frames = [{"t": t, "dialogue_subtitle": shows(t)} for t in times] if shows else []
            return {"frames": frames, "boundary_ok": boundary_ok, "better_time": better, "reason": "frames say so"}
        return {"first_dialogue_cue": first, "last_dialogue_cue": last, "reason": "credits come first and last"}

    monkeypatch.setattr(L, "chat_json", chat)
    return asked


CFG = {"base_url": "http://x/v1", "key": KEY, "model": "m", "vision_model": "mv", "timeout": 5.0}


def test_env_file_quotes_comments_and_the_real_environment_win(tmp_path, monkeypatch):
    for name in ("LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_MODEL", "OPENAI_MODEL",
                 "LLM_VISION_MODEL", "LLM_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text('# service\nexport LLM_BASE_URL="https://api.example.com/v1"\nLLM_API_KEY=abc # note\nLLM_MODEL=big\n',
                   encoding="utf-8")
    cfg = L.load_config(env)
    assert cfg == {"base_url": "https://api.example.com/v1", "key": "abc", "model": "big", "vision_model": "big",
                   "timeout": 120.0}
    monkeypatch.setenv("LLM_MODEL", "from-environment")
    monkeypatch.setenv("LLM_VISION_MODEL", "eyes")
    cfg = L.load_config(env)
    assert cfg["model"] == "from-environment" and cfg["vision_model"] == "eyes"


def test_a_missing_setting_is_named_but_a_secret_never_printed(tmp_path, monkeypatch):
    for name in ("LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_MODEL", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(f"LLM_API_KEY={KEY}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        L.load_config(env)
    assert "LLM_BASE_URL" in str(e.value) and "LLM_MODEL" in str(e.value) and KEY not in str(e.value)
    with pytest.raises(SystemExit):
        L.load_config(tmp_path / "missing.env")


def test_the_json_of_a_reply_is_found_in_a_fence_or_in_talk():
    assert L.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert L.extract_json('Sure! {"a": {"b": 2}} Hope it helps.') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        L.extract_json("no json here")


def test_requests_retry_server_errors_but_not_a_refused_key_and_never_show_the_key(monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(f"bad key {KEY}".encode()))

    monkeypatch.setattr(L.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError) as e:
        L._post(CFG, {})
    assert len(calls) == 1 and calls[0] == "http://x/v1/chat/completions"
    assert "HTTP 401" in str(e.value) and KEY not in str(e.value) and "[key]" in str(e.value)
    calls.clear()

    def flaky(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, io.BytesIO(b"busy"))

        class Reply(io.BytesIO):
            __enter__ = lambda self: self
            __exit__ = lambda self, *a: None
        return Reply(b'{"choices": [{"message": {"content": "{}"}}]}')

    monkeypatch.setattr(L.urllib.request, "urlopen", flaky)
    assert L._post(CFG, {})["choices"] and len(calls) == 3


def test_credits_before_the_first_and_after_the_last_dialogue_are_kept(film, tmp_path, monkeypatch):
    asked = fake_model(monkeypatch)
    res = L.suggest_keep(film, film.with_suffix(".srt"), tmp_path / "out", CFG, margin=0.5)
    assert res["keep_ranges"] == "0-5.500,25.500-end"          # 6.0 - 0.5 s and 25.0 + 0.5 s
    assert res["head"]["text"] == "你好吗" and res["tail"]["text"] == "再见"
    assert res["warnings"] == [] and not res["needs_review"]
    text_call, head_call, tail_call = asked
    assert text_call[0] == "m" and head_call[0] == "mv" == tail_call[0]       # the frames go to the vision model
    prompt = text_call[1][0]["text"]
    assert "#1 [0:00:02.00 - 0:00:03.50] 录音 张磊" in prompt and "#5 " in prompt
    # frames before 0 s and behind the end of the 30 s film become its first and last frame, which are sent once
    heads = {min(max(0.0, 6.0 + o), 29.9) for o in L.HEAD_SHOTS}
    tails = {min(max(0.0, 25.0 + o), 29.9) for o in L.TAIL_SHOTS}
    assert sum(p["type"] == "image_url" for p in head_call[1]) == len(heads) == 8
    assert sum(p["type"] == "image_url" for p in tail_call[1]) == len(tails) == 7
    saved = json.loads((tmp_path / "out" / "film.keep.json").read_text(encoding="utf-8"))
    assert saved["keep_ranges"] == res["keep_ranges"] and len(saved["head"]["check"]["frames"]) == 8
    assert all((tmp_path / "out" / f).stat().st_size > 500 for f in saved["head"]["check"]["frames"])
    assert KEY not in json.dumps(saved)


def test_a_subtitle_that_the_srt_ends_too_early_is_found_by_the_frames(film, tmp_path, monkeypatch):
    # Tianyan: the SRT ended the last dialogue at 1:40:05.3, the frames show it until about 1:40:08, credits at 1:40:27
    fake_model(monkeypatch, shows=lambda t: 6.0 <= t <= 26.0)                         # the dialogue is visible until 26.0 s
    res = L.suggest_keep(film, film.with_suffix(".srt"), tmp_path, CFG, margin=0.5)
    assert res["tail"]["srt_boundary"] == 25.0 and res["tail"]["boundary"] == 27.5      # the first sampled frame after 26.0 s
    assert res["keep_ranges"] == "0-5.500,28.000-end" and res["needs_review"]
    assert any("SRT time of the tail boundary" in w for w in res["warnings"])
    fake_model(monkeypatch, shows=lambda t: 6.0 <= t <= 25.0)                         # the frames agree with the SRT
    res = L.suggest_keep(film, film.with_suffix(".srt"), tmp_path, CFG, margin=0.5)
    assert "srt_boundary" not in res["tail"] and res["keep_ranges"] == "0-5.500,25.500-end" and not res["needs_review"]
    res = L.suggest_keep(film, film.with_suffix(".srt"), tmp_path, CFG)               # the default margin is 5 s:
    assert res["keep_ranges"] == "" and "keep" not in res["head"] and "keep" not in res["tail"]   # 6-5 and 25+5 s: nothing
    assert res["margin"] == 5.0                                                        # left worth protecting in 30 s


def test_when_the_outermost_frame_still_shows_a_subtitle_the_frames_are_taken_further_out(film, tmp_path, monkeypatch):
    # Bagua Zhang: the model took cue 13 (2:14) for the first dialogue, but the earliest frame, 20 s before it, showed narration
    asked = fake_model(monkeypatch, first=4, last=4, shows=lambda t: t >= 3.0)       # a subtitle from 3 s on; the "first" cue is at 23 s
    res = L.suggest_keep(film, film.with_suffix(".srt"), tmp_path, CFG, margin=0.5)
    pictures = [c for c in asked if any(p["type"] == "image_url" for p in c[1])]
    assert res["head"]["check"]["scale"] == 2                         # 23 - 20 = 3.0 s already shows one: twice as far out
    assert res["head"]["srt_boundary"] == 23.0 and res["head"]["boundary"] == 0.0     # no subtitle at 0.0 s: the 1st one is after
    assert "keep" not in res["head"] and res["tail"]["check"]["scale"] == 1           # 0 - 0.5 s: nothing to protect
    assert len(pictures) == 3                                         # the head twice, the tail once (it reaches the end)


def test_the_boundary_the_frames_show():
    times = [10.0, 15.0, 20.0, 25.0, 30.0]
    assert L.refine("tail", 12.0, times, [True, True, True, False, False]) == 25.0     # shown until 20 s: gone by 25 s
    assert L.refine("tail", 21.0, times, [True, True, True, False, False]) == 21.0     # nothing shown after the SRT end
    assert L.refine("tail", 12.0, times, [None, None, None, None, None]) == 12.0       # the model said nothing: the SRT stands
    assert L.refine("head", 28.0, times, [False, False, True, True, True]) == 15.0     # shown from 20 s: not yet at 15 s
    assert L.refine("head", 28.0, times, [True, True, True, True, True]) == 9.0        # shown in the first frame: 1 s before it
    assert L.flags_of([10.0, 15.0], [{"t": 15.2, "dialogue_subtitle": True}, {"t": 99, "dialogue_subtitle": True},
                                      {"t": 10.0, "dialogue_subtitle": "yes"}, "junk"]) == [None, True]


def test_frames_that_do_not_agree_make_it_a_matter_for_a_person(film, tmp_path, monkeypatch):
    fake_model(monkeypatch, boundary_ok=False, better=9.0)
    res = L.suggest_keep(film, film.with_suffix(".srt"), tmp_path, CFG, margin=0.5)
    assert res["needs_review"] and res["keep_ranges"] == "0-5.500,25.500-end"      # still the suggestion, but flagged
    assert any("do not confirm the head boundary" in w and "9.0" in w for w in res["warnings"])


def test_no_dialogue_cue_means_no_ranges_and_a_bad_cue_number_is_ignored(film, tmp_path, monkeypatch):
    fake_model(monkeypatch, first=None, last=None)
    res = L.suggest_keep(film, film.with_suffix(".srt"), tmp_path, CFG, margin=0.5)
    assert res["keep_ranges"] == "" and res["head"] is None and "no dialogue subtitle" in res["warnings"][0]
    fake_model(monkeypatch, first=99, last=4)
    res = L.suggest_keep(film, film.with_suffix(".srt"), tmp_path, CFG, margin=0.5)
    assert res["head"] is None and res["tail"]["cue"] == 4 and "not one of the cues" in res["warnings"][0]


def test_the_model_reads_the_cues_of_the_first_and_the_last_minutes_by_their_times(tmp_path, monkeypatch):
    # a long list of credits (97 cues at the end of Tianyan) must not push the dialogue out of the window: it is time, not a count
    film = tmp_path / "long.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=s=160x90:r=25:d=12", "-c:v", "libx264", "-pix_fmt", "yuv420p", film)
    ts = lambda s: f"00:00:{int(s):02d},{int(round((s % 1) * 1000)):03d}"
    cues = [f"{k}\n{ts(k - 0.8)} --> {ts(k - 0.3)}\ntext {k}\n" for k in range(1, 13)]      # cue k: k-0.8 .. k-0.3 s
    film.with_suffix(".srt").write_text("\n".join(cues), encoding="utf-8")
    asked = fake_model(monkeypatch, first=3, last=10)
    res = L.suggest_keep(film, film.with_suffix(".srt"), tmp_path, CFG, window=3.0, margin=0.5)
    prompt = asked[0][1][0]["text"]
    shown = {int(x) for x in re.findall(r"^#(\d+) ", prompt, flags=re.M)}
    assert shown == {1, 2, 3, 10, 11, 12}               # start before 3 s, or end after 9 s
    assert res["head"]["cue"] == 3 and res["tail"]["cue"] == 10
    fake_model(monkeypatch, first=5, last=10)           # a cue the model was not shown is ignored
    res = L.suggest_keep(film, film.with_suffix(".srt"), tmp_path, CFG, window=3.0, margin=0.5)
    assert res["head"] is None and "not one of the cues" in res["warnings"][0]
    assert any("no dialogue found at the head" in w for w in res["warnings"])


def test_the_command_prints_a_suggestion_and_exits_2_when_a_person_should_look(film, tmp_path, monkeypatch, capsys):
    env = tmp_path / "my.env"
    env.write_text(f"LLM_BASE_URL=http://x/v1\nLLM_API_KEY={KEY}\nLLM_MODEL=m\n", encoding="utf-8")
    fake_model(monkeypatch)
    out = tmp_path / "o"
    assert cli.main(["--suggest-keep", str(film), "--llm-env", str(env), "-o", str(out), "--keep-margin", "0.5"]) == 0
    assert '--keep-ranges "0-5.500,25.500-end"' in capsys.readouterr().out and (out / "film.keep.json").exists()
    fake_model(monkeypatch, boundary_ok=False)
    assert cli.main(["--suggest-keep", str(film), "--llm-env", str(env), "-o", str(out), "--keep-margin", "0.5"]) == 2
    film.with_suffix(".srt").unlink()
    with pytest.raises(SystemExit) as e:
        cli.main(["--suggest-keep", str(film), "--llm-env", str(env)])
    assert "SRT" in str(e.value)
