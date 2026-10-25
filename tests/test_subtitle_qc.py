import csv

import numpy as np
import pytest

from clean_clear import subtitle_qc as Q
from clean_clear.masks import white_pixels
from conftest import ffmpeg


def test_normalisation_and_comparison_follow_the_customers_rules():
    assert Q.norm("老师，身上 香不香啊！ Hello-World 123") == "老师身上香不香啊helloworld123"
    assert Q.lcs("上香不香啊", "老师身上香不香啊") == 5
    cues = [Q.norm("老师身上香不香啊"), Q.norm("走")]
    assert Q.suspect("上香不香啊", cues) == "最长公共子串>=2"      # the half sentence of the customer's example
    assert Q.suspect("走", cues) == "单字出现在相邻台词"            # a single character is not filtered
    assert Q.suspect("干", cues) is None                           # a single character that is in no neighbour
    assert Q.suspect("Hello", [Q.norm("oh hello there")]) == "最长公共子串>=2"      # the English line of a bilingual subtitle
    assert Q.suspect("a", [Q.norm("what a day")]) == "单字出现在相邻台词"
    assert Q.suspect("###", cues) is None


def test_five_frames_per_cue_and_three_for_a_cue_under_a_second_fraction():
    cues = [(10.0, 12.0, "你好"), (20.0, 20.3, "走")]
    t = [(j, label) for j, label, _ in Q.cue_frame_times(cues, 100)]
    assert [x for x in t if x[0] == 0] == [(0, "mid"), (0, "start+0.1"), (0, "end-0.1"), (0, "start-0.5"), (0, "end+0.5")]
    assert [x[1] for x in t if x[0] == 1] == ["mid", "start-0.5", "end+0.5"]   # under 0.4 s: the middle only
    assert all(0 <= x[2] < 100 for x in Q.cue_frame_times([(0.2, 0.9, "a")], 1.0))  # frames outside the video are dropped


def test_subtitle_look_rules():
    look = {"line_y": 986.0, "line_h": 90.0, "style": 0.7, "outlined": True}
    ok = (700, 940, 1200, 1030)
    assert Q.looks_like_subtitle(ok, 1920, look, 0.6)
    assert not Q.looks_like_subtitle(ok, 1920, look, 0.05)                  # white print without an outline
    assert not Q.looks_like_subtitle((100, 940, 500, 1030), 1920, look, 0.6)    # not centred
    assert not Q.looks_like_subtitle((700, 700, 1200, 790), 1920, look, 0.6)    # another line
    assert not Q.looks_like_subtitle((700, 960, 1200, 1000), 1920, look, 0.6)   # half the height
    assert Q.looks_like_subtitle(ok, 1920, {"line_y": 986.0, "line_h": 90.0, "outlined": False}, 0.0)  # yellow subtitles


def test_a_blind_finding_that_stays_on_screen_is_one_row():
    rows = [{"method": "blind", "frame": 25 * k, "t": float(k), "text": "出品", "box": (800, 900, 1000, 960), "verdict": "待人工复核"}
            for k in range(5, 9)]
    out = Q.merge_runs(rows, 25.0)
    assert len(out) == 1 and out[0]["count"] == 4 and out[0]["frame"] == 125


def fake_ocr(img):
    ys, xs = np.nonzero(white_pixels(img))
    if len(ys) < 50:
        return []
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return [(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), "字幕", 0.95)]


def passed_calibration(*a, **k):
    return {"passed": True, "det_side": 1920, "samples": [], "detail": "test"}


def run_qc(tmp_path, monkeypatch, subtitle_video, result):
    monkeypatch.setattr(Q, "calibrate", passed_calibration)
    srt = tmp_path / "film.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\n字幕\n", encoding="utf-8")
    out = tmp_path / "qc"
    summary = Q.check_subtitles(subtitle_video, result, out, fake_ocr, srt)
    with open(out / "findings.csv", encoding="utf-8-sig") as fh:
        return summary, list(csv.DictReader(fh)), out


def test_a_result_that_still_shows_the_subtitle_fails(tmp_path, monkeypatch, subtitle_video):
    result = tmp_path / "film_clean.mp4"
    result.write_bytes(subtitle_video.read_bytes())      # nothing erased
    summary, rows, out = run_qc(tmp_path, monkeypatch, subtitle_video, result)
    assert summary["conclusion"] == "需返工" and summary["leftover"] >= 1
    assert any(r["verdict"] == "残留" and r["method"] == "cue" for r in rows)
    assert (out / "report.md").exists() and list((out / "evidence").glob("film_*.jpg"))
    assert not list((out / "evidence").glob("_f*.jpg"))     # working files are gone
    assert summary["cues"] == summary["cues_checked"] == 1


def test_a_clean_result_passes(tmp_path, monkeypatch, subtitle_video):
    result = tmp_path / "film_clean.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=0x3060a0:s=640x360:r=25:d=4", "-f", "lavfi", "-i", "sine=d=4",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", result)
    summary, rows, out = run_qc(tmp_path, monkeypatch, subtitle_video, result)
    assert summary["conclusion"] == "合格" and summary["leftover"] == 0 and rows == []


def test_text_that_is_the_same_in_the_source_but_not_like_the_subtitles_is_scene_text(tmp_path, monkeypatch, subtitle_video):
    # the bar is in the lower left corner of the result: not centred, so it is no subtitle, and unchanged from the source
    result = tmp_path / "film_clean.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=0x3060a0:s=640x360:r=25:d=4", "-f", "lavfi", "-i", "sine=d=4",
           "-vf", "drawbox=x=250:y=300:w=140:h=20:color=white:t=fill:enable='between(n,25,74)'",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", result)
    monkeypatch.setattr(Q, "looks_like_subtitle", lambda *a, **k: False)
    summary, rows, out = run_qc(tmp_path, monkeypatch, subtitle_video, result)
    assert summary["leftover"] == 0 and summary["dismissed"] >= 1 and summary["conclusion"] == "合格"
    assert all(r["frame_path"] == "" for r in rows if r["verdict"].startswith("场景文字"))


LOOK = {"line_y": 986.0, "line_h": 90.0, "style": 0.7, "outlined": True}


def test_a_half_erased_sentence_that_still_reads_is_a_leftover_whatever_changed():
    # 1304 at 12:29: "行行行" at the end of its erase range, half erased (difference to the source 10.8)
    box = (700, 940, 1200, 1030)
    cue = {"box": box, "style": 0.37, "diff": 10.8, "method": "cue", "t": 749.7}
    blind = {**cue, "method": "blind"}
    scene = {**cue, "style": 0.0, "diff": 2.0, "box": (100, 940, 500, 1030)}
    Q.judge([cue, blind, scene], 1920, LOOK)
    assert cue["verdict"] == "残留"
    assert blind["verdict"] == "待人工复核"            # text that is tied to no cue: a person says what it is
    assert scene["verdict"] == "场景文字(自动排除)"     # untouched and not like the subtitles
    credits = {**blind, "t": 12.0}
    Q.judge([credits], 1920, LOOK, keep=[(0.0, 95.0)])
    assert credits["verdict"] == "片头片尾(保留)"


def test_the_frames_are_split_into_consecutive_groups_that_cover_each_frame_once():
    runner = Q.Runner(fake_ocr, 4, 1920)      # an OCR that cannot be rebuilt in a worker: the groups still follow `workers`
    runner.workers = 4
    for count in (1, 5, 11, 12, 13, 2940):
        items = list(range(100, 100 + 7 * count, 7))
        groups = runner.chunks(items)
        assert [x for g in groups for x in g] == items and len(groups) <= 12 and all(groups)


def test_scanning_in_groups_finds_what_one_pass_finds(tmp_path, monkeypatch, subtitle_video):
    cues = [(1.0, 3.0, "字幕")]
    W, H, fps, n = Q.probe(subtitle_video)

    def scan(workers, out):
        runner = Q.Runner(fake_ocr, 1, 1920)
        runner.workers = workers
        jobs, stats = Q.start_scan(runner, subtitle_video, cues, W, H, fps, n, n / fps, out)
        found, stats = Q.finish_scan(jobs, stats)
        learned = Q.finish_learn(Q.start_learn(runner, subtitle_video, cues, W, H, fps))
        return found, stats, learned

    one, stats1, look1 = scan(1, tmp_path / "a")
    many, stats4, look4 = scan(4, tmp_path / "b")
    assert one and one == many and stats1 == stats4 and look1 == look4
    assert stats1["ocr_frames"] >= 5      # the five frames of the cue, and the blind scan
    assert sorted(p.name for p in (tmp_path / "a" / "evidence").glob("_f*")) == \
        sorted(p.name for p in (tmp_path / "b" / "evidence").glob("_f*"))


def test_a_text_that_matches_its_cue_is_a_leftover_whatever_its_outline_scores():
    # Tianyan: whole subtitles on a bright picture stayed in the result and the check made them "to be looked at" because
    # their outline scored 0.1-0.2: the score measures the picture behind the text, a text of the cue is a leftover
    box = (700, 940, 1200, 1030)
    cue = {"box": box, "style": 0.04, "diff": 0.5, "method": "cue", "t": 100.0}
    blind = {**cue, "method": "blind"}
    Q.judge([cue, blind], 1920, LOOK)
    assert cue["verdict"] == "残留"
    assert blind["verdict"] == "场景文字(自动排除)"       # tied to no cue: the score still tells scene text


def test_a_text_of_no_cue_that_reads_like_language_is_not_called_scene_text_whatever_it_scores():
    # Shuimo Qingchun: "What about your class?" (0.07) and "Ok!" (0.0) stayed on screen, and the check that found them called
    # them scene text because they scored under 0.1; a number plate or a shirt reads as "9714" or "mg"
    box = (700, 940, 1200, 1030)
    base = {"box": box, "style": 0.0, "diff": 0.5, "method": "blind", "t": 100.0}
    said = {**base, "text": "What about your class?"}
    plate = {**base, "text": "9714"}
    sign = {**base, "text": "What about your class?", "box": (100, 940, 500, 1030)}      # off the middle: a sign
    Q.judge([said, plate, sign], 1920, LOOK)
    assert said["verdict"] == "待人工复核"
    assert plate["verdict"] == "场景文字(自动排除)"
    assert sign["verdict"] == "场景文字(自动排除)"


def test_psnr_of_two_videos_by_ffmpeg(tmp_path):
    a, b, c = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=128x96:rate=25:duration=2", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", a)
    ffmpeg("-i", a, "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", b)                      # the same picture
    ffmpeg("-i", a, "-vf", "noise=alls=40:allf=t", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", c)   # noisy
    same = Q.finish_psnr(Q.start_psnr(a, b, tmp_path, "s", top=0.6), 25.0)
    noisy = Q.finish_psnr(Q.start_psnr(a, c, tmp_path, "n"), 25.0)
    assert same["n"] == 50 and same["min"] >= 90 and same["share_below"] == 0.0
    assert noisy["n"] == 50 and noisy["mean"] < 40 and noisy["share_below"] == 1.0
    part = Q.finish_psnr(Q.start_psnr(a, b, tmp_path, "p", t0=0.5, dur=1.0), 25.0)
    assert part["n"] == 25


def test_a_kept_range_is_compared_and_its_text_is_not_a_finding(tmp_path, monkeypatch, subtitle_video):
    result = tmp_path / "film_clean.mp4"
    result.write_bytes(subtitle_video.read_bytes())      # the bar (1.0-3.0 s) is still there, inside the kept range 0-3.5
    monkeypatch.setattr(Q, "calibrate", passed_calibration)
    srt = tmp_path / "film.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\n字幕\n", encoding="utf-8")
    summary = Q.check_subtitles(subtitle_video, result, tmp_path / "qc", fake_ocr, srt, keep_ranges="0-3.5")
    assert summary["leftover"] == 0 and summary["conclusion"] in ("合格", "待人工复核")
    report = (tmp_path / "qc" / "report.md").read_text(encoding="utf-8")
    assert "片头片尾保留段" in report and "99" in report     # identical to the source: 99 dB (capped)
    assert summary["psnr_outside"]["min"] >= Q.MIN_PSNR and summary["psnr_ok"]
