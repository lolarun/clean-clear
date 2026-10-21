import json

import pytest

from clean_clear import qc
from conftest import ffmpeg
from test_video import round_trip


def test_subtitle_language():
    assert qc.subtitle_language([]) == "none"
    assert qc.subtitle_language(["我们走吧，天快黑了"] * 5) == "zh-hans"
    assert qc.subtitle_language(["我們走吧，這個時候天快黑了"] * 5) == "zh-hant"
    assert qc.subtitle_language(["我们走吧\nLet us go now"] * 5) == "zh-hans;en"
    assert qc.subtitle_language(["Let us go now"] * 5) == "en"


def test_read_srt(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:01,040 --> 00:00:02,000\n你好\n\n2\n00:01:00,000 --> 00:01:01,500\n再见\n", encoding="utf-8")
    cues = qc.read_srt(p)
    assert [c[2] for c in cues] == ["你好", "再见"]
    assert [c[:2] for c in cues] == [pytest.approx((1.04, 2.0)), pytest.approx((60.0, 61.5))]
    p.write_text("1\nnot a time\ntext\n", encoding="utf-8")
    with pytest.raises(ValueError):
        qc.read_srt(p)


def test_check_passes_a_faithful_result_and_names_what_differs(subtitle_video, tmp_path):
    good = tmp_path / "good.mp4"
    round_trip(subtitle_video, good)
    res = qc.check(subtitle_video, good, tmp_path / "qc_good", sheets=False)
    failed = {c["name"] for c in res["checks"] if not c["ok"]}
    assert failed <= {"bitrate"}  # a flat synthetic picture may encode below the source's bitrate
    assert json.loads((tmp_path / "qc_good" / "qc.json").read_text(encoding="utf-8"))["result"]["md5"]

    short = tmp_path / "short.mp4"
    ffmpeg("-i", subtitle_video, "-frames:v", "90", "-c:v", "libx264", "-c:a", "copy", short)
    res = qc.check(subtitle_video, short, tmp_path / "qc_short", sheets=False)
    assert not res["pass"]
    assert {"frames", "duration"} <= {c["name"] for c in res["checks"] if not c["ok"]}


def test_contact_sheets_are_written(subtitle_video, tmp_path):
    out = tmp_path / "out.mp4"
    round_trip(subtitle_video, out)
    res = qc.check(subtitle_video, out, tmp_path / "qc")
    assert set(res["sheets"]) == {"sheet_full.jpg", "sheet_detail.jpg"}
    assert all((tmp_path / "qc" / n).stat().st_size > 0 for n in res["sheets"])
