from clean_clear.subtitles import build_segments, segments_from_ocr, ts, write_srt


def frames(spec, n):
    """spec: {(a, b): text} -> per-frame [(text, boxes)]"""
    out = [(None, [])] * n
    for (a, b), t in spec.items():
        for i in range(a, b + 1):
            out[i] = (t, [(100, 300, 300, 330)])
    return out


def test_short_segments_are_noise_unless_they_touch_an_edge():
    per = frames({(0, 3): "开头", (20, 60): "中间一句", (70, 72): "噪声", (96, 99): "结尾"}, 100)
    assert [s.text for s in build_segments(per, max_gap=2, min_len=10)] == ["中间一句"]
    kept = build_segments(per, max_gap=2, min_len=10, keep_edges=True)
    assert [s.text for s in kept] == ["开头", "中间一句", "结尾"]


def test_segments_from_raw_boxes():
    box = lambda t: [[100, 300, 300, 330, t]]
    per = [[]] * 10 + [box("你好")] * 20 + [[]] * 5 + [box("再见")] * 20
    line, segs = segments_from_ocr(per, 25, max_gap=2, min_dur=0.4)
    assert [(s.start, s.end, s.text) for s in segs] == [(10, 29, "你好"), (35, 54, "再见")]


def test_srt_timestamp():
    assert ts(3725.04) == "01:02:05,040"


def test_srt_follows_a_picture_that_starts_after_the_audio(tmp_path):
    per = [[]] * 25 + [[[100, 300, 300, 330, "line"]]] * 25 + [[]] * 50
    _, segs = segments_from_ocr(per, 25.0)
    assert [(s.start, s.end) for s in segs] == [(25, 49)]
    p = tmp_path / "a.srt"
    write_srt(segs, 25.0, p, offset=0.04)
    assert "00:00:01,040 --> 00:00:02,040" in p.read_text(encoding="utf-8")


def test_a_subtitle_whose_last_characters_OCR_lost_stays_one_segment():
    # 1304 at 09:12: "刚开始啊两三毛一斤" was followed by 0.4 s read as "刚开始啊两三毛"; the mask of that second
    # segment did not cover the two characters left, and they stayed on screen
    per = frames({(0, 29): "刚开始啊两三毛一斤", (30, 39): "刚开始啊两三毛"}, 60)
    segs = build_segments(per, max_gap=2, min_len=10)
    assert [(s.start, s.end) for s in segs] == [(0, 39)]
    other = frames({(0, 29): "刚开始啊两三毛一斤", (30, 39): "后来呀"}, 60)
    assert len(build_segments(other, max_gap=2, min_len=10)) == 2
