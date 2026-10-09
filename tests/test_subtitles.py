from clean_clear.subtitles import build_segments, segments_from_ocr, ts


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
