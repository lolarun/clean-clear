from clean_clear.subtitles import build_segments, segments_from_ocr, ts, write_srt


def frames(spec, n):
    """spec: {(a, b): text or (text, [boxes])} -> per-frame [(text, boxes)]"""
    out = [(None, [])] * n
    for (a, b), t in spec.items():
        t, boxes = t if isinstance(t, tuple) else (t, [(100, 300, 300, 330)])
        for i in range(a, b + 1):
            out[i] = (t, boxes)
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
    # boxes as OCR gave them: the shorter read keeps its left edge, only its right edge moved in
    per = frames({(0, 29): ("刚开始啊两三毛一斤", [(602, 944, 1318, 1031)]),
                  (30, 39): ("刚开始啊两三毛", [(608, 944, 1182, 1031)])}, 60)
    segs = build_segments(per, max_gap=2, min_len=10)
    assert [(s.start, s.end) for s in segs] == [(0, 39)]
    other = frames({(0, 29): "刚开始啊两三毛一斤", (30, 39): "后来呀"}, 60)
    assert len(build_segments(other, max_gap=2, min_len=10)) == 2


def test_a_short_subtitle_of_its_own_is_not_merged_into_the_longer_one_that_contains_its_text():
    # 1304 at 11:22: "远程教育" is shown centred on its own after "要架那个远程教育网"; merged, its glyphs were
    # outvoted in the segment's mask and partly left on screen
    per = frames({(0, 61): ("要架那个远程教育网", [(620, 944, 1300, 1031)]),
                  (62, 79): ("远程教育", [(810, 944, 1116, 1035)])}, 100)
    segs = build_segments(per, max_gap=2, min_len=10)
    assert [(s.start, s.end, s.text) for s in segs] == [(0, 61, "要架那个远程教育网"), (62, 79, "远程教育")]


def test_scene_text_above_the_subtitle_does_not_split_it():
    # 1304 at 08:33: "MONTBLANC" on a T-shirt is read as a first line above the subtitle for 10 frames; the subtitle
    # line is in the same place, so it stays one segment (and the T-shirt print is not erased on its own)
    sub = (546, 944, 1374, 1031)
    per = frames({(0, 29): ("这样才能更方便大家工作", [sub]),
                  (30, 39): ("MONTBLANC\n这样才能更方便大家工作", [(700, 780, 1000, 830), sub])}, 60)
    segs = build_segments(per, max_gap=2, min_len=10)
    assert [(s.start, s.end) for s in segs] == [(0, 39)]


def _scored(n, sub_score, extra=None):
    """n frames of a centred subtitle on the line (y 300-330, its box jittering by a pixel or two from frame to
    frame as OCR boxes do) with an outline score; extra: (box, text, score) of other text in the first 40 frames"""
    out = []
    for i in range(n):
        boxes = [[800 + i % 3, 300, 1120 - i % 2, 330, f"第{i // 30}句", sub_score]]
        if extra and i < 40:
            (x0, y0, x1, y1), t, s = extra
            boxes.append([x0, y0, x1, y1, t, s])
        out.append(boxes)
    return out


def test_text_in_the_picture_is_not_a_subtitle_when_the_subtitles_are_outlined():
    # 1304: the title "Red Peach", credits, a sign and print on a T-shirt were read as subtitles and erased
    per = _scored(90, 0.7, ((850, 268, 1070, 296), "Red Peach", 0.0))
    from clean_clear.subtitles import outlined_subtitles, subtitle_line
    assert outlined_subtitles(per, subtitle_line(per)) is True
    _, segs = segments_from_ocr(per, 25, max_gap=2, min_dur=0.4, outlined=True)
    assert [s.text for s in segs] == ["第0句", "第1句", "第2句"]
    _, before = segments_from_ocr(per, 25, max_gap=2, min_dur=0.4)
    assert any("Red Peach" in s.text for s in before)  # without the style it is taken as an upper subtitle line


def _same_text(scores, box=(800, 300, 1120, 330), text="我开封府定不负皇恩"):
    """one subtitle, one frame per score: its outline score frame by frame"""
    return [[[*box, text, s]] for s in scores]


def test_a_subtitle_on_a_bright_picture_is_not_cut_where_its_outline_score_drops():
    # Tianyan 1:40:05: "皇上万岁皇上万岁" scored 0.33-0.51, then 0.13 for 3.7 s where the background brightened: the SRT ended
    # there and the last 3.7 s stayed on screen. The same subtitle is one subtitle whatever its score
    per = _same_text([0.5] * 40 + [0.13] * 60)
    _, segs = segments_from_ocr(per, 25, max_gap=2, min_dur=0.4, outlined=True)
    assert [(s.start, s.end) for s in segs] == [(0, 99)]


def test_a_subtitle_that_scores_low_all_the_way_is_still_a_subtitle_above_the_scene_text_scores():
    # dialogue on a bright picture scored 0.11-0.23 (Tianyan "此妖女妖言惑众" 0.15 for 4.3 s) and was dropped whole by the
    # old limit of 0.25; the scene text of 1304 (title, sign, T-shirt print) scored 0 to 0.09
    for score, kept in ((0.15, True), (0.11, True), (0.09, False), (0.0, False)):
        _, segs = segments_from_ocr(_same_text([score] * 60), 25, max_gap=2, min_dur=0.4, outlined=True)
        assert bool(segs) is kept, score


def _boxes(frames_of):
    return [[list(b) for b in boxes] for boxes in frames_of]


def test_a_subtitle_is_what_sits_on_the_subtitle_template_whatever_its_outline_scores():
    # Tianyan, Shuimo Qingchun: dialogue on a bright picture scores 0.00-0.20 and was dropped; every dialogue segment sat
    # within 2 % of the middle on the subtitle line, the scene text of the same low score (plates, signs) 15-30 % off it
    line = (315.0, 30.0)
    for score in (0.0, 0.05, 0.09):
        per = _same_text([score] * 60, text="你们宋人要赔偿我们党项")
        _, segs = segments_from_ocr(per, 25, max_gap=2, min_dur=0.4, outlined=True, width=1920)
        assert len(segs) == 1, score                                  # centred (960 of 1920), on the line, reads like language
        _, segs = segments_from_ocr(per, 25, max_gap=2, min_dur=0.4, line=line, outlined=True, width=1920)
        assert len(segs) == 1, score                                  # also with the line known, as in the verify pass
    # off the middle, as a plate or a sign is
    plate = _same_text([0.0] * 60, box=(250, 300, 570, 330), text="你们宋人要赔偿我们党项")
    assert segments_from_ocr(plate, 25, max_gap=2, min_dur=0.4, outlined=True, width=1920)[1] == []
    # not on the line: half the height, or above it
    small = _same_text([0.0] * 60, box=(800, 310, 1120, 325), text="你们宋人要赔偿我们党项")
    assert segments_from_ocr(small, 25, max_gap=2, min_dur=0.4, line=line, outlined=True, width=1920)[1] == []
    # no width given: the middle is what the high-scoring boxes say (here a subtitle of 20 frames that scores 0.6)
    per = _same_text([0.6] * 20, text="第一句话在这里") + _same_text([0.0] * 60, text="你们宋人要赔偿我们党项")
    assert [s.text for s in segments_from_ocr(per, 25, max_gap=2, min_dur=0.4, outlined=True)[1]] == ["第一句话在这里", "你们宋人要赔偿我们党项"]


def test_text_that_does_not_read_like_language_is_not_taken_for_a_subtitle_on_the_template():
    # a number plate or print read as "9714", "77", "mg" is centred on the line in a bad moment; a word is not
    for text, kept in (("9714", False), ("77", False), ("mg", False), ("好", True), ("Oh, please wait", True), ("GB6072", False)):
        per = _same_text([0.0] * 60, text=text)
        assert bool(segments_from_ocr(per, 25, max_gap=2, min_dur=0.4, outlined=True, width=1920)[1]) is kept, text


def test_two_people_talking_put_two_subtitles_on_one_line_and_both_are_subtitles():
    # Shuimo Qingchun 10:22: "Oh, please pass my greetings!" in the middle and "Ok!" to the right of it, on the same row: the
    # second is off the middle and too short to read like language, and stayed on screen after the erasing
    from clean_clear.subtitles import style_filter
    main, right = (362, 465, 815, 504), (852, 464, 917, 502)
    per = [[[*main, "Oh, please pass my greetings!", 0.0], [*right, "Ok!", 0.0]] for _ in range(40)]
    _, segs = segments_from_ocr(per, 25, max_gap=2, min_dur=0.4, outlined=True, width=1280)
    assert len(segs) == 1 and "Ok!" in segs[0].text
    texts = lambda extra: [b[4] for b in style_filter([[[*main, "Oh, please pass my greetings!", 0.0], extra]], width=1280)[0]]
    assert "EXIT" not in texts([1100, 464, 1200, 502, "EXIT", 0.0])      # 285 px away: more than three text heights
    assert "Ok!" not in texts([852, 300, 917, 338, "Ok!", 0.0])          # another row: a sign above the subtitle
    assert "Ok!" not in texts([852, 480, 880, 495, "Ok!", 0.0])          # a third of the height: not the same print
    assert "Ok!" in texts([852, 464, 917, 502, "Ok!", 0.0])
    # nothing to hang on to: the same two boxes without the centred one are dropped
    assert style_filter([[[*right, "Ok!", 0.0], [1100, 464, 1200, 502, "EXIT", 0.0]]], width=1280)[0] == []


def test_a_title_above_the_subtitle_line_is_still_dropped_when_it_is_centred():
    # 1304: the title "Red Peach" sits above the line, in the middle: the template rule is for the line only
    per = _scored(90, 0.7, ((850, 268, 1070, 296), "Red Peach", 0.0))
    _, segs = segments_from_ocr(per, 25, max_gap=2, min_dur=0.4, outlined=True, width=1920)
    assert [s.text for s in segs] == ["第0句", "第1句", "第2句"]


def test_a_low_score_text_is_kept_only_next_to_the_same_text_with_a_good_score():
    from clean_clear.subtitles import style_filter
    off = (250, 300, 570, 330)          # off the middle: the template rule does not apply, only the neighbouring frames count
    per = _same_text([0.6] * 10 + [0.05] * 10 + [0.05] * 80 + [0.05] * 10, box=off)    # frames 0-9 score well, 10-109 do not
    kept = [bool(f) for f in style_filter(per, width=1920)]
    assert all(kept[:10 + 50]) and not any(kept[10 + 55:])                            # 50 frames after the last good one: gone
    other = [[[*off, "OPEN", 0.04]] for _ in range(60)] + _same_text([0.6] * 5, box=off)
    assert not any(style_filter(other, width=1920)[:60])                              # another text next to it: not rescued


def test_subtitles_without_an_outline_are_not_filtered_by_style():
    # yellow subtitles, or white ones without an outline: every box scores low, nothing is dropped
    per = _scored(90, 0.0, ((850, 268, 1070, 296), "Red Peach", 0.0))
    from clean_clear.subtitles import outlined_subtitles, subtitle_line
    assert outlined_subtitles(per, subtitle_line(per)) is False
    assert outlined_subtitles([[[800, 300, 1120, 330, "旧缓存"]]] * 30, (315, 30)) is None  # boxes without scores


def test_a_dropout_of_up_to_the_bridge_joins_the_very_same_text_in_the_same_place():
    # 1304 at 12:29: "行行行" read for 10 frames, lost for 13 behind a sign that slid past, read again for 2 frames
    box = (835, 937, 1085, 1041)
    per = frames({(0, 9): ("行行行", [box]), (23, 24): ("行行行", [box])}, 40)
    assert [(s.start, s.end) for s in build_segments(per, max_gap=5, min_len=10)] == [(0, 9)]  # the 2 frames were noise
    segs = build_segments(per, max_gap=5, min_len=10, bridge=25)
    assert [(s.start, s.end, s.gaps) for s in segs] == [(0, 24, [(10, 22)])]
    assert build_segments(per, max_gap=5, min_len=10, bridge=12)[0].end == 9   # a longer dropout than the bridge: not joined


def test_similar_but_different_subtitles_with_a_blank_between_are_not_joined():
    # 1304 at 2:04: "卖什么油桃", 0.8 s of plain picture, "还卖什么油桃": the texts are 0.91 similar and one contains the other
    per = frames({(0, 29): ("卖什么油桃", [(770, 940, 1154, 1030)]), (48, 77): ("还卖什么油桃", [(730, 940, 1195, 1030)])}, 90)
    segs = build_segments(per, max_gap=5, min_len=10, bridge=25)
    assert [(s.start, s.end) for s in segs] == [(0, 29), (48, 77)]


def test_the_same_text_in_another_place_is_not_joined():
    per = frames({(0, 14): ("好", [(900, 940, 960, 1030)]), (25, 39): ("好", [(300, 940, 360, 1030)])}, 50)
    assert len(build_segments(per, max_gap=5, min_len=10, bridge=25)) == 2
