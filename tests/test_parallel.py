from types import SimpleNamespace

from clean_clear import parallel
from clean_clear.cli import build_argparser
from clean_clear.parallel import _child_args, merge_cues

SRT_A = "1\n00:00:01,000 --> 00:00:02,000\n第一句\n\n2\n00:00:58,500 --> 00:01:00,000\n跨过分段的一句话\n"
SRT_B = "1\n00:00:00,000 --> 00:00:01,200\n跨过分段的一句话\n\n2\n00:00:05,000 --> 00:00:06,000\n第三句\n"


def test_a_subtitle_cut_by_the_split_becomes_one_cue():
    cues = merge_cues([(0.0, SRT_A), (60.0, SRT_B)])
    assert [(a, b, t) for a, b, t in cues] == [(1000, 2000, "第一句"), (58500, 61200, "跨过分段的一句话"),
                                              (65000, 66000, "第三句")]


def test_different_subtitles_at_the_split_stay_apart():
    b = SRT_B.replace("跨过分段的一句话", "完全不同")
    assert len(merge_cues([(0.0, SRT_A), (60.0, b)])) == 4


def test_parts_are_told_they_are_parts():
    args = build_argparser().parse_args([])
    cmd = _child_args(args, "off")
    assert "--split-part" in cmd
    assert build_argparser().parse_args(cmd).split_part


def test_vram_warning(monkeypatch, capsys):
    monkeypatch.setattr(parallel, "gpu_memory_gb", lambda: 24.0)
    parallel._vram_check(SimpleNamespace(jobs=4, model="propainter", pp_chunk=120))
    assert "WARNING" in capsys.readouterr().out
    parallel._vram_check(SimpleNamespace(jobs=4, model="lama", pp_chunk=120))
    assert "WARNING" not in capsys.readouterr().out
