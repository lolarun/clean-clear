import os
import time
from types import SimpleNamespace

from clean_clear import cli
from clean_clear.pipeline import all_frames, ocr_cache_path, watermark_cache_path


def touch(p, data=b"x"):
    p.write_bytes(data)
    return p


def test_collect_inputs_skips_outputs_and_temporary_files(tmp_path):
    for n in ["a.mp4", "b.mkv", "a_clean.mp4", "a_refined.mp4", ".a_clean.tmp.mp4", "a_clean.fix.mp4", "c.txt"]:
        touch(tmp_path / n)
    assert [f.name for f in cli.collect_inputs([tmp_path])] == ["a.mp4", "b.mkv"]


def test_name_clashes(tmp_path):
    files = [touch(tmp_path / n) for n in ["a.mp4", "A.mkv", "b.mp4"]]
    assert cli.name_clashes(files) == {files[1]: files[0]}


def test_ocr_cache_key_follows_options_and_file(tmp_path):
    src = touch(tmp_path / "a.mp4")
    args = SimpleNamespace(min_score=0.6, min_height=0.015, ocr_interval=10)
    k1 = ocr_cache_path(tmp_path, src, (10, 20), args)
    assert ocr_cache_path(tmp_path, src, (10, 20), SimpleNamespace(**{**vars(args), "min_score": 0.5})) != k1
    w1 = watermark_cache_path(tmp_path, src)
    time.sleep(0.01)
    touch(src, b"yy")
    os.utime(src, ns=(time.time_ns(), time.time_ns()))
    assert ocr_cache_path(tmp_path, src, (10, 20), args) != k1
    assert watermark_cache_path(tmp_path, src) != w1


def test_all_frames_reaches_well_past_the_estimate():
    assert len(all_frames(100_000)) >= 102_000 and len(all_frames(10)) >= 260


def stub_models(monkeypatch):
    monkeypatch.setattr(cli, "onnx_providers", lambda d: ("cpu", ["CPUExecutionProvider"]))
    monkeypatch.setattr(cli, "OCR", lambda *a, **k: object())
    monkeypatch.setattr(cli.backends, "create", lambda *a, **k: object())
    monkeypatch.setattr(cli, "pick_encoder", lambda e: "libx264")


def test_finished_videos_are_skipped_and_failures_logged_with_traceback(tmp_path, monkeypatch, capsys):
    stub_models(monkeypatch)
    touch(tmp_path / "done.mp4"), touch(tmp_path / "done_clean.mp4"), touch(tmp_path / "done.srt")
    touch(tmp_path / "bad.mp4")
    calls = []

    def fake_process(f, *a):
        calls.append(f.name)
        raise ValueError("broken file")

    monkeypatch.setattr(cli, "process", fake_process)
    assert cli.main([str(tmp_path), "-o", str(tmp_path)]) == 1
    assert calls == ["bad.mp4"]
    out = capsys.readouterr().out
    assert "skipping done.mp4" in out and "Traceback" in out and "ValueError: broken file" in out
    calls.clear()
    cli.main([str(tmp_path), "-o", str(tmp_path), "--force"])
    assert calls == ["bad.mp4", "done.mp4"]
