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
    parallel._vram_check(SimpleNamespace(jobs=4, model="propainter", pp_chunk=120), 1920, 1080)
    assert "WARNING" in capsys.readouterr().out
    parallel._vram_check(SimpleNamespace(jobs=4, model="lama", pp_chunk=120), 1920, 1080)
    assert "WARNING" not in capsys.readouterr().out


def test_auto_jobs_fit_the_gpu_memory_of_the_frame_size(monkeypatch):
    monkeypatch.setattr(parallel, "gpu_memory_gb", lambda: 31.8)  # RTX 5090
    auto = SimpleNamespace(jobs=4, jobs_auto=True, model="propainter", pp_chunk=120)
    assert parallel._fit_jobs(auto, 1280, 720) == 4    # ~6.5 GB each, as measured
    assert parallel._fit_jobs(auto, 1920, 1080) == 2   # 4 x ~13 GB ran out of memory
    fixed = SimpleNamespace(jobs=4, jobs_auto=False, model="propainter", pp_chunk=120)
    assert parallel._fit_jobs(fixed, 1920, 1080) == 4  # an explicit --jobs is the user's choice (with the warning)


def test_oom_detection_and_part_folder_cleaning(tmp_path):
    out = tmp_path / "part00.out"
    out.write_text("start\n" + "x" * 100 + "\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate\n")
    assert parallel._out_of_memory(out)
    out.write_text("[erase] 10/100 fine\n")
    assert not parallel._out_of_memory(out) and not parallel._out_of_memory(tmp_path / "missing.out")
    odir = tmp_path / "out00"
    (odir / ".cache").mkdir(parents=True)
    (odir / ".cache" / "a.ocr.json").write_text("[]")
    (odir / "part00_clean.mp4").write_bytes(b"x")
    (odir / "sub").mkdir()
    parallel._clean_part_dir(odir)
    assert [p.name for p in odir.iterdir()] == [".cache"] and (odir / ".cache" / "a.ocr.json").exists()


def test_a_part_that_runs_out_of_memory_starts_again_with_fewer_processes(monkeypatch, tmp_path):
    launches = []

    class FakeProc:
        def __init__(self, cmd, stdout=None, **kw):
            part = cmd[3]
            launches.append(part)
            self.rc = 0
            if part.endswith("part01.mkv") and launches.count(part) == 1:
                self.rc = 1
                stdout.write("torch.OutOfMemoryError: CUDA out of memory\n")
                stdout.flush()

        def poll(self):
            return self.rc

        def wait(self):
            return self.rc

        def kill(self):
            pass

    monkeypatch.setattr(parallel.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(parallel.time, "sleep", lambda s: None)
    monkeypatch.setattr(parallel, "LAUNCH_GAP", 0)
    args = build_argparser().parse_args(["x.mp4"])
    args.jobs = 2
    files = [tmp_path / f"part{k:02d}.mkv" for k in range(4)]
    done = []
    parallel._run_parts(args, files, tmp_path, "off", [None, None], [0, 1, 2, 3], 0.0, on_done=done.append)
    assert sorted(done) == [0, 1, 2, 3]  # no RuntimeError: the film is not lost
    assert launches.count(str(files[1])) == 2 and launches.count(str(files[0])) == 1


def test_finished_parts_are_kept_for_the_same_plan(tmp_path):
    from conftest import ffmpeg
    part = tmp_path / "part00.mkv"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", part)
    odir = tmp_path / "out00"
    odir.mkdir()
    ffmpeg("-i", part, "-c", "copy", odir / "part00_clean.mp4")
    assert not parallel._part_finished(tmp_path, 0, part, "key")      # no marker yet
    (odir / ".done").write_text("key")
    assert parallel._part_finished(tmp_path, 0, part, "key")
    assert not parallel._part_finished(tmp_path, 0, part, "other")    # another source, options or version
    short = tmp_path / "short.mkv"
    ffmpeg("-i", part, "-frames:v", "10", "-c", "copy", short)
    (odir / "part00_clean.mp4").unlink()
    ffmpeg("-i", short, "-c", "copy", odir / "part00_clean.mp4")
    assert not parallel._part_finished(tmp_path, 0, part, "key")      # fewer frames than the part: not finished


def test_the_plan_key_follows_source_options_and_version(tmp_path):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"1234")
    a = build_argparser().parse_args([str(src)])
    key = parallel._plan_key(src, 8, a, "off")
    assert key == parallel._plan_key(src, 8, a, "off")
    assert key != parallel._plan_key(src, 4, a, "off")
    assert key != parallel._plan_key(src, 8, a, "/m.png")
    b = build_argparser().parse_args([str(src), "--grow", "9"])
    assert key != parallel._plan_key(src, 8, b, "off")


def test_keep_ranges_of_the_film_become_the_pieces_of_each_part(monkeypatch):
    counts = {"p0": 3000, "p1": 3000, "p2": 3000}      # three parts of 120 s at 25 fps
    monkeypatch.setattr(parallel, "packet_count", lambda f: counts[f])
    specs = parallel._part_keeps("0-95,200-end", ["p0", "p1", "p2"], 25.0, 360.0)
    # 360 s film: parts [0,120) [120,240) [240,360); the ranges are 0-95 and 200-end (= 360)
    assert specs == ["0.000-95.000", "80.000-120.000", "0.000-120.000"]
    assert parallel._part_keeps("", ["p0"], 25.0, 120.0) == [""]
