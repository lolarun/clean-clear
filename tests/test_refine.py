import cv2
import numpy as np

from clean_clear.pipeline import watermark_cache_path
from clean_clear.refine import first_pass_watermark


def test_first_pass_mask_comes_from_the_cache(tmp_path):
    src = tmp_path / "film.mp4"
    src.write_bytes(b"x")
    assert first_pass_watermark(src, [tmp_path], 36, 64) is None
    m = np.zeros((36, 64), np.uint8)
    m[2:6, 3:9] = 255
    (tmp_path / ".cache").mkdir()
    cv2.imwrite(str(tmp_path / ".cache" / "film.watermark.png"), m)  # name used before 0.3.4
    assert first_pass_watermark(src, [tmp_path], 36, 64).sum() == 24
    cv2.imwrite(str(watermark_cache_path(tmp_path, src)), np.zeros_like(m))  # current name wins: nothing erased
    got = first_pass_watermark(src, [tmp_path], 36, 64)
    assert got is not None and not got.any()
