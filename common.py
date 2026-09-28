"""Shared constants and helpers."""
from pathlib import Path

__version__ = "0.2.0"

# Repository root: holds models/ and an optional ffmpeg/ folder
ROOT = Path(__file__).resolve().parent


def log(*a):
    print(*a, flush=True)
