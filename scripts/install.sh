#!/usr/bin/env bash
# Clean Clear installer (Linux): creates .venv in the repository root and installs the package in editable mode
set -e
cd "$(dirname "$0")/.."
# PyPI by default; in mainland China use a local mirror: PIP_MIRROR=https://mirrors.cloud.tencent.com/pypi/simple/
MIRROR=${PIP_MIRROR:-https://pypi.org/simple/}
python3 -m venv .venv
.venv/bin/python -m pip install -U pip -i "$MIRROR"
.venv/bin/python -m pip install -e . -i "$MIRROR"
# the bundled ProPainter model needs PyTorch; RTX 50 series needs a CUDA 12.8+ build (current PyPI wheels have it)
if [ -f ProPainter/weights/ProPainter.pth ]; then
  .venv/bin/python -m pip install torch torchvision -i "$MIRROR"
  .venv/bin/python -m pip install -e ".[propainter]" -i "$MIRROR"
  .venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
fi
.venv/bin/python -c "import onnxruntime as o; o.preload_dlls(); print(o.get_available_providers())"
echo "Install OK. Command: $(pwd)/.venv/bin/clean-clear --help"
