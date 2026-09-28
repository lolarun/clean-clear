"""ONNX Runtime device selection (shared by OCR and LaMa)."""
import re
import subprocess
import sys


def onnx_providers(device):
    """'auto' / 'cuda' / 'dml' / 'cpu' -> (resolved device, onnxruntime providers)"""
    import onnxruntime as ort
    try:
        ort.preload_dlls()  # use the pip-installed CUDA/cuDNN (onnxruntime-gpu[cuda,cudnn])
    except Exception:
        pass
    avail = ort.get_available_providers()
    if device == "auto":
        device = "cuda" if "CUDAExecutionProvider" in avail else (
            "dml" if "DmlExecutionProvider" in avail else "cpu")
    prov = {"cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
            "cpu": ["CPUExecutionProvider"]}[device]
    if prov[0] not in avail:
        sys.exit(f"Device {device} is not available; onnxruntime providers: {avail}")
    return device, prov


def gpu_compute_capability():
    """Highest CUDA compute capability reported by nvidia-smi (e.g. 12.0 for Blackwell), 0.0 if unknown"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return 0.0
    caps = [float(l) for l in out.split() if re.fullmatch(r"\d+\.\d+", l)]
    return max(caps, default=0.0)
