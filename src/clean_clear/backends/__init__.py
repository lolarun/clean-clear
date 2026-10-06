"""Inpainting backends.

A backend erases subtitles from a sequential frame stream:

    backend.erase(frames, frame_seg, masks, cuts=()) -> iterator of frames

- frames:    iterator of HxWx3 BGR uint8 frames, in order, starting at frame 0
- frame_seg: {frame index: mask key} for every frame that needs erasing
- masks:     {mask key: HxW bool mask}
- cuts:      frame indices that start a new shot; only computed and passed when the backend sets
             `uses_cuts = True` (video models must not propagate pixels across a cut)
It must yield exactly one output frame per input frame, in the same order.
"""

MODELS = ("lama", "propainter")


def create(model, providers=None, propainter_dir=None, chunk=120, raft_iter=20, ctx=10, pad=8, margin=80,
           ref_stride=10):
    if model == "lama":
        from .lama import LamaBackend
        return LamaBackend(providers)
    if model == "propainter":
        from .propainter import ProPainterBackend
        return ProPainterBackend(propainter_dir, chunk=chunk, raft_iter=raft_iter, ctx=ctx, pad=pad, margin=margin,
                                  ref_stride=ref_stride)
    raise ValueError(f"unknown model: {model}")
