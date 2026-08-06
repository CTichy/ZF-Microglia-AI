"""
_gpu_check.py — Hard precondition gate for Tab 4 (AI Tools).

Training (MONAI U-Net, Cellpose-SAM fine-tuning) needs real GPU headroom,
not just "a CUDA device exists" — a 2GB card can enumerate as CUDA-capable
and still OOM on a single inference forward pass (confirmed directly: a
T400 failed at every precision tried, bf16/fp32/fp16, on one ViT-SAM tile).
So the gate checks total VRAM, not just torch.cuda.is_available().

Checked once at import time and cached: total_memory is a static hardware
property that cannot change mid-session, so re-probing on every tab-show
would just add latency for zero informational gain (same reasoning as
_labeling.py's _detect_backend(), which also probes once at import).
"""

import os

_MIN_VRAM_BYTES = 8 * 1024 ** 3  # 8 GB floor — user-specified


def check_gpu_ok() -> tuple:
    """Return (ok: bool, message: str). Checks CUDA device 0 only."""
    if os.environ.get("NAPARI_AI_TOOLS_FORCE_HIDDEN"):
        return False, "AI Tools forcibly hidden (NAPARI_AI_TOOLS_FORCE_HIDDEN set)."
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "No CUDA-capable GPU detected — AI Tools disabled."
        props = torch.cuda.get_device_properties(0)
        gb = props.total_memory / 1024 ** 3
        if props.total_memory < _MIN_VRAM_BYTES:
            return False, (
                f"GPU '{props.name}' has {gb:.1f} GB VRAM "
                f"(< 8 GB minimum) — AI Tools disabled."
            )
        return True, f"GPU: {props.name} — {gb:.1f} GB VRAM"
    except Exception as exc:
        return False, f"GPU check failed: {exc} — AI Tools disabled."


GPU_OK, GPU_MSG = check_gpu_ok()
