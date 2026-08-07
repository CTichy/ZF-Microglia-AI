"""
_gpu_check.py — GPU/VRAM classification for Tab 4 (AI Tools)'s disclaimer banner.

Previously a hard precondition gate: Tab 4 was hidden entirely below
8GB VRAM or with no CUDA device at all. Changed on explicit instruction
-- don't stop a user from using the AI tools just because their GPU
(or lack of one) isn't ideal. CPU-only training/inference still works,
just far slower (days-months instead of hours for a full run); a
smaller GPU (e.g. 2GB) may also work with a reduced batch_size. Tab 4
is now always visible; this module only classifies the GPU situation
so the tab can show an accurate, prominent disclaimer instead of a
hard block.

Checked once at import time and cached: total_memory is a static
hardware property that cannot change mid-session, so re-probing on
every tab-show would just add latency for zero informational gain
(same reasoning as _labeling.py's _detect_backend(), which also probes
once at import).
"""

_MIN_RECOMMENDED_VRAM_BYTES = 8 * 1024 ** 3  # 8 GB — recommended, not required


def check_gpu() -> dict:
    """
    Returns {has_cuda, vram_gb, meets_recommended, name, message}.
    Never raises -- any detection failure is treated as "no CUDA GPU"
    (informational only now, not a gate, so failing safe just means a
    more cautious disclaimer, not a hidden tab).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return dict(has_cuda=False, vram_gb=0.0, meets_recommended=False,
                        name=None, message="No CUDA-capable GPU detected.")
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024 ** 3
        meets = props.total_memory >= _MIN_RECOMMENDED_VRAM_BYTES
        return dict(has_cuda=True, vram_gb=vram_gb, meets_recommended=meets,
                    name=props.name, message=f"GPU: {props.name} — {vram_gb:.1f} GB VRAM")
    except Exception as exc:
        return dict(has_cuda=False, vram_gb=0.0, meets_recommended=False,
                    name=None, message=f"GPU check failed: {exc}")


_info = check_gpu()
GPU_HAS_CUDA = _info["has_cuda"]
GPU_VRAM_GB = _info["vram_gb"]
GPU_MEETS_RECOMMENDED = _info["meets_recommended"]
GPU_NAME = _info["name"]
GPU_MSG = _info["message"]
# Kept as an alias for anything still importing the old name -- purely
# informational now (e.g. disclaimer banner text), never used to gate
# visibility or functionality.
GPU_OK = GPU_MEETS_RECOMMENDED
