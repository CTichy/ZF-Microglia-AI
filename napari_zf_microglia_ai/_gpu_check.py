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


# ── Free-VRAM check before launching a training run ────────────────────── #
# Training is a separate detached process, so it only gets the VRAM that
# is free at launch. napari's own PyTorch cache (and any other program on
# the GPU) is invisible to it until it hits CUDA out-of-memory, possibly
# after ~15 min of data loading. These helpers let the launcher check
# first. nvidia-smi is used (not torch) because torch only sees this
# process's own memory, not other processes'.

# Measured 2026-11-24 on the RTX 4060 Ti: train_xzyz.py at batch size 4
# (512 crops, branch-weighted loss) holds a steady 9.3 GB once training
# starts. Required free = that plus ~1.5 GB margin.
CELLPOSE_TRAIN_MIN_FREE_MIB = 11 * 1024


def release_own_gpu_cache() -> None:
    """Hand napari's unused cached CUDA blocks back to the driver.
    Memory still held by live tensors (e.g. a loaded model) stays."""
    try:
        import gc
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            gc.collect()
            torch.cuda.empty_cache()
    except Exception:
        pass


def gpu_memory_status(device: int = 0) -> dict:
    """Free/total VRAM of GPU `device` and the processes using it, from
    nvidia-smi. Returns {ok, total_mib, free_mib, procs: [(pid, name,
    used_mib)], error}. Never raises; ok=False when nvidia-smi fails, so
    the caller can launch anyway instead of blocking on a broken probe."""
    import subprocess
    out = dict(ok=False, total_mib=0, free_mib=0, procs=[], error=None)
    try:
        q = subprocess.run(
            ["nvidia-smi", f"--id={device}", "--query-gpu=uuid,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
        uuid, total, free = [s.strip() for s in q.split(",")]
        out.update(total_mib=int(total), free_mib=int(free))
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
        for line in apps:
            parts = [s.strip() for s in line.split(",")]
            if len(parts) == 4 and parts[0] == uuid:
                try:
                    out["procs"].append((int(parts[1]), parts[2], int(parts[3])))
                except ValueError:
                    pass
        out["ok"] = True
    except Exception as exc:
        out["error"] = str(exc)
    return out
