"""
_ai_tools.py — argv builders + best-effort default script-path resolution
for the AI Tools tab's external training scripts.

These scripts (prepare_data.py, train.py, train_xzyz.py) are not part of
the plugin package and aren't distributed with it — they live in sibling
research-project folders on this specific machine. The default paths
below are a best-effort guess based on this project's own directory
layout (same pattern as _inference.py's DEFAULT_MODEL / _SKIN_SEG_DIR),
always overridable via a Browse button; nothing here assumes the guess
is correct.
"""

from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[1]        # .../skin_segmentation/napari-skin-remover
_SKIN_SEG_DIR = _PLUGIN_DIR.parent                        # .../skin_segmentation
_MASTER_PROJECT_DIR = _SKIN_SEG_DIR.parent                # .../MasterProject

DEFAULT_PREPARE_DATA_SCRIPT = _SKIN_SEG_DIR / "prepare_data.py"
DEFAULT_MONAI_TRAIN_SCRIPT = _SKIN_SEG_DIR / "train.py"
DEFAULT_CELLPOSE_TRAIN_SCRIPT = _MASTER_PROJECT_DIR / "microglia_segmentation" / "train_xzyz.py"


def build_prepare_data_argv(script_path, brain_dirs, skin_dirs, output_dir,
                             tissue, n_val, n_test, split_seed, num_workers):
    argv = ["python", str(script_path)]
    if brain_dirs:
        argv += ["--brain_dirs", *[str(d) for d in brain_dirs]]
    if skin_dirs:
        argv += ["--skin_dirs", *[str(d) for d in skin_dirs]]
    argv += [
        "--output_dir", str(output_dir),
        "--tissue", tissue,
        "--n_val", str(n_val),
        "--n_test", str(n_test),
        "--split_seed", str(split_seed),
        "--num_workers", str(num_workers),
    ]
    return argv


def build_monai_train_argv(script_path, data_dir, model_dir, epochs, batch_size,
                            lr, resume, patience, val_every, ckpt_every, gpu):
    argv = [
        "python", str(script_path),
        "--data_dir", str(data_dir),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--model_dir", str(model_dir),
        "--patience", str(patience),
        "--val_every", str(val_every),
        "--ckpt_every", str(ckpt_every),
        "--gpu", str(gpu),
    ]
    if resume:
        argv += ["--resume", str(resume)]
    return argv


def build_cellpose_train_argv(script_path, data_dir, model_name, pretrained,
                               n_epochs, batch_size, save_every, log_every,
                               lr, branch_weight, branch_radius):
    argv = [
        "python", str(script_path),
        "--data_dir", str(data_dir),
        "--model_name", str(model_name),
        "--pretrained", str(pretrained),
        "--n_epochs", str(n_epochs),
        "--batch_size", str(batch_size),
        "--save_every", str(save_every),
        "--log_every", str(log_every),
        "--lr", str(lr),
        "--branch_weight", str(branch_weight),
        "--branch_radius", str(branch_radius),
    ]
    return argv
