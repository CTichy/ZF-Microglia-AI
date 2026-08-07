"""
_ai_tools.py — argv builders + default script-path resolution for the AI
Tools tab's training scripts.

These scripts (prepare_data.py, train.py, train_xzyz.py) are project-
specific research code, launched as separate subprocesses rather than
imported — but they ship *inside the installed package*, under
training_scripts/ next to this file, and are declared as package-data in
pyproject.toml. That placement matters: this project's main documented
install path (environment.yml's pip section) installs straight from
`git+https://github.com/...` into a throwaway build directory, not from
whatever local clone the user happens to have on disk — so anything
living outside the actual Python package (e.g. a training_scripts/ folder
at the repo root, sibling to napari_zf_microglia_ai/) never survives that
install and would 404 at runtime. Living *inside* the package directory
means these scripts travel with it into site-packages regardless of
whether the install is editable (`pip install -e .`) or the normal
git+https install. (An earlier version of this file pointed at sibling
folders in this project's private monorepo, which only existed on the
original dev machine and were never published at all — anyone else's
Tab 4 launchers failed with "script not found" out of the box.) Still
always overridable via a Browse button / the config file, e.g. to point
at a locally modified copy.
"""

from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parent             # .../napari_zf_microglia_ai (package dir)
_TRAINING_SCRIPTS_DIR = _PLUGIN_DIR / "training_scripts"

DEFAULT_PREPARE_DATA_SCRIPT = _TRAINING_SCRIPTS_DIR / "prepare_data.py"
DEFAULT_MONAI_TRAIN_SCRIPT = _TRAINING_SCRIPTS_DIR / "train.py"
DEFAULT_CELLPOSE_TRAIN_SCRIPT = _TRAINING_SCRIPTS_DIR / "train_xzyz.py"


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
