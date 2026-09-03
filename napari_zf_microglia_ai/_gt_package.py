"""
_gt_package.py — build the GT-correction package this project has sent
to Nathalie (Peri Lab) for every fish's ground-truth creation: the
most-advanced Cellpose-SAM correction stage available as a correction
starting point, a reference copy of the raw pre-merge Cellpose masks, a
lightweight per-cell statistics CSV, the source image, and the
ground-truth creation guide, all zipped together.

Not a new methodology -- automates a manual step repeated by hand at
least three times in this project's history (D1F1, D1F2, two different
D1F4 fish), always producing the exact same file layout:

  <stem>_GT_package/
    GROUND_TRUTH_CREATION_GUIDE.md
    <stem>_cp_corrected.tif       ("start here" per the guide -- whatever
                                   correction stage was fed in, e.g. the
                                   sanded or auto-corrected result if
                                   that ran, Krendl-only otherwise; see
                                   _gt_toolkit.py's best_corrected_masks())
    <stem>_cp_masks_3D.tif        (raw pre-merge Cellpose masks, reference only)
    <stem>_cell_statistics.csv    (label/volume/centroid/bbox per cell)
    <stem>_brain_only_ExtRm.tif   (source image)
"""

import csv
import shutil
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import find_objects

# Best-effort default location of the guide template, matching this
# project's own directory layout (same pattern as _ai_tools.py's
# DEFAULT_*_SCRIPT constants) -- always overridable via the GUI.
_PLUGIN_DIR = Path(__file__).resolve().parents[1]         # .../skin_segmentation/napari-zf-microglia-ai
_SKIN_SEG_DIR = _PLUGIN_DIR.parent                          # .../skin_segmentation
_MASTER_PROJECT_DIR = _SKIN_SEG_DIR.parent                  # .../MasterProject
DEFAULT_GT_GUIDE_PATH = _MASTER_PROJECT_DIR / "microglia_segmentation" / "outputs" / "GROUND_TRUTH_CREATION_GUIDE.md"


def _cell_statistics_csv(labels, scale_zyx, out_csv):
    """
    Minimal per-cell CSV -- label/volume/centroid/bbox only, matching
    this project's established GT-package convention (deliberately not
    the full ~51-column Tab 3 Statistics output: this is a quick
    reference for a reviewer correcting labels, not a research CSV).
    Sorted by volume descending, largest cells first. Returns row count.
    """
    objs = find_objects(labels)
    vox_vol = scale_zyx[0] * scale_zyx[1] * scale_zyx[2]
    rows = []
    for label_id in range(1, len(objs) + 1):
        sl = objs[label_id - 1]
        if sl is None:
            continue
        mask = labels[sl] == label_id
        if not mask.any():
            continue
        coords = np.where(mask)
        vol_vox = int(mask.sum())
        centroid = [float(coords[i].mean() + sl[i].start) for i in range(3)]
        rows.append(dict(
            label=label_id,
            volume_vox=vol_vox,
            volume_um3=round(vol_vox * vox_vol, 2),
            centroid_z_vox=round(centroid[0], 2),
            centroid_y_vox=round(centroid[1], 2),
            centroid_x_vox=round(centroid[2], 2),
            bbox_z0_vox=sl[0].start, bbox_y0_vox=sl[1].start, bbox_x0_vox=sl[2].start,
            bbox_z1_vox=sl[0].stop, bbox_y1_vox=sl[1].stop, bbox_x1_vox=sl[2].stop,
        ))
    rows.sort(key=lambda r: -r["volume_vox"])

    fieldnames = ["label", "volume_vox", "volume_um3",
                  "centroid_z_vox", "centroid_y_vox", "centroid_x_vox",
                  "bbox_z0_vox", "bbox_y0_vox", "bbox_x0_vox",
                  "bbox_z1_vox", "bbox_y1_vox", "bbox_x1_vox"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_gt_package(stem, out_dir, image_path, corrected_masks_path,
                      raw_cellpose_masks_path=None, scale_zyx=(1.0, 0.174, 0.174),
                      guide_path=None, progress_cb=None):
    """
    Assembles <out_dir>/<stem>_GT_package/ and zips it to
    <out_dir>/<stem>_GT_package.zip, matching this project's established
    layout exactly.

    stem                     : fish identifier used for all output filenames
    out_dir                  : where to create the package folder + zip
    image_path                : source image (copied in as brain_only_ExtRm.tif)
    corrected_masks_path      : whichever Cellpose-SAM correction stage is
                                being sent (Krendl-only, auto-corrected, or
                                sanded -- caller's choice, typically the
                                most-advanced one available, see
                                _gt_toolkit.best_corrected_masks()); copied
                                in as cp_corrected.tif, the correction
                                starting point
    raw_cellpose_masks_path   : optional raw pre-merge Cellpose masks
                                (copied in as cp_masks_3D.tif, reference only)
    scale_zyx                 : (Z, Y, X) um/voxel, for the statistics CSV
    guide_path                 : path to GROUND_TRUTH_CREATION_GUIDE.md,
                                defaults to DEFAULT_GT_GUIDE_PATH

    Returns (package_dir, zip_path, n_cells).
    """
    def _report(msg):
        if progress_cb:
            progress_cb(msg)

    guide_path = Path(guide_path) if guide_path else DEFAULT_GT_GUIDE_PATH
    if not guide_path.exists():
        raise FileNotFoundError(
            f"GROUND_TRUTH_CREATION_GUIDE.md not found at {guide_path} -- "
            "browse to it explicitly if it lives somewhere else on this machine."
        )

    out_dir = Path(out_dir)
    package_dir = out_dir / f"{stem}_GT_package"
    package_dir.mkdir(parents=True, exist_ok=True)

    _report("copying source image...")
    shutil.copy2(image_path, package_dir / f"{stem}_brain_only_ExtRm.tif")

    _report("copying corrected masks (correction starting point)...")
    shutil.copy2(corrected_masks_path, package_dir / f"{stem}_cp_corrected.tif")

    if raw_cellpose_masks_path:
        _report("copying raw pre-merge Cellpose masks (reference)...")
        shutil.copy2(raw_cellpose_masks_path, package_dir / f"{stem}_cp_masks_3D.tif")

    _report("copying ground-truth creation guide...")
    shutil.copy2(guide_path, package_dir / "GROUND_TRUTH_CREATION_GUIDE.md")

    _report("computing per-cell statistics CSV...")
    labels = tifffile.imread(corrected_masks_path).astype(np.int32)
    n_cells = _cell_statistics_csv(labels, scale_zyx, package_dir / f"{stem}_cell_statistics.csv")

    _report("zipping package...")
    zip_path = out_dir / f"{stem}_GT_package.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(package_dir.iterdir()):
            zf.write(f, arcname=f"{stem}_GT_package/{f.name}")

    _report(f"done — {n_cells} cells, package at {zip_path}")
    return package_dir, zip_path, n_cells
