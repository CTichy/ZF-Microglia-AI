"""
_crop_extraction.py — ported from skin_segmentation/extract_cellpose_crops.py.

Extracts per-cell training patches for Cellpose-SAM fine-tuning: for each
labelled cell, single/double/triple/quadruple bbox crops (cell alone, plus
its 1st/2nd/3rd nearest neighbours), deduplicated by cell-ID set, with an
optional cell-level train/val split.

Ported directly (like _cellpose_seg.py did for krendl_do3d.py) rather than
subprocessed: the source script is __main__-guarded with no import-time
side effects, and its only dependencies (numpy/pandas/scipy/tifffile) are
already plugin dependencies — importing it is simpler and faster than
shelling out for a job this quick (seconds-minutes).

One change from the original: sys.exit() on a missing CSV column would
kill the whole napari process if called in-process — replaced with
raise ValueError so the GUI can show it in a status label instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree


def _merged_bbox(label_ids, lookup):
    z0 = min(lookup[l]["bbox_z0_vox"] for l in label_ids)
    y0 = min(lookup[l]["bbox_y0_vox"] for l in label_ids)
    x0 = min(lookup[l]["bbox_x0_vox"] for l in label_ids)
    z1 = max(lookup[l]["bbox_z1_vox"] for l in label_ids)
    y1 = max(lookup[l]["bbox_y1_vox"] for l in label_ids)
    x1 = max(lookup[l]["bbox_x1_vox"] for l in label_ids)
    return z0, y0, x0, z1, y1, x1


def _extract_patch(target_ids, img, lbl, lookup, pad):
    Z, Y, X = img.shape
    z0, y0, x0, z1, y1, x1 = _merged_bbox(target_ids, lookup)
    cz0 = max(0, z0 - pad); cy0 = max(0, y0 - pad); cx0 = max(0, x0 - pad)
    cz1 = min(Z, z1 + pad); cy1 = min(Y, y1 + pad); cx1 = min(X, x1 + pad)
    img_crop = img[cz0:cz1, cy0:cy1, cx0:cx1].copy()
    lbl_crop = lbl[cz0:cz1, cy0:cy1, cx0:cx1].copy()
    mask = np.zeros_like(lbl_crop, dtype=bool)
    for t in target_ids:
        mask |= (lbl_crop == t)
    lbl_crop[~mask] = 0
    return img_crop, lbl_crop


def _patch_name(sorted_ids, kind):
    return "lbl" + "_".join(f"{i:03d}" for i in sorted_ids) + f"_{kind}"


def extract_crops(csv_path, img_path, lbl_path, pad=15,
                   out_subdir="train_cellpose", val_cells=None):
    """
    Returns a summary dict: {train: {single,double,triple,quadruple},
    val: {...}, dropped: int, skipped: int, train_dir: Path, val_dir: Path|None}.
    Raises ValueError if the CSV is missing required columns.
    """
    csv_path = Path(csv_path)
    img_path = Path(img_path)
    lbl_path = Path(lbl_path)
    val_set = set(val_cells) if val_cells else set()

    print(f"\n{'=' * 60}")
    print(f"CSV    : {csv_path.name}")
    print(f"Image  : {img_path.name}")
    print(f"Labels : {lbl_path.name}")
    print(f"Pad    : {pad} vox")
    if val_set:
        print(f"Val cells ({len(val_set)}): {sorted(val_set)}")
    print(f"{'=' * 60}")

    df = pd.read_csv(csv_path)
    img = tifffile.imread(str(img_path))
    lbl_vol = tifffile.imread(str(lbl_path))

    print(f"Volume shape  : {img.shape}")
    print(f"Labels in CSV : {len(df)}")

    required = {"label", "bbox_z0_vox", "bbox_y0_vox", "bbox_x0_vox",
                "bbox_z1_vox", "bbox_y1_vox", "bbox_x1_vox",
                "nearest_neighbor_label", "nearest_neighbor_2_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    lookup = {int(r.label): r._asdict() for r in df.itertuples(index=False)}

    label_ids = df["label"].values.astype(int)
    centroids = df[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].values
    k = min(len(df), 4)
    _, idxs = cKDTree(centroids).query(centroids, k=k)
    nn3_col = 3 if k == 4 else (2 if k >= 3 else 1)
    nn3_by_lbl = {int(label_ids[i]): int(label_ids[idxs[i, nn3_col]]) for i in range(len(df))}

    if val_set:
        train_dir = csv_path.parent / out_subdir / "train"
        val_dir = csv_path.parent / out_subdir / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
    else:
        train_dir = csv_path.parent / out_subdir
        val_dir = None
        train_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train dir : {train_dir}")
    if val_dir:
        print(f"Val dir   : {val_dir}")
    print()

    seen = set()
    n = {"train": {"single": 0, "double": 0, "triple": 0, "quadruple": 0},
         "val": {"single": 0, "double": 0, "triple": 0, "quadruple": 0},
         "dropped": 0, "skipped": 0}

    for row in df.itertuples(index=False):
        cell = int(row.label)
        nn1 = int(row.nearest_neighbor_label)
        nn2 = int(row.nearest_neighbor_2_label)
        nn3 = nn3_by_lbl[cell]

        for kind, ids in [("single", [cell]), ("double", [cell, nn1]),
                           ("triple", [cell, nn1, nn2]), ("quadruple", [cell, nn1, nn2, nn3])]:
            key = frozenset(ids)
            if key in seen:
                n["skipped"] += 1
                continue
            seen.add(key)

            if val_set:
                in_val = set(ids) & val_set
                in_train = set(ids) - val_set
                if in_val and in_train:
                    n["dropped"] += 1
                    print(f"  [DROPPED ] {_patch_name(sorted(ids), kind)}  (mixed)")
                    continue
                split = "val" if in_val else "train"
                out_dir = val_dir if split == "val" else train_dir
            else:
                split = "train"
                out_dir = train_dir

            sorted_ids = sorted(ids)
            name = _patch_name(sorted_ids, kind)
            img_crop, lbl_crop = _extract_patch(ids, img, lbl_vol, lookup, pad)

            tifffile.imwrite(str(out_dir / f"{name}_img.tif"), img_crop, compression="zlib")
            tifffile.imwrite(str(out_dir / f"{name}_lbl.tif"), lbl_crop.astype(np.int32), compression="zlib")
            n[split][kind] += 1
            print(f"  [{split:5s}/{kind:9s}]  {name}  shape={img_crop.shape}")

    print(f"\n{'=' * 60}")
    for split in ("train", "val"):
        s = n[split]
        total = sum(s.values())
        print(f"{split.upper():5s}: {s['single']} single | {s['double']} double | "
              f"{s['triple']} triple | {s['quadruple']} quadruple = {total} patches")
    print(f"Dropped (mixed): {n['dropped']}  |  Skipped (duplicate): {n['skipped']}")
    print(f"{'=' * 60}\n")

    n["train_dir"] = train_dir
    n["val_dir"] = val_dir
    return n
