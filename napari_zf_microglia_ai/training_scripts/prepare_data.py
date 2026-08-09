#!/usr/bin/env python3
"""
Data Preparation v2 — Full GT Dataset

Datasets:
  Brain+Skin GT : NT26 (10), NT39 (5), NT54 (20)  →  35 brain fish
  Skin-only GT  : NT72 (5)                         →  negative examples

Split:
  Brain fish are randomly split 70 / 15 / 15  →  train / val / test
  NT72 skin-only fish always go to train (negative examples)

Usage (default paths already set):
  python prepare_data.py

Override dirs:
  python prepare_data.py --brain_dirs /path/NT26 /path/NT39 /path/NT54 \
                         --skin_dirs  /path/NT72  \
                         --output_dir ./training_data_v2
"""

import os
import numpy as np
import tifffile
import h5py
import json
from pathlib import Path
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


# ── Data loading helpers (unchanged from v1) ──────────────────────────────────

def load_volume_raw(tif_path):
    """Load TIF without normalization — preserves original uint16 intensity."""
    return tifffile.imread(str(tif_path))


def load_mask(tif_path):
    """Load binary mask."""
    return (tifffile.imread(str(tif_path)) > 0).astype(np.uint8)


def create_training_sample(fish_folder, output_dir, tissue='brain'):
    """
    Create one H5 training file from a fish folder.

    Stored datasets:
      volume     — raw uint16 microscopy data (no normalization)
      mask       — binary brain mask (GT, zeros for skin-only fish)
      brain_only — GT peeled brain at original intensity
      skin_only  — GT skin-only volume at original intensity

    Stored attributes:
      volume_p01 / volume_p99  — used for runtime normalisation in train.py
    """
    volume_file     = list(fish_folder.glob("*_original.tif"))
    mask_file       = list(fish_folder.glob(f"*_{tissue}_mask.tif"))
    brain_only_file = list(fish_folder.glob(f"*_{tissue}_only.tif"))
    skin_only_file  = list(fish_folder.glob("*_skin_only.tif"))

    if not volume_file or not mask_file:
        return None

    has_brain_only = len(brain_only_file) > 0
    has_skin_only  = len(skin_only_file) > 0

    volume = load_volume_raw(volume_file[0])
    mask   = load_mask(mask_file[0])

    if volume.shape != mask.shape:
        print(f"  Warning: shape mismatch in {fish_folder.name}")
        return None

    if has_brain_only:
        brain_only = load_volume_raw(brain_only_file[0])
        if brain_only.shape != volume.shape:
            print(f"  Warning: brain_only shape mismatch in {fish_folder.name}")
            return None
    else:
        brain_only = (volume.astype(np.float32) * mask.astype(np.float32)).astype(volume.dtype)

    if has_skin_only:
        skin_only = load_volume_raw(skin_only_file[0])
        if skin_only.shape != volume.shape:
            print(f"  Warning: skin_only shape mismatch in {fish_folder.name}")
            return None
    else:
        skin_only = (volume.astype(np.float32) * (1 - mask.astype(np.float32))).astype(volume.dtype)

    output_path = output_dir / f"{fish_folder.name}_{tissue}.h5"
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('volume',     data=volume,     compression='gzip')
        f.create_dataset('mask',       data=mask,       compression='gzip')
        f.create_dataset('brain_only', data=brain_only, compression='gzip')
        f.create_dataset('skin_only',  data=skin_only,  compression='gzip')

        f.attrs['shape']             = volume.shape
        f.attrs['tissue']            = tissue
        f.attrs['dtype']             = str(volume.dtype)
        f.attrs['has_brain_only_gt'] = has_brain_only
        f.attrs['has_skin_only_gt']  = has_skin_only
        f.attrs['source']            = str(fish_folder)
        f.attrs['volume_min']        = float(volume.min())
        f.attrs['volume_max']        = float(volume.max())
        f.attrs['volume_p01']        = float(np.percentile(volume, 1))
        f.attrs['volume_p99']        = float(np.percentile(volume, 99))

    return output_path


# ── Multiprocessing worker (top-level so it can be pickled) ───────────────────

def _process_one(args):
    """Worker function for Pool.map — processes one fish folder."""
    folder_str, output_dir_str, tissue = args
    folder     = Path(folder_str)
    output_dir = Path(output_dir_str)
    result = create_training_sample(folder, output_dir, tissue)
    return (folder.name, result is not None)


# ── Directory scanning ─────────────────────────────────────────────────────────

def find_gt_folders(data_dir, gt_glob):
    """Return subdirectories of data_dir that contain at least one GT file."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"  WARNING: directory not found: {data_dir}")
        return []
    all_dirs = sorted([f for f in data_dir.iterdir()
                       if f.is_dir() and not f.name.startswith('.')])
    gt_dirs = [f for f in all_dirs if list(f.glob(gt_glob))]
    print(f"  {data_dir.name}: {len(gt_dirs)}/{len(all_dirs)} subdirs have GT")
    return gt_dirs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Prepare v2 training data — full GT dataset')
    parser.add_argument('--brain_dirs', nargs='+', default=[],
                        help='Dirs with brain+skin GT fish')
    parser.add_argument('--skin_dirs', nargs='+', default=[],
                        help='Dirs with skin-only GT fish (negative examples)')
    parser.add_argument('--output_dir',    default='./training_data_v2')
    parser.add_argument('--tissue',        default='brain', choices=['skin', 'brain'])
    parser.add_argument('--n_val',       type=int, default=5,
                        help='Number of brain fish for validation (default: 5)')
    parser.add_argument('--n_test',      type=int, default=5,
                        help='Number of brain fish for test (default: 5)')
    parser.add_argument('--split_seed',   type=int, default=16)
    parser.add_argument('--num_workers',  type=int,
                        default=max(1, int(os.cpu_count() * 0.75)),
                        help='Parallel workers for H5 creation (default: 75%% of CPUs)')
    args = parser.parse_args()
    if not args.brain_dirs and not args.skin_dirs:
        parser.error("at least one of --brain_dirs / --skin_dirs must be given")

    output_dir = Path(args.output_dir)
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DATA PREPARATION v2 — Full GT Dataset (NT26 + NT39 + NT54 + NT72)")
    print("=" * 80)
    print(f"  Tissue        : {args.tissue}")
    print(f"  Workers       : {args.num_workers} / {os.cpu_count()} CPUs")
    print(f"  n_val         : {args.n_val}")
    print(f"  n_test        : {args.n_test}")
    print(f"  Split seed    : {args.split_seed}")
    print(f"  Output dir    : {output_dir}")

    # ── Collect brain GT folders ──────────────────────────────────────────
    print("\n--- Brain+Skin GT directories ---")
    brain_folders = []
    for d in args.brain_dirs:
        brain_folders.extend(find_gt_folders(d, '*_brain_mask.tif'))
    print(f"Total brain fish with GT: {len(brain_folders)}")

    # ── Collect skin-only GT folders ──────────────────────────────────────
    print("\n--- Skin-only GT directories (NT72) ---")
    skin_folders = []
    for d in args.skin_dirs:
        skin_folders.extend(find_gt_folders(d, '*_skin_mask.tif'))
    print(f"Total skin-only fish with GT: {len(skin_folders)}")

    # ── Random train / val / test split for brain fish ─────────────────────
    n_val   = args.n_val
    n_test  = args.n_test
    n_train = len(brain_folders) - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"Not enough brain fish ({len(brain_folders)}) for "
            f"n_val={n_val} + n_test={n_test} + at least 1 train sample.")

    rng  = np.random.default_rng(args.split_seed)
    idxs = np.arange(len(brain_folders))
    rng.shuffle(idxs)

    train_brain = [brain_folders[i] for i in idxs[:n_train]]
    val_brain   = [brain_folders[i] for i in idxs[n_train:n_train + n_val]]
    test_brain  = [brain_folders[i] for i in idxs[n_train + n_val:n_train + n_val + n_test]]

    print(f"\nSplit (seed={args.split_seed}):")
    print(f"  train : {n_train} brain + {len(skin_folders)} skin-only = "
          f"{n_train + len(skin_folders)} total")
    print(f"  val   : {len(val_brain)} brain")
    print(f"  test  : {len(test_brain)} brain")

    print("\nVal set:")
    for f in val_brain:
        print(f"  {f.name}")
    print("Test set:")
    for f in test_brain:
        print(f"  {f.name}")

    # ── Build manifest (before processing, so we keep it even if a sample fails)
    manifest = {
        'seed':        args.split_seed,
        'tissue':      args.tissue,
        'n_train':     n_train,
        'n_val':       n_val,
        'n_test':      n_test,
        'train_brain': [f.name for f in train_brain],
        'train_skin':  [f.name for f in skin_folders],
        'val':         [f.name for f in val_brain],
        'test':        [f.name for f in test_brain],
    }
    manifest_path = output_dir / 'split_manifest.json'
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nSplit manifest saved: {manifest_path}")

    # ── Process all splits in parallel ────────────────────────────────────
    for split_name, folders, out_dir in [
        ('train', train_brain + skin_folders, output_dir / 'train'),
        ('val',   val_brain,                  output_dir / 'val'),
        ('test',  test_brain,                 output_dir / 'test'),
    ]:
        print(f"\n{'='*60}")
        print(f"Processing split: {split_name}  ({len(folders)} samples, "
              f"{args.num_workers} workers)")
        print('='*60)

        tasks = [(str(f), str(out_dir), args.tissue) for f in folders]
        ok = 0
        with Pool(processes=args.num_workers) as pool:
            for name, success in tqdm(
                pool.imap_unordered(_process_one, tasks),
                total=len(tasks), desc=split_name
            ):
                if success:
                    ok += 1
                else:
                    print(f"  SKIP (missing files): {name}")
        print(f"  Created {ok}/{len(folders)} H5 files")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    for split in ['train', 'val', 'test']:
        files = list((output_dir / split).glob('*.h5'))
        print(f"  {split:5s}: {len(files)} H5 files")
    print(f"\nSplit manifest: {manifest_path}")
    print(f"\nNext step:")
    print(f"  python train.py --data_dir {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
