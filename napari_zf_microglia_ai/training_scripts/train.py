#!/usr/bin/env python3
"""
MONAI 3D U-Net Training — v2 (Full GT Dataset, 1500 epochs)

Key design choices:
  - 35 brain fish (NT26 + NT39 + NT54) + 5 NT72 skin-only negatives
  - Random 70/15/15 train/val/test split (done in prepare_data.py)
  - 4-component loss: seg + brain_recon + skin_supervision + microglia
  - FULL-BRAIN sliding window dice for model selection and early stopping
    (patch-based dice was 99.98% and misleading — full-brain is the real metric)
  - 1500 epochs, early stopping patience 50 (validation cycles)
  - Best model saved as models_v2/best_model_fullstack.pth
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, RandRotate90d, RandFlipd, RandGaussianNoised,
    RandScaleIntensityd, EnsureTyped
)
import os
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import json
from datetime import datetime


# ── Logger ────────────────────────────────────────────────────────────────────

class TeeLogger:
    """Writes to both console and file simultaneously."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ── Patch-based dataset (for training + monitoring validation) ─────────────────

class SegmentationDataset(Dataset):
    """
    Yields random patches from H5 volumes.
    Normalisation is done at runtime using stored p01/p99 percentiles.
    """

    def __init__(self, h5_files, patch_size=(64, 192, 192),
                 patches_per_volume=10, transform=None, cache_data=False):
        self.h5_files = h5_files
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.transform = transform
        self.cache_data = cache_data
        self.data_cache = {}

        if self.cache_data:
            print(f"  Preloading {len(h5_files)} volumes into memory...")
            for i, h5_file in enumerate(h5_files):
                with h5py.File(h5_file, 'r') as f:
                    self.data_cache[i] = {
                        'volume':     f['volume'][:],
                        'mask':       f['mask'][:],
                        'brain_only': f['brain_only'][:],
                        'skin_only':  f['skin_only'][:],
                        'p01':        f.attrs['volume_p01'],
                        'p99':        f.attrs['volume_p99'],
                    }
            print(f"  Cached!")

    def __len__(self):
        return len(self.h5_files) * self.patches_per_volume

    def normalize_volume(self, volume, p01, p99):
        volume = volume.astype(np.float32)
        return np.clip((volume - p01) / max(p99 - p01, 1.0), 0, 1)

    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume

        if self.cache_data:
            d = self.data_cache[vol_idx]
            volume, mask = d['volume'], d['mask']
            brain_only, skin_only = d['brain_only'], d['skin_only']
            p01, p99 = d['p01'], d['p99']
        else:
            with h5py.File(self.h5_files[vol_idx], 'r') as f:
                volume     = f['volume'][:]
                mask       = f['mask'][:]
                brain_only = f['brain_only'][:]
                skin_only  = f['skin_only'][:]
                p01        = f.attrs['volume_p01']
                p99        = f.attrs['volume_p99']

        volume_norm     = self.normalize_volume(volume,     p01, p99)[np.newaxis].astype(np.float32)
        mask            = mask[np.newaxis].astype(np.float32)
        brain_only_norm = self.normalize_volume(brain_only, p01, p99)[np.newaxis].astype(np.float32)
        skin_only_norm  = self.normalize_volume(skin_only,  p01, p99)[np.newaxis].astype(np.float32)

        pz, py, px = self.patch_size
        z_start = np.random.randint(0, max(1, volume_norm.shape[1] - pz + 1))
        y_start = np.random.randint(0, max(1, volume_norm.shape[2] - py + 1))
        x_start = np.random.randint(0, max(1, volume_norm.shape[3] - px + 1))

        s = (slice(None),
             slice(z_start, z_start + pz),
             slice(y_start, y_start + py),
             slice(x_start, x_start + px))

        data = {
            'image':     volume_norm[s],
            'label':     mask[s],
            'brain_only': brain_only_norm[s],
            'skin_only':  skin_only_norm[s],
        }

        if self.transform:
            data = self.transform(data)

        return data['image'], data['label'], data['brain_only'], data['skin_only']


# ── Augmentation ──────────────────────────────────────────────────────────────

def get_transforms(training=True):
    if training:
        return Compose([
            EnsureTyped(keys=['image', 'label', 'brain_only', 'skin_only']),
            RandRotate90d(keys=['image', 'label', 'brain_only', 'skin_only'],
                          prob=0.5, spatial_axes=(1, 2)),
            RandFlipd(keys=['image', 'label', 'brain_only', 'skin_only'],
                      prob=0.5, spatial_axis=0),
            RandGaussianNoised(keys=['image'], prob=0.2, std=0.01),
            RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.2),
        ])
    return Compose([
        EnsureTyped(keys=['image', 'label', 'brain_only', 'skin_only'])
    ])


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


# ── Loss ──────────────────────────────────────────────────────────────────────

class FourComponentLoss(nn.Module):
    """
    4-component loss:
      1. Segmentation (DiceCE on binary brain mask)
      2. Brain reconstruction (predicted brain must match brain_only GT)
      3. Skin supervision (predicted skin must match skin_only GT)
      4. Microglia protection (heavy penalty for removing bright pixels)
    """

    def __init__(self, seg_weight=1.0, brain_recon_weight=2.0,
                 skin_super_weight=3.0, microglia_weight=5.0):
        super().__init__()
        self.seg_weight        = seg_weight
        self.brain_recon_weight = brain_recon_weight
        self.skin_super_weight = skin_super_weight
        self.microglia_weight  = microglia_weight
        self.dice_ce_loss      = DiceCELoss(sigmoid=True)
        self.mse_loss          = nn.MSELoss()

    def forward(self, pred_logits, gt_mask, volume, brain_only_gt, skin_only_gt):
        pred_mask = torch.sigmoid(pred_logits)

        loss_seg        = self.dice_ce_loss(pred_logits, gt_mask)
        loss_brain_recon = self.mse_loss(volume * pred_mask, brain_only_gt)
        loss_skin_super  = self.mse_loss(volume * (1.0 - pred_mask), skin_only_gt)

        thr = brain_only_gt.flatten(1).quantile(0.90, dim=1, keepdim=True).view(-1, 1, 1, 1, 1)
        microglia_mask = (brain_only_gt > thr).float()
        loss_microglia = self.mse_loss(pred_mask * microglia_mask,
                                       torch.ones_like(microglia_mask))

        total = (self.seg_weight        * loss_seg +
                 self.brain_recon_weight * loss_brain_recon +
                 self.skin_super_weight  * loss_skin_super +
                 self.microglia_weight   * loss_microglia)

        return total, {
            'seg':         loss_seg.item(),
            'brain_recon': loss_brain_recon.item(),
            'skin_super':  loss_skin_super.item(),
            'microglia':   loss_microglia.item(),
            'total':       total.item(),
        }


# ── Training / patch-validation loops ─────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_losses = {k: 0.0 for k in ['seg', 'brain_recon', 'skin_super', 'microglia', 'total']}

    pbar = tqdm(loader, desc="Train", file=sys.stdout)
    for volume, mask, brain_only, skin_only in pbar:
        volume     = volume.to(device)
        mask       = mask.to(device)
        brain_only = brain_only.to(device)
        skin_only  = skin_only.to(device)

        optimizer.zero_grad()
        pred = model(volume)
        loss, loss_dict = criterion(pred, mask, volume, brain_only, skin_only)
        loss.backward()
        optimizer.step()

        for k in epoch_losses:
            epoch_losses[k] += loss_dict[k]
        pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})

    for k in epoch_losses:
        epoch_losses[k] /= len(loader)
    return epoch_losses


def validate_patches(model, loader, criterion, dice_metric, device):
    """Patch-based validation — used for loss monitoring only."""
    model.eval()
    epoch_losses = {k: 0.0 for k in ['seg', 'brain_recon', 'skin_super', 'microglia', 'total']}
    dice_metric.reset()

    with torch.no_grad():
        for volume, mask, brain_only, skin_only in tqdm(loader, desc="Val (patches)", file=sys.stdout):
            volume     = volume.to(device)
            mask       = mask.to(device)
            brain_only = brain_only.to(device)
            skin_only  = skin_only.to(device)

            pred = model(volume)
            loss, loss_dict = criterion(pred, mask, volume, brain_only, skin_only)

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]
            dice_metric((torch.sigmoid(pred) > 0.5).float(), mask)

    for k in epoch_losses:
        epoch_losses[k] /= len(loader)

    patch_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return epoch_losses, patch_dice


# ── Full-brain sliding window validation (TRUE metric for model selection) ─────

def full_brain_validate(model, h5_files, patch_size, device):
    """
    Runs sliding window inference on complete volumes and returns mean Dice.

    This is the REAL metric — patch dice is misleading (~99.98%) because
    random patches rarely contain ambiguous brain/skin boundaries.
    Full-brain dice reflects the actual segmentation quality.
    """
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                volume = f['volume'][:]
                mask   = f['mask'][:]
                p01    = f.attrs['volume_p01']
                p99    = f.attrs['volume_p99']

            # Runtime normalisation
            volume = np.clip((volume.astype(np.float32) - p01) / max(p99 - p01, 1.0), 0, 1)

            # [1, 1, D, H, W]  — .float() guards against float64 upcast from HDF5 attrs
            volume_t = torch.from_numpy(volume[np.newaxis, np.newaxis]).float().to(device)
            mask_t   = torch.from_numpy(mask.astype(np.float32)[np.newaxis, np.newaxis])

            # Sliding window — predictor returns probabilities (sigmoid applied)
            pred = sliding_window_inference(
                inputs=volume_t,
                roi_size=patch_size,
                sw_batch_size=4,
                predictor=lambda x: torch.sigmoid(model(x)),
                overlap=0.5,
                mode='gaussian',
            )

            pred_binary = (pred > 0.5).float().cpu()
            del pred, volume_t
            torch.cuda.empty_cache()

            tp    = (pred_binary * mask_t).sum().item()
            total = pred_binary.sum().item() + mask_t.sum().item()
            dice  = (2.0 * tp) / (total + 1e-8)
            dice_scores.append(dice)

            name = Path(h5_file).stem
            print(f"    {name}: Dice={dice:.4f}")

    mean_dice = float(np.mean(dice_scores))
    print(f"  Mean full-brain Dice: {mean_dice:.4f}")
    return mean_dice


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train v2 — full GT, 1500 epochs, full-brain selection')
    parser.add_argument('--data_dir',         default='training_data_v2')
    parser.add_argument('--epochs',      type=int,   default=1500)
    parser.add_argument('--batch_size',  type=int,   default=2)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--model_dir',            default='models_v2')
    parser.add_argument('--resume',               default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--patience',    type=int,   default=50,
                        help='Early stopping: max val cycles without improvement')
    parser.add_argument('--val_every',   type=int,   default=5,  help='Validate every N epochs')
    parser.add_argument('--ckpt_every',  type=int,   default=50, help='Save checkpoint every N epochs')
    parser.add_argument('--cache_data',  action='store_true', help='Preload all data into RAM')
    parser.add_argument('--num_workers', type=int,
                        default=4,
                        help='DataLoader workers')
    parser.add_argument('--gpu',         type=int,   default=0, help='GPU device index')
    args = parser.parse_args()

    # Logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = f'training_v2_{timestamp}.log'
    logger    = TeeLogger(log_file)
    sys.stdout = logger

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    Path(args.model_dir).mkdir(exist_ok=True)

    PATCH_SIZE = (64, 192, 192)

    print("=" * 80)
    print("TRAINING v2 — Full GT Dataset, Full-Brain Model Selection")
    print("=" * 80)
    print(f"Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file  : {log_file}")
    print(f"Device    : {device}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Epochs    : {args.epochs}")
    print(f"Patience  : {args.patience} val cycles "
          f"(= {args.patience * args.val_every} epochs without improvement)")
    print(f"Val every : every {args.val_every} epochs")
    print(f"Model sel.: FULL-BRAIN sliding window Dice (not misleading patch Dice)")
    print("=" * 80)

    # ── Load data ──────────────────────────────────────────────────────────
    data_dir    = Path(args.data_dir)
    train_files = sorted(list((data_dir / 'train').glob('*.h5')))
    val_files   = sorted(list((data_dir / 'val').glob('*.h5')))
    test_files  = sorted(list((data_dir / 'test').glob('*.h5')))

    print(f"\nDataset ({data_dir}):")
    print(f"  train: {len(train_files)} H5 files")
    print(f"  val  : {len(val_files)} H5 files")
    print(f"  test : {len(test_files)} H5 files")

    # Print split manifest if available
    manifest_path = data_dir / 'split_manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        print(f"\nSplit manifest (seed={manifest.get('seed', '?')}):")
        print(f"  val  : {manifest.get('val', [])}")
        print(f"  test : {manifest.get('test', [])}")

    # Patch-based datasets (for training and loss monitoring)
    train_dataset = SegmentationDataset(
        train_files, patch_size=PATCH_SIZE, patches_per_volume=10,
        transform=get_transforms(True), cache_data=args.cache_data)
    val_dataset = SegmentationDataset(
        val_files, patch_size=PATCH_SIZE, patches_per_volume=10,
        transform=get_transforms(False), cache_data=args.cache_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader   = DataLoader(val_dataset,   batch_size=1,
                              shuffle=False, num_workers=max(1, args.num_workers // 2),
                              pin_memory=True, persistent_workers=args.num_workers > 0)

    # ── Model, loss, optimiser ─────────────────────────────────────────────
    model     = build_model().to(device)
    criterion = FourComponentLoss(seg_weight=1.0, brain_recon_weight=2.0,
                                  skin_super_weight=3.0, microglia_weight=5.0)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nLoss weights:")
    print(f"  Segmentation       : {criterion.seg_weight}")
    print(f"  Brain reconstruction: {criterion.brain_recon_weight}")
    print(f"  Skin supervision   : {criterion.skin_super_weight}")
    print(f"  Microglia protection: {criterion.microglia_weight}")

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch = 1
    best_fullbrain_dice = 0.0

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_fullbrain_dice = ckpt.get('fullbrain_dice', ckpt.get('best_dice', 0.0))
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best full-brain Dice so far: {best_fullbrain_dice:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────
    patience_counter = 0

    print("\n" + "=" * 80)
    print("TRAINING START")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)

        if epoch % args.val_every == 0:
            # 1. Patch-based validation (loss monitoring only)
            val_losses, patch_dice = validate_patches(
                model, val_loader, criterion, dice_metric, device)

            # 2. Full-brain sliding window validation (model selection!)
            print(f"\n  Full-brain validation ({len(val_files)} volumes):")
            fullbrain_dice = full_brain_validate(model, val_files, PATCH_SIZE, device)

            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train : total={train_losses['total']:.4f}  "
                  f"seg={train_losses['seg']:.4f}  "
                  f"brain={train_losses['brain_recon']:.4f}  "
                  f"skin={train_losses['skin_super']:.4f}  "
                  f"micro={train_losses['microglia']:.4f}")
            print(f"  Val   : total={val_losses['total']:.4f}  "
                  f"seg={val_losses['seg']:.4f}  "
                  f"brain={val_losses['brain_recon']:.4f}  "
                  f"skin={val_losses['skin_super']:.4f}  "
                  f"micro={val_losses['microglia']:.4f}")
            print(f"  Patch Dice     : {patch_dice:.4f}  (monitoring only — misleading metric)")
            print(f"  Full-brain Dice: {fullbrain_dice:.4f}  [MODEL SELECTION]  "
                  f"  patience={patience_counter}/{args.patience}")

            # Periodic checkpoint
            if epoch % args.ckpt_every == 0:
                ckpt_path = f"{args.model_dir}/checkpoint_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fullbrain_dice':       fullbrain_dice,
                    'patch_dice':           patch_dice,
                    'best_fullbrain_dice':  best_fullbrain_dice,
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

            # Best model (full-brain dice)
            if fullbrain_dice > best_fullbrain_dice:
                best_fullbrain_dice = fullbrain_dice
                best_path = f"{args.model_dir}/best_model_fullstack.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fullbrain_dice':       fullbrain_dice,
                    'patch_dice':           patch_dice,
                    'best_fullbrain_dice':  best_fullbrain_dice,
                }, best_path)
                torch.save(model.state_dict(), f"{args.model_dir}/best_model_weights.pth")
                print(f"  *** New best model! Full-brain Dice: {fullbrain_dice:.4f} "
                      f"-> {best_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{args.patience} val cycles)")

            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}: "
                      f"{patience_counter} consecutive val cycles without improvement "
                      f"(= {patience_counter * args.val_every} epochs)")
                break

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"Training complete.")
    print(f"  Best full-brain Dice: {best_fullbrain_dice:.4f}")
    print(f"  Best model: {args.model_dir}/best_model_fullstack.pth")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── TEST SET evaluation with best model ───────────────────────────────
    if test_files:
        print("\n" + "=" * 80)
        print(f"TEST SET EVALUATION ({len(test_files)} volumes)")
        print("=" * 80)
        print(f"Loading best model: {args.model_dir}/best_model_fullstack.pth")
        best_ckpt = torch.load(f"{args.model_dir}/best_model_fullstack.pth",
                               map_location=device)
        model.load_state_dict(best_ckpt['model_state_dict'])
        test_dice = full_brain_validate(model, test_files, PATCH_SIZE, device)
        print(f"\nTEST full-brain Dice: {test_dice:.4f}")
        print(f"(Best model was from epoch {best_ckpt['epoch']}, "
              f"val full-brain Dice={best_ckpt['fullbrain_dice']:.4f})")
    else:
        print("\nNo test files found — skipping test evaluation.")

    print("\n" + "=" * 80)
    print(f"Log: {log_file}")
    print("=" * 80)

    logger.close()


if __name__ == '__main__':
    main()
