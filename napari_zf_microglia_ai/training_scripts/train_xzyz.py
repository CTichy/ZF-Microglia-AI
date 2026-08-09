#!/usr/bin/env python3
"""Fine-tune cpsam on isotropic XY+XZ+YZ microglia patches."""

import argparse
import os, sys, time, logging, re
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from cellpose import models, train as cp_train

BUILTIN = {"cpsam", "cyto3", "cyto2", "cyto", "nuclei"}


def _erosion_distance(mask01, max_iter):
    """Per-pixel erosion-survival count (chessboard distance to background/boundary),
    capped at max_iter. Foreground pixels that erode away in few steps are thin
    (branch tips); pixels that survive every step are deep interior (soma)."""
    dist = torch.zeros_like(mask01)
    cur = mask01
    for i in range(1, max_iter + 1):
        eroded = -F.max_pool2d(-cur, kernel_size=3, stride=1, padding=1)
        newly_removed = (cur > 0.5) & (eroded <= 0.5)
        dist[newly_removed] = i
        cur = eroded
        if cur.sum() == 0:
            break
    dist[cur > 0.5] = max_iter + 1  # survived all erosions -> thick interior
    return dist


def make_branch_weighted_loss_fn(radius_thresh, weight_boost):
    """Drop-in replacement for cellpose.train._loss_fn_seg that up-weights thin
    structures (branch tips) in both the flow MSE and cellprob BCE terms, so the
    network is penalized as much for missing a branch tip as for missing soma."""

    def _loss_fn(lbl, y, device):
        mask01 = (lbl[:, -3:-2] > 0.5).float()  # [B,1,H,W]
        with torch.no_grad():
            dist = _erosion_distance(mask01, radius_thresh + 1)
            thin = (mask01 > 0.5) & (dist <= radius_thresh)
            w = torch.ones_like(mask01)
            w[thin] = 1.0 + weight_boost

        veci = 5.0 * lbl[:, -2:]
        se = (y[:, -3:-1] - veci) ** 2
        loss = (se * w).mean() / 2.0

        bce = F.binary_cross_entropy_with_logits(
            y[:, -1], mask01[:, 0], reduction="none"
        )
        loss2 = (bce * w[:, 0]).mean()
        return loss + loss2

    return _loss_fn

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",    required=True,
                     help="Directory of XZYZ training crops (e.g. from Extract XZYZ Patches).")
parser.add_argument("--model_name", default="cpsam_microglia_xzyz")
parser.add_argument("--pretrained", default="cpsam")
parser.add_argument("--n_epochs",   type=int, default=200)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--save_every", type=int,   default=10)
parser.add_argument("--log_every",  type=int,   default=5)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--branch_weight", type=float, default=0.0,
                    help="Extra weight (added to 1.0) for thin/branch-tip foreground "
                         "pixels in the loss. 0 = disabled (standard cellpose loss).")
parser.add_argument("--branch_radius", type=int, default=3,
                    help="Erosion-distance threshold (px) below which a foreground "
                         "pixel counts as thin/branch and gets the extra weight.")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = args.data_dir
MODEL_NAME  = args.model_name
N_EPOCHS    = args.n_epochs
BATCH_SIZE  = args.batch_size

if args.pretrained in BUILTIN or os.path.isabs(args.pretrained):
    PRETRAINED = args.pretrained
else:
    PRETRAINED = os.path.join(DATA_DIR, "models", args.pretrained)
LR           = args.lr
WD           = 1e-4
VAL_FRACTION = 0.15
SAVE_EVERY   = args.save_every
LOG_EVERY    = args.log_every
SEED         = 42

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = os.path.join(
    os.path.dirname(__file__), "logs",
    f"train_xzyz_{time.strftime('%Y%m%d_%H%M%S')}.log"
)
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger()

# ── Load data ─────────────────────────────────────────────────────────────────
log.info(f"Loading patches from {DATA_DIR}")
img_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".tif") and "_masks" not in f
)
log.info(f"  Found {len(img_files)} image files")

images, labels = [], []
skipped = 0
for fname in img_files:
    stem      = fname[:-4]
    img_path  = os.path.join(DATA_DIR, fname)
    mask_path = os.path.join(DATA_DIR, stem + "_masks.tif")
    if not os.path.exists(mask_path):
        skipped += 1
        continue

    img  = tifffile.imread(img_path)
    mask = tifffile.imread(mask_path).astype(np.int32)

    # Squeeze any singleton dims
    if img.ndim  > 2: img  = img.squeeze()
    if mask.ndim > 2: mask = mask.squeeze()
    if img.ndim != 2 or mask.ndim != 2:
        skipped += 1
        continue

    # Drop objects < 9 px (crash GPU flow computation)
    if mask.max() > 0:
        counts = np.bincount(mask.ravel())
        small  = np.where((counts > 0) & (counts < 9))[0]
        if len(small):
            mask[np.isin(mask, small)] = 0

    # Relabel to consecutive integers starting at 1
    unique = np.unique(mask[mask > 0])
    if len(unique):
        lut = np.zeros(int(unique.max()) + 1, dtype=np.int32)
        for new_id, old_id in enumerate(unique, start=1):
            lut[old_id] = new_id
        mask = lut[mask]

    images.append(img)
    labels.append(mask)

log.info(f"  Loaded {len(images)} pairs  ({skipped} skipped)")

# ── Train / val split ─────────────────────────────────────────────────────────
rng   = np.random.default_rng(SEED)
n_val = max(10, int(len(images) * VAL_FRACTION))
val_idx = set(rng.choice(len(images), size=n_val, replace=False).tolist())

tr_imgs, tr_lbls, va_imgs, va_lbls = [], [], [], []
for i, (img, lbl) in enumerate(zip(images, labels)):
    if i in val_idx:
        va_imgs.append(img); va_lbls.append(lbl)
    else:
        tr_imgs.append(img); tr_lbls.append(lbl)

log.info(f"  Train: {len(tr_imgs)}  Val: {len(va_imgs)}")

# ── Model ─────────────────────────────────────────────────────────────────────
log.info(f"Loading pretrained model: {PRETRAINED}")
model = models.CellposeModel(pretrained_model=PRETRAINED, gpu=True)
log.info(f"  GPU: {model.gpu}")

# ── Train ─────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info(f"Training: {MODEL_NAME}")
log.info(f"  n_epochs={N_EPOCHS}  lr={LR}  wd={WD}  batch={BATCH_SIZE}")
log.info(f"  save_every={SAVE_EVERY}  seed={SEED}")
log.info("=" * 60)

if args.branch_weight > 0:
    cp_train._loss_fn_seg = make_branch_weighted_loss_fn(args.branch_radius, args.branch_weight)
    log.info(f"Branch-weighted loss ACTIVE: foreground pixels within "
             f"{args.branch_radius}px of erosion (branch tips) get weight "
             f"x{1.0 + args.branch_weight:.1f} (vs x1.0 elsewhere)")
else:
    log.info("Branch-weighted loss: disabled (standard cellpose loss)")

# Intercept Cellpose's own stdout prints to capture test_loss + LR per epoch
_cp_stats = {}   # epoch -> {test_loss, lr}
_cp_pattern = re.compile(r"^\s*(\d+),\s*train_loss=([\d.]+),\s*test_loss=([\d.]+),\s*LR=([\d.]+)")

class _TeeCapture:
    def __init__(self, stream):
        self._s = stream
    def write(self, txt):
        self._s.write(txt)
        for line in txt.splitlines():
            m = _cp_pattern.match(line)
            if m:
                ep = int(m.group(1))
                _cp_stats[ep] = {"test": float(m.group(3)), "lr": float(m.group(4))}
    def flush(self): self._s.flush()
    def fileno(self): return self._s.fileno()

sys.stdout = _TeeCapture(sys.stdout)

model_path, train_losses, test_losses = cp_train.train_seg(
    model.net,
    train_data   = tr_imgs,
    train_labels = tr_lbls,
    test_data    = va_imgs,
    test_labels  = va_lbls,
    save_path    = DATA_DIR,
    model_name   = MODEL_NAME,
    n_epochs     = N_EPOCHS,
    learning_rate= LR,
    weight_decay = WD,
    batch_size   = BATCH_SIZE,
    nimg_per_epoch     = len(tr_imgs),
    nimg_test_per_epoch= len(va_imgs),
    save_every       = SAVE_EVERY,
    save_each        = True,
    channel_axis     = None,
    normalize        = True,
    min_train_masks  = 1,
)

sys.stdout = sys.stdout._s   # restore stdout

log.info("=" * 60)
log.info(f"Training complete — model saved to: {model_path}")
log.info("")
log.info(f"{'Epoch':>6}  {'train_loss':>10}  {'test_loss':>10}  {'LR':>10}")
for ep, tl in enumerate(train_losses):
    if ep % LOG_EVERY == 0 or ep == len(train_losses) - 1:
        cp = _cp_stats.get(ep, {})
        vl_str = f"{cp['test']:>10.4f}" if "test" in cp else f"{'—':>10}"
        lr_str = f"{cp['lr']:>10.6f}" if "lr"   in cp else f"{'—':>10}"
        log.info(f"{ep:>6}  {tl:>10.4f}  {vl_str}  {lr_str}")
log.info(f"Log: {log_path}")
