"""
_inference.py — MONAI 3D U-Net sliding-window inference.
"""

import warnings
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from scipy.ndimage import binary_erosion, binary_fill_holes, label

# skin_segmentation/ is two levels up from this file
_SKIN_SEG_DIR = Path(__file__).parent.parent

DEFAULT_MODEL = (
    _SKIN_SEG_DIR / "models" / "v1" / "best_model_fullstack_v1_epoch460_dice9573.pth"
)


def _load_model(model_path, device):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    ckpt = torch.load(str(model_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"   Model loaded from epoch {ckpt.get('epoch', '?')}: {model_path.name}")
    return model


def _normalize(volume):
    """Percentile normalisation matching inference_production.py."""
    p01 = np.percentile(volume, 1)
    p99 = np.percentile(volume, 99)
    return np.clip(
        (volume.astype(np.float32) - p01) / max(p99 - p01, 1.0),
        0, 1,
    )


def predict_probability(volume, model_path, device):
    """
    Just the expensive part of run_inference: sliding-window MONAI
    inference, returning the raw per-voxel probability map (0-1, before
    thresholding). Split out from run_inference so callers that need to
    try multiple thresholds (e.g. a GT-verification sweep) can run
    inference once and cheaply re-threshold/re-postprocess afterward,
    instead of reloading the model and re-running the sliding window for
    every candidate threshold.

    Returns pred_prob : (Z, Y, X) float32 ndarray, same shape as volume.
    """
    model = _load_model(model_path, device)

    volume_t = (
        torch.from_numpy(_normalize(volume))
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*non-tuple sequence.*multidimensional indexing.*",
        )
        pred_logits = sliding_window_inference(
            volume_t,
            roi_size=(64, 192, 192),
            sw_batch_size=4,
            predictor=model,
            overlap=0.25,
            mode="gaussian",
        )
        pred_prob = torch.sigmoid(pred_logits).cpu().numpy()[0, 0]

    del model, volume_t, pred_logits
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return pred_prob


def postprocess_probability(pred_prob, threshold):
    """
    Threshold + clean up a probability map (see predict_probability):
      1. Keep largest connected component  → drops isolated blobs and
         any outer tissue wrongly classified as brain
      2. Fill internal holes               → solid, contiguous brain
    Mirrors the polygon annotation tool's own output. This is the cheap
    part of inference -- safe to call repeatedly for different threshold
    candidates without re-running the network.

    Returns brain_mask : (Z, Y, X) uint8 (0/1), un-eroded.
    """
    raw_mask = pred_prob > threshold

    print("   Post-processing: largest component + fill holes...")
    labeled_arr, n_comp = label(raw_mask)
    if n_comp == 0:
        clean = raw_mask
    else:
        sizes = np.bincount(labeled_arr.ravel())
        sizes[0] = 0                          # ignore background label
        clean = labeled_arr == sizes.argmax()

    clean = binary_fill_holes(clean)
    brain_mask = clean.astype(np.uint8)
    print(
        f"   Components found: {n_comp}  →  kept largest"
        f"  ({brain_mask.sum():,} voxels, {100.*brain_mask.mean():.1f}% of volume)"
    )
    return brain_mask


def run_inference(volume, model_path, threshold, device, erosion_voxels=0):
    """
    Sliding-window MONAI inference on a 3D volume.

    Parameters
    ----------
    volume         : (Z, Y, X) np.ndarray
    model_path     : Path to .pth checkpoint
    threshold      : float in (0, 1)
    device         : torch.device
    erosion_voxels : int >= 0
        Erode the predicted mask by this many voxels before applying it to
        produce brain_only.  The raw (un-eroded) mask is always returned as
        brain_mask so the TIF save is unaffected.  0 = no erosion.

    Returns
    -------
    brain_mask  : (Z, Y, X) uint8   — raw predicted mask (0/1), no erosion.
                  Always the un-eroded mask, e.g. for saving brain_mask.tif.
    brain_only  : (Z, Y, X) same dtype as input
                  volume × eroded_mask  (skin + outer-rim tissue zeroed out)
    eroded_mask : (Z, Y, X) uint8   — the mask actually used to build
                  brain_only (== brain_mask when erosion_voxels == 0).
                  Callers doing further mask-based processing on brain_only
                  (e.g. background removal) should use this, not brain_mask,
                  or erosion is silently discarded downstream.
    """
    pred_prob = predict_probability(volume, model_path, device)
    brain_mask = postprocess_probability(pred_prob, threshold)

    # ------------------------------------------------------------------ #
    # Erosion (optional) — strips outer skin rim from brain_only
    # brain_mask is always saved un-eroded
    # ------------------------------------------------------------------ #
    if erosion_voxels > 0:
        print(f"   Eroding mask by {erosion_voxels} voxel(s)...")
        eroded_mask = binary_erosion(brain_mask, iterations=erosion_voxels).astype(np.uint8)
        vox_removed = int(brain_mask.sum()) - int(eroded_mask.sum())
        print(f"   Erosion removed {vox_removed:,} voxels from brain_only mask.")
        brain_only = (volume * eroded_mask).astype(volume.dtype)
    else:
        eroded_mask = brain_mask
        brain_only = (volume * brain_mask).astype(volume.dtype)

    return brain_mask, brain_only, eroded_mask
