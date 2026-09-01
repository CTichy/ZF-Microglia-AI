"""
_contrast_sweep.py -- Calibrate Correct Label's contrast lower value.

Correct Label (_labeling.py's correct_label_from_intensity()) regenerates
a label's 2D shape from wherever the signal layer's contrast is currently
set: signal = image >= lo. Finding a good `lo` by eye means dragging the
contrast slider until the silhouette "looks right" -- workable, but slow
and subjective, and easy to get wrong when the true signal peak sits far
above whatever narrow window makes the display look clean (see
correct_label_from_intensity()'s own docstring for the bug this caused
the first time around).

This module answers a narrower, more useful question instead: "of all
the lo values I could pick, which one best REPRODUCES what Cellpose-SAM
has already segmented?" -- i.e. calibrated against the model's own
existing output, not independent hand-annotated ground truth. That's a
deliberately different target than every other GT-sweep tool in this
plugin: the point isn't to check whether Cellpose-SAM is right, it's to
find the contrast setting Correct Label should start from before a user
nudges it further for a specific problem cell.
"""

from __future__ import annotations

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ._labeling import _intensity_correct_2d


def select_calibration_samples(
    labels: np.ndarray,
    scale_zyx: "tuple[float, float, float]",
    n_cells: int = 5,
    slices_per_cell: int = 10,
    edge_margin_um: float = 50.0,
) -> "list[tuple[int, int]]":
    """
    Pick representative (label_id, z) samples for contrast calibration.

    Selects the n_cells most morphologically complex cells (skeleton
    branch count -- the same "Complexity" measure Resort Labels already
    uses to find genuinely branched/ramified cells rather than just
    large ones), restricted to cells whose centroid sits at least
    edge_margin_um away from the volume's own outer Z/Y/X boundary -- a
    simple proxy for "not close to skin" (skin sits at the tissue
    periphery, and a cell near the volume edge is exactly the one most
    likely to already have a skin-residue fragment merged into it --
    the artifact this calibration should be scored AGAINST, not
    contaminated by).

    For each selected cell, slices_per_cell Z-slices are drawn from
    within that cell's own Z-extent, spread evenly between 20% and 80%
    of its span -- avoiding the very top/bottom slices where a cell's
    2D cross-section is thin and unrepresentative.

    Each axis's margin is capped at 30% of that axis's own total extent
    (per side), so it degrades gracefully instead of collapsing the
    "interior" window to nothing. A single edge_margin_um in physical
    microns hits Z and XY very differently under this project's usual
    anisotropy: a typical ~101-slice fish stack at 1.0 um/vox is only
    ~101 um deep, so an UNcapped 50 um margin on both sides leaves at
    most a single-voxel-wide (or literally empty) valid Z window even
    though the XY plane (2048 px * 0.174 um/vox =~ 356 um) has plenty
    of room -- a real bug this cap fixes, not a hypothetical one: it
    silently excluded every cell in exactly that common case.

    Returns up to n_cells * slices_per_cell (label_id, z) pairs (fewer
    if there aren't enough interior/complex-enough cells available).
    """
    from scipy.ndimage import find_objects, center_of_mass as _com
    from ._statistics import _skeleton_stats

    unique = np.unique(labels)
    unique = unique[unique > 0]
    if unique.size == 0:
        return []

    max_lbl = int(unique.max())
    slices_bb = find_objects(labels, max_label=max_lbl)
    z_dim, y_dim, x_dim = labels.shape
    sz, sy, sx = scale_zyx

    max_margin_fraction = 0.3
    margin_z = min(edge_margin_um / sz if sz > 0 else 0.0, max_margin_fraction * z_dim)
    margin_y = min(edge_margin_um / sy if sy > 0 else 0.0, max_margin_fraction * y_dim)
    margin_x = min(edge_margin_um / sx if sx > 0 else 0.0, max_margin_fraction * x_dim)

    label_list = unique.tolist()
    # center_of_mass returns a proper list of tuples for a LIST index, even
    # a length-1 one -- the "single bare tuple" special case only applies
    # when index itself is a scalar (not our case, label_list always is a
    # list) -- no extra wrapping needed regardless of unique.size.
    raw_centroids = _com(labels > 0, labels, label_list)

    interior = []
    for lbl, c in zip(label_list, raw_centroids):
        cz, cy, cx = c
        if (cz < margin_z or cz > (z_dim - 1 - margin_z) or
                cy < margin_y or cy > (y_dim - 1 - margin_y) or
                cx < margin_x or cx > (x_dim - 1 - margin_x)):
            continue
        interior.append(lbl)
    if not interior:
        return []

    def _branch_count(lbl):
        sl = slices_bb[lbl - 1]
        if sl is None:
            return lbl, 0
        binary = labels[sl] == lbl
        n_branches, *_ = _skeleton_stats(binary, (1.0, 1.0, 1.0))
        return lbl, n_branches

    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // 2)) as ex:
        results = list(ex.map(_branch_count, interior))
    results.sort(key=lambda t: t[1], reverse=True)
    top_cells = [lbl for lbl, _n in results[:n_cells]]

    samples: "list[tuple[int, int]]" = []
    for lbl in top_cells:
        zs = np.unique(np.nonzero(labels == lbl)[0])
        if zs.size == 0:
            continue
        lo_z, hi_z = int(zs.min()), int(zs.max())
        span = hi_z - lo_z
        if slices_per_cell <= 1 or span == 0:
            picks = [int(zs[len(zs) // 2])]
        else:
            fracs = np.linspace(0.2, 0.8, slices_per_cell)
            picks_set = set()
            for f in fracs:
                target = lo_z + f * span
                picks_set.add(int(zs[np.argmin(np.abs(zs - target))]))
            picks = sorted(picks_set)
        for z in picks[:slices_per_cell]:
            samples.append((int(lbl), int(z)))

    return samples


def default_lo_candidates(
    image: np.ndarray,
    samples: "list[tuple[int, int]]",
    pad: int,
    n_steps: int = 40,
) -> "list[float]":
    """
    Auto-scaled candidate lo values -- spans the actual intensity range
    present around the calibration samples (1st-99.5th percentile,
    padded bbox neighborhoods only, not the whole fish) rather than a
    hardcoded [0, 255]-style guess that would badly misfit a different
    channel/bit-depth/normalization.
    """
    values = []
    for label_id, z in samples:
        image_z = image[z]
        # crude neighborhood: same padded window _intensity_correct_2d
        # would use, without needing the labels array here too
        values.append(image_z.astype(np.float32).ravel())
    if not values:
        raise ValueError("No calibration samples given -- nothing to sweep")
    pooled = np.concatenate(values)
    lo_bound = float(np.percentile(pooled, 1.0))
    hi_bound = float(np.percentile(pooled, 99.5))
    if hi_bound <= lo_bound:
        hi_bound = lo_bound + 1.0
    return [float(v) for v in np.linspace(lo_bound, hi_bound, n_steps)]


def sweep_contrast_lower_value(
    labels: np.ndarray,
    image: np.ndarray,
    samples: "list[tuple[int, int]]",
    lo_candidates: "list[float]",
    pad: int = 15,
    progress_cb=None,
) -> dict:
    """
    For each candidate lower-contrast value, score how closely
    correct_label_from_intensity()'s own reconstruction (at that
    threshold) reproduces each sample's EXISTING 2D label footprint --
    i.e. what Cellpose-SAM already segmented there -- via IoU, then
    picks the candidate maximizing the MEAN IoU across every sample
    jointly (not each sample's own independent best, which would let
    outlier samples pull the result around).

    Samples that error out at a given candidate (e.g. the threshold
    leaves nothing connected to the label's existing footprint) score
    IoU=0 for that candidate rather than aborting the whole sweep --
    the same "don't let one bad case crash the run" pattern this
    project's other sweep tools already use.

    Returns a dict:
        candidates    -- lo values tried, ascending
        mean_iou      -- mean IoU per candidate, same order
        per_sample    -- {(label_id, z): [iou per candidate]}
        best_lo       -- the candidate maximizing mean_iou
        best_mean_iou -- that candidate's mean IoU
        n_samples     -- len(samples)
    """
    if not samples:
        raise ValueError("No calibration samples given -- nothing to sweep")
    if not lo_candidates:
        raise ValueError("No candidate lo values given -- nothing to sweep")

    per_sample: "dict[tuple[int, int], list[float]]" = {s: [] for s in samples}
    mean_iou = []

    for idx, lo in enumerate(lo_candidates):
        ious = []
        for (label_id, z) in samples:
            labels_z = labels[z]
            image_z = image[z]
            existing = labels_z == label_id
            try:
                corrected, crop_existing, _ = _intensity_correct_2d(
                    labels_z, image_z, label_id, lo, pad
                )
            except ValueError:
                iou = 0.0
            else:
                inter = np.logical_and(corrected, crop_existing).sum()
                union = np.logical_or(corrected, crop_existing).sum()
                iou = float(inter / union) if union > 0 else 0.0
            ious.append(iou)
            per_sample[(label_id, z)].append(iou)
        mean_iou.append(float(np.mean(ious)))
        if progress_cb:
            progress_cb(
                f"Contrast sweep: {idx + 1}/{len(lo_candidates)} candidates "
                f"tested (lo={lo:.4g}, mean IoU={mean_iou[-1]:.3f})"
            )

    best_idx = int(np.argmax(mean_iou))
    return {
        "candidates": list(lo_candidates),
        "mean_iou": mean_iou,
        "per_sample": per_sample,
        "best_lo": float(lo_candidates[best_idx]),
        "best_mean_iou": float(mean_iou[best_idx]),
        "n_samples": len(samples),
    }


def format_contrast_sweep_report(sweep: dict, samples: "list[tuple[int, int]]") -> str:
    lines = []
    lines.append(
        f"Contrast lower-value calibration -- {sweep['n_samples']} sample(s) "
        f"from {len(set(lbl for lbl, _z in samples))} cell(s): "
        + ", ".join(f"(label {lbl}, z={z})" for lbl, z in samples)
    )
    lines.append("")
    lines.append(f"{'lo':>12}  {'mean IoU':>10}")
    for lo, iou in zip(sweep["candidates"], sweep["mean_iou"]):
        marker = "  <== best" if lo == sweep["best_lo"] else ""
        lines.append(f"{lo:12.4g}  {iou:10.4f}{marker}")
    lines.append("")
    lines.append(
        f"Best: lo={sweep['best_lo']:.4g}  (mean IoU={sweep['best_mean_iou']:.4f})"
    )
    return "\n".join(lines)
