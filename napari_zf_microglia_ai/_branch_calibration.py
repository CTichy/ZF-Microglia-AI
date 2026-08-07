"""
_branch_calibration.py — measures the real branch radius of GT-labeled
microglia cells, to recalibrate train_xzyz.py's branch_radius parameter
(make_branch_weighted_loss_fn(radius_thresh, ...)) from actual measured
morphology instead of a guessed/frozen value.

branch_radius is a threshold on erosion-survival distance in pixels
(Chebyshev/chessboard distance via iterative max-pooling erosion),
applied to training crops that are effectively isotropic at the XY
resolution (~0.174 um/px native XY, Z-stretched to match in XZ/YZ
crops). This module measures branch radius in real microns via 3D
skeletonization + an anisotropic Euclidean distance transform, then
converts to pixels at the crop's XY scale for direct comparison against
the training parameter -- an approximation (Euclidean vs. Chebyshev
distance aren't identical), close enough to calibrate a threshold, not
an exact reconstruction.

skan API gotcha (found the hard way): skan.Skeleton(skeleton_image,
source_image=...)'s source_image is NOT used by summarize()'s
mean-pixel-value column -- that column is computed from the
skeleton_image array's own nonzero values. Fix: bake the EDT values
into the skeleton array itself before passing it in as skeleton_image.
"""

import numpy as np
from scipy.ndimage import find_objects, distance_transform_edt
from skimage.morphology import skeletonize
import skan


def measure_branch_diameters(gt_labels, scale_zyx=(1.0, 0.174, 0.174),
                              pad=5, min_volume=50, progress_cb=None):
    """
    Skeletonize every labeled cell in gt_labels, decompose into branch
    segments, and measure each segment's mean diameter (um) via an
    anisotropic EDT baked into the skeleton array.

    Returns dict: {
      'n_cells': int, 'n_segments': int,
      'diam_um': np.ndarray, 'length_um': np.ndarray,
    }
    """
    gt_labels = np.asarray(gt_labels)
    objs = find_objects(gt_labels)
    diam_um, length_um = [], []
    n_cells = 0
    Z, Y, X = gt_labels.shape
    for label_id in range(1, len(objs) + 1):
        sl = objs[label_id - 1]
        if sl is None:
            continue
        if progress_cb:
            progress_cb(f"cell {label_id}: measuring skeleton...")
        z0, z1 = max(0, sl[0].start - pad), min(Z, sl[0].stop + pad)
        y0, y1 = max(0, sl[1].start - pad), min(Y, sl[1].stop + pad)
        x0, x1 = max(0, sl[2].start - pad), min(X, sl[2].stop + pad)
        crop = gt_labels[z0:z1, y0:y1, x0:x1]
        binary = crop == label_id
        if binary.sum() < min_volume:
            continue
        edt = distance_transform_edt(binary, sampling=scale_zyx)
        skel = skeletonize(binary)
        if not skel.any():
            continue
        skel_valued = np.where(skel, edt, 0.0).astype(np.float64)
        try:
            sk = skan.Skeleton(skel_valued, spacing=scale_zyx)
            bd = skan.summarize(sk, separator="-")
        except Exception:
            continue
        if len(bd) == 0:
            continue
        diam_um.extend((2 * bd["mean-pixel-value"].values).tolist())
        length_um.extend(bd["euclidean-distance"].values.tolist())
        n_cells += 1

    return dict(
        n_cells=n_cells, n_segments=len(diam_um),
        diam_um=np.array(diam_um), length_um=np.array(length_um),
    )


def recommend_branch_radius(gt_labels, scale_zyx=(1.0, 0.174, 0.174), xy_scale=None,
                             pad=5, min_volume=50, progress_cb=None):
    """
    Full pipeline: measure branch diameters from gt_labels, then convert
    the thinnest-quartile radius (distal branch tips -- the structures
    branch_weight/branch_radius exist to protect) to pixels at xy_scale
    (defaults to scale_zyx's own XY component), rounded to the nearest
    integer pixel -- directly comparable to train_xzyz.py's branch_radius.

    Raises ValueError if no branch segments could be measured (e.g. an
    empty or non-cellular label volume).

    Returns dict: {..stats from measure_branch_diameters, plus..
      'tip_radius_um': float, 'recommended_branch_radius_px': int,
    }
    """
    stats = measure_branch_diameters(gt_labels, scale_zyx, pad, min_volume, progress_cb)
    if stats["n_segments"] == 0:
        raise ValueError("No branch segments could be measured from this GT — check the label volume.")
    if xy_scale is None:
        xy_scale = (scale_zyx[1] + scale_zyx[2]) / 2.0

    diam_um = stats["diam_um"]
    tip_diam_um = diam_um[diam_um <= np.percentile(diam_um, 25)]
    tip_radius_um = tip_diam_um.mean() / 2.0
    recommended_px = max(1, round(tip_radius_um / xy_scale))

    stats["tip_radius_um"] = float(tip_radius_um)
    stats["recommended_branch_radius_px"] = int(recommended_px)
    return stats


def format_branch_calibration_report(stats, current_branch_radius=None):
    """Plain-text summary, same spirit as the plugin's other sweep/
    calibration-tool reports."""
    diam_um = stats["diam_um"]
    lines = [
        f"Measured {stats['n_segments']} branch segments across {stats['n_cells']} cells.",
        f"Diameter: mean={diam_um.mean():.3f} um  median={np.median(diam_um):.3f} um  "
        f"min={diam_um.min():.3f}  max={diam_um.max():.3f}",
        f"Thinnest-quartile (distal branch tip) radius: {stats['tip_radius_um']:.3f} um "
        f"-> recommended branch_radius = {stats['recommended_branch_radius_px']} px at this scale",
    ]
    if current_branch_radius is not None:
        if stats["recommended_branch_radius_px"] == current_branch_radius:
            lines.append(f"\nCurrent branch_radius={current_branch_radius} matches the measured recommendation.")
        else:
            lines.append(
                f"\nCurrent branch_radius={current_branch_radius} differs from the measured "
                f"recommendation ({stats['recommended_branch_radius_px']})."
            )
    return "\n".join(lines)
