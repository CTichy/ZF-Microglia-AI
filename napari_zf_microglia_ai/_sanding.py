"""
_sanding.py -- Sigma-softening ("sanding") pass, chained onto the end of
Cellpose-SAM Segmentation's auto-correction stage (auto_contrast_correct_stack,
see _auto_correction.py).

Auto-correction re-derives each label's shape from a calibrated intensity
threshold -- accurate, but still built voxel-by-voxel, so contours can
come out blocky/jagged at the pixel scale. This pass runs AFTER that,
purely geometric (no image/intensity involved): each label's own binary
mask is Gaussian-blurred and re-thresholded at 0.5, rounding off small
jagged steps without meaningfully changing the cell's real shape or
volume. Same foreign-label protection as every other Correct Label tool
in this plugin -- a neighbor's already-claimed voxels can never be grown
into, so sanding one cell can never eat into another.
"""

from __future__ import annotations

import numpy as np

from ._labeling import sand_label, remove_debris


def sanding_pad(sigma_xy: float, sigma_z: float) -> int:
    """Bbox padding (voxels) that gives a Gaussian blur of the given
    sigmas room to round a boundary without being clipped by the crop
    edge -- shared by sand_labels_stack's own auto-pad and every single-
    label caller (Correct Label, Correct Adjacent Labels) so they all
    pad consistently for whatever sigma is currently dialed in."""
    return max(10, int(round(3 * max(sigma_xy, sigma_z))) + 3)


def sand_labels_stack(
    labels: np.ndarray,
    sigma_xy: float = 0.7,
    sigma_z: float = 0.7,
    pad: "int | None" = None,
    min_volume: "int | None" = None,
    final_min_fraction: float = 0.618,
    progress_cb=None,
) -> "tuple[np.ndarray, dict]":
    """
    Softens every label's contour in the stack. See the module docstring.

    labels              : (Z, Y, X) label volume
    sigma_xy, sigma_z    : Gaussian sigma in voxels (same units/meaning as
                          Create Labels' own Smooth sigma XY/Z) -- kept
                          small by default, this rounds off blocky voxel
                          edges, it isn't meant to reshape cells
    pad                  : bbox padding in voxels; None auto-picks enough
                          room for the given sigmas (3*max(sigma)+3, floor 10)
    min_volume, final_min_fraction : same final debris safety net as every
                          other stage in this plugin -- None skips it
    progress_cb           : optional callable(str), called with a
                          human-readable status line as each cell advances

    Returns (new_labels, report). report is a dict:
        sigma_xy, sigma_z       -- the sigmas actually used
        n_cells_total            -- labels present at the start
        n_cells_sanded           -- how many were actually softened
        skipped_cells            -- {label_id: reason} for every label
                                   sanding left untouched (never crashes
                                   the whole run)
        n_debris_fragments_removed -- fragments cleared by the final pass

    Raises ValueError only if no labels are present at all -- a failure
    on any individual cell is caught and recorded in the report instead.
    """
    def _report(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    labels = np.asarray(labels).astype(np.int32)
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]
    if unique_ids.size == 0:
        raise ValueError("no labels present -- nothing to sand")

    if pad is None:
        pad = sanding_pad(sigma_xy, sigma_z)

    new_labels = labels
    n_sanded = 0
    skipped_cells: "dict[int, str]" = {}
    n_total = int(unique_ids.size)
    for idx, lid in enumerate(unique_ids.tolist()):
        new_labels, info = sand_label(new_labels, int(lid), sigma_xy, sigma_z, pad=pad)
        if info["applied"]:
            n_sanded += 1
        else:
            skipped_cells[int(lid)] = info["reason"]
        if idx % 5 == 0 or idx == n_total - 1:
            _report(
                f"Sanding: {idx + 1}/{n_total} (label {lid}) -- "
                f"{n_sanded} sanded, {len(skipped_cells)} skipped so far"
            )

    n_debris_removed = 0
    if min_volume is not None:
        threshold = int(round(final_min_fraction * min_volume))
        _report(f"Sanding: removing debris below {threshold} vox...")
        new_labels, n_debris_removed = remove_debris(new_labels, threshold)

    report = {
        "sigma_xy": sigma_xy,
        "sigma_z": sigma_z,
        "n_cells_total": n_total,
        "n_cells_sanded": n_sanded,
        "skipped_cells": skipped_cells,
        "n_debris_fragments_removed": n_debris_removed,
    }
    return new_labels.astype(np.int32), report


def format_sanding_report(report: dict) -> str:
    lines = []
    lines.append(
        f"Sanding: sigma_xy={report['sigma_xy']:.2f}, sigma_z={report['sigma_z']:.2f}"
    )
    lines.append(
        f"  {report['n_cells_sanded']}/{report['n_cells_total']} cell(s) softened"
        + (f", {len(report['skipped_cells'])} skipped" if report["skipped_cells"] else "")
    )
    for lid, reason in report["skipped_cells"].items():
        lines.append(f"    label {lid} skipped: {reason}")
    lines.append(f"  Debris removed: {report['n_debris_fragments_removed']} fragment(s)")
    return "\n".join(lines)
