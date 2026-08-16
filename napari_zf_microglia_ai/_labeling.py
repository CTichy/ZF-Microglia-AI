"""
_labeling.py — True 3D connected component labeling.

Backend priority
----------------
1. CUDA  — CuPy + cupyx.scipy.ndimage  (full GPU path)
2. MPS   — Apple Silicon Metal          (threaded CPU; MPS lacks ndimage ops)
3. CPU   — scipy.ndimage + ThreadPool   (multithreaded, portable fallback)

Workflow
--------
1. Binary mask  : volume > 0
2. Gaussian smooth (σ_xy, σ_z) → re-threshold at 0.5
3. Fill holes per Z slice
4. 3D connected components (26-connectivity via ones(3,3,3) structure)
5. Remove blobs < final_min_fraction * min_volume voxels (golden ratio
   safety-net relaxation by default, see create_labels()'s docstring)
6. Renumber 1…N by descending volume  (label 1 = largest)
"""

from __future__ import annotations

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import (
    gaussian_filter as cpu_gaussian,
    label        as cpu_label,
    binary_fill_holes as cpu_fill_holes,
)
from skimage.morphology import remove_small_holes as _cpu_remove_small_holes


# ─────────────────────────────────────────────────────────────────────────────
# Backend detection  (run once at import time)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_backend() -> tuple[str, object, object]:
    """Return (backend_name, cupy_module, cupyx_ndimage_module)."""

    # ── CUDA via CuPy ──────────────────────────────────────────────────────
    import io, sys
    _saved, sys.stdout = sys.stdout, io.StringIO()
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cpnd
        # Exercise NVRTC/JIT so a broken install fails here, not mid-run
        _t = cp.zeros((4, 4), dtype=cp.float32)
        cpnd.gaussian_filter(_t, sigma=1.0)
        return "cuda", cp, cpnd
    except Exception:
        pass
    finally:
        sys.stdout = _saved

    # ── Apple Silicon MPS ──────────────────────────────────────────────────
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps", None, None
    except Exception:
        pass

    return "cpu", None, None


_BACKEND, _CP, _CPND = _detect_backend()
_N_THREADS = max(1, (os.cpu_count() or 4) // 2)


def _free_gpu_cache() -> None:
    """Free CuPy and PyTorch GPU memory pools to prevent OOM."""
    if _CP is not None:
        try:
            _CP.get_default_memory_pool().free_all_blocks()
            _CP.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CUDA path
# ─────────────────────────────────────────────────────────────────────────────

def _fill_holes_capped_gpu(slice_gpu, min_hole_size, cp, cpnd):
    """Per-slice hole fill: a hole survives as background only if its area
    is >= min_hole_size voxels; anything smaller is filled as noise.
    <=0 means no floor -- fills every enclosed background region,
    matching the old unconditional binary_fill_holes behaviour exactly."""
    if min_hole_size <= 0:
        return cpnd.binary_fill_holes(slice_gpu)
    filled = cpnd.binary_fill_holes(slice_gpu)
    holes = filled & (~slice_gpu)
    if not bool(holes.any()):
        return slice_gpu
    hole_labels, n_holes = cpnd.label(holes)
    if n_holes == 0:
        return slice_gpu
    counts = cp.bincount(hole_labels.ravel().astype(cp.int64), minlength=n_holes + 1)
    fillable = counts < min_hole_size
    fillable[0] = False
    return slice_gpu | fillable[hole_labels]


def _create_labels_cuda(
    volume: np.ndarray,
    sigma_xy: float,
    sigma_z: float,
    min_volume: int,
    min_hole_size: int = 0,
    final_min_fraction: float = 0.618,
) -> np.ndarray:
    cp   = _CP
    cpnd = _CPND
    Z, Y, X = volume.shape

    # ── Steps 1–2: binary → Gaussian smooth → re-threshold ────────────────
    vol_gpu     = cp.asarray(volume, dtype=cp.float32)
    binary_gpu  = (vol_gpu > 0).astype(cp.float32)
    blurred_gpu = cpnd.gaussian_filter(binary_gpu, sigma=(sigma_z, sigma_xy, sigma_xy))
    smooth_gpu  = blurred_gpu > 0.5
    del vol_gpu, binary_gpu, blurred_gpu

    print(f"   σ_xy={sigma_xy:.1f}  σ_z={sigma_z:.1f}  "
          f"signal voxels: {int(smooth_gpu.sum()):,}")

    # ── Step 3: fill holes per slice (GPU loop), floored at min_hole_size ──
    for z in range(Z):
        smooth_gpu[z] = _fill_holes_capped_gpu(smooth_gpu[z], min_hole_size, cp, cpnd)

    # ── Step 4: true 3D connected components (26-connectivity) ────────────
    structure  = cp.ones((3, 3, 3), dtype=cp.int32)
    labeled_gpu, n_objects = cpnd.label(smooth_gpu, structure=structure)
    del smooth_gpu
    n_objects = int(n_objects)
    print(f"   3D blobs: {n_objects}")

    if n_objects == 0:
        result = labeled_gpu.get().astype(np.int32)
        del labeled_gpu
        _free_gpu_cache()
        return result

    # ── Step 5: remove small blobs — vectorised on GPU ────────────────────
    # Final cutoff is final_min_fraction * min_volume, not min_volume
    # itself -- same golden-ratio safety-net philosophy _cellpose_seg.py's
    # final_min_size_cleanup() uses, applied here since this route has no
    # merge/reattach stage of its own to leave a gray-zone object standing
    # for a later stage to reconsider (unlike Cellpose-SAM's GMM/safe-
    # merge/large-contact chain): min_volume alone as a hard cutoff would
    # discard a legitimately smaller-than-average real cell just as
    # readily as real debris. final_min_fraction=1.0 recovers the exact
    # historical behaviour (cutoff == min_volume) for any caller that
    # doesn't pass a fraction.
    threshold = max(1, round(final_min_fraction * min_volume))
    max_out = int(labeled_gpu.max())
    counts  = cp.bincount(labeled_gpu.ravel().astype(cp.int64), minlength=max_out + 1)

    keep_lut    = counts >= threshold
    keep_lut[0] = True
    output_gpu  = cp.where(keep_lut[labeled_gpu], labeled_gpu, cp.int32(0))
    removed     = int(((counts[1:] > 0) & (counts[1:] < threshold)).sum())
    del labeled_gpu

    # ── Step 6: renumber 1…N by descending volume ─────────────────────────
    remaining      = cp.unique(output_gpu[output_gpu > 0]).get().tolist()
    counts_cpu     = counts.get()
    volumes_sorted = sorted(
        [(int(counts_cpu[lbl]), int(lbl)) for lbl in remaining], reverse=True
    )
    max_out2 = int(output_gpu.max())
    lut2     = np.zeros(max_out2 + 1, dtype=np.int32)
    for new_id, (_vol, old_id) in enumerate(volumes_sorted, start=1):
        lut2[old_id] = new_id

    output  = cp.asarray(lut2)[output_gpu].get()
    n_final = int(output.max())
    print(f"   3D blobs removed (< {threshold} vox = {final_min_fraction:.3f} x min_volume {min_volume}): {removed}")
    print(f"   Final 3D labels: {n_final}  (label 1 = largest)")
    del output_gpu, counts, keep_lut
    _free_gpu_cache()
    return output.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Threaded CPU path  (also used for Apple MPS — MPS lacks ndimage ops)
# ─────────────────────────────────────────────────────────────────────────────

def _create_labels_threaded(
    volume: np.ndarray,
    sigma_xy: float,
    sigma_z: float,
    min_volume: int,
    min_hole_size: int = 0,
    final_min_fraction: float = 0.618,
) -> np.ndarray:
    Z, Y, X = volume.shape

    # ── Steps 1–2: Gaussian smooth (scipy already multi-threaded internally)
    binary      = (volume > 0).astype(np.float32)
    blurred     = cpu_gaussian(binary, sigma=(sigma_z, sigma_xy, sigma_xy))
    smooth_mask = blurred > 0.5
    del binary, blurred

    print(f"   σ_xy={sigma_xy:.1f}  σ_z={sigma_z:.1f}  "
          f"signal voxels: {int(smooth_mask.sum()):,}")

    # ── Step 3: fill holes per slice in parallel, floored at min_hole_size ─
    # min_hole_size<=0 keeps the old unconditional-fill behaviour exactly
    # (cpu_fill_holes fills every enclosed background region regardless of
    # size); a positive floor switches to skimage's area-limited fill,
    # which leaves any hole at or above the floor as real background
    # instead of erasing it.
    def _fill_slice(args: tuple) -> tuple:
        z, slc = args
        if min_hole_size <= 0:
            return z, cpu_fill_holes(slc)
        return z, _cpu_remove_small_holes(slc, area_threshold=min_hole_size)

    with ThreadPoolExecutor(max_workers=_N_THREADS) as pool:
        results = list(pool.map(_fill_slice, [(z, smooth_mask[z]) for z in range(Z)]))
    del smooth_mask

    results.sort(key=lambda r: r[0])
    filled_3d = np.stack([r[1] for r in results])

    # ── Step 4: true 3D connected components (26-connectivity) ────────────
    structure         = np.ones((3, 3, 3), dtype=np.int32)
    labeled, n_objects = cpu_label(filled_3d, structure=structure)
    del filled_3d
    print(f"   3D blobs: {n_objects}")

    if n_objects == 0:
        return labeled.astype(np.int32)

    # ── Step 5: remove small blobs ────────────────────────────────────────
    # See _create_labels_cuda's matching comment: the deletion cutoff is
    # final_min_fraction * min_volume, not min_volume itself.
    threshold = max(1, round(final_min_fraction * min_volume))
    max_out = int(labeled.max())
    counts  = np.bincount(labeled.ravel().astype(np.int64), minlength=max_out + 1)

    keep_lut    = counts >= threshold
    keep_lut[0] = True
    output      = np.where(keep_lut[labeled], labeled, 0).astype(np.int32)
    removed     = int(((counts[1:] > 0) & (counts[1:] < threshold)).sum())
    del labeled

    # ── Step 6: renumber 1…N by descending volume ─────────────────────────
    remaining      = np.unique(output[output > 0]).tolist()
    volumes_sorted = sorted(
        [(int(counts[lbl]), int(lbl)) for lbl in remaining], reverse=True
    )
    max_out2 = int(output.max())
    lut2     = np.zeros(max_out2 + 1, dtype=np.int32)
    for new_id, (_vol, old_id) in enumerate(volumes_sorted, start=1):
        lut2[old_id] = new_id

    output  = lut2[output]
    n_final = int(output.max())
    print(f"   3D blobs removed (< {threshold} vox = {final_min_fraction:.3f} x min_volume {min_volume}): {removed}")
    print(f"   Final 3D labels: {n_final}  (label 1 = largest)")
    return output.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def resort_labels(
    labels: np.ndarray,
    sort_by: str = "size",
    reverse: bool = False,
) -> np.ndarray:
    """
    Renumber labels 1…N by the chosen criterion.

    Parameters
    ----------
    labels   : (Z, Y, X) int32 ndarray — existing label volume (0 = background)
    sort_by  : "size" | "centroid_z" | "centroid_y" | "centroid_x"
    reverse  : reverse the natural sort order
                 size      — natural = descending (largest = label 1)
                 centroid  — natural = ascending  (smallest coord = label 1)

    Returns
    -------
    (Z, Y, X) int32 ndarray — same objects, renumbered 1…N
    """
    from scipy.ndimage import center_of_mass as _com

    unique = np.unique(labels)
    unique = unique[unique > 0]
    if unique.size == 0:
        return labels.copy()

    label_list = unique.tolist()
    max_lbl    = int(unique.max())

    if sort_by == "size":
        counts = np.bincount(labels.ravel().astype(np.int64), minlength=max_lbl + 1)
        keyed  = [(int(counts[lbl]), int(lbl)) for lbl in label_list]
        # natural: descending (largest first → label 1)
        keyed.sort(key=lambda t: t[0], reverse=not reverse)
    else:
        axis_map = {"centroid_z": 0, "centroid_y": 1, "centroid_x": 2}
        axis     = axis_map[sort_by]
        raw      = _com(labels > 0, labels, label_list)
        if unique.size == 1:
            raw = [raw]   # scipy returns a single tuple when only one label
        keyed = [(float(c[axis]), int(lbl)) for lbl, c in zip(label_list, raw)]
        # natural: ascending (smallest coordinate first → label 1)
        keyed.sort(key=lambda t: t[0], reverse=reverse)

    lut = np.zeros(max_lbl + 1, dtype=np.int32)
    for new_id, (_key, old_id) in enumerate(keyed, start=1):
        lut[old_id] = new_id

    return lut[labels].astype(np.int32)


def remove_debris(labels: np.ndarray, threshold: int) -> "tuple[np.ndarray, int]":
    """Zero out every spatially-connected fragment smaller than threshold
    voxels. A manual edit in napari -- deleting a whole label that turned
    out to be misclassified skin, splitting a label, painting part of
    one away -- can leave small disconnected fragments behind that never
    went through create_labels()'s own volume filter (which only ever
    ran once, before the edit). This is that same filter, callable again
    on demand against whatever the Labels layer currently looks like.

    Deliberately evaluated **per connected component, not per raw label
    ID** -- a naive per-label-ID voxel count (bincount over the whole
    array) silently missed exactly the case this function exists for: a
    manually-erased label's own small leftover fragments still carry
    that same original ID, and summing their voxels together can clear
    the threshold even though each individual disconnected piece is
    genuine debris on its own. 26-connectivity, matching create_labels()'s
    own connected-component convention.

    Deliberately does NOT renumber surviving labels -- unlike
    create_labels()'s own filter, this is a targeted cleanup step on an
    already-in-use label set, not a fresh labeling pass, so IDs the user
    may already be tracking (via Split Label, manual annotation, etc.)
    are left exactly as they were. A label ID with one real surviving
    piece and one small disconnected debris piece keeps its ID on the
    real piece; only the debris piece is zeroed.

    threshold : typically final_min_fraction * min_volume (Common
    Settings), the same golden-ratio-relaxed cutoff create_labels()'s
    own filter and Cellpose-SAM's final_min_size_cleanup() already use.

    Returns (labels, n_removed) -- n_removed counts fragments, not label
    IDs (one ID can contribute more than one removed fragment)."""
    from scipy.ndimage import label as _cc_label, find_objects

    labels = np.asarray(labels)
    max_lbl = int(labels.max()) if labels.size else 0
    if max_lbl == 0:
        return labels.copy(), 0

    out = labels.copy()
    structure = np.ones((3, 3, 3), dtype=np.int32)
    objs = find_objects(labels)
    n_removed = 0
    for lbl in range(1, max_lbl + 1):
        sl = objs[lbl - 1] if lbl - 1 < len(objs) else None
        if sl is None:
            continue
        crop = out[sl]
        mask = crop == lbl
        if not mask.any():
            continue
        cc, n_cc = _cc_label(mask, structure=structure)
        if n_cc <= 1:
            if int(mask.sum()) < threshold:
                crop[mask] = 0
                n_removed += 1
            continue
        counts = np.bincount(cc.ravel())
        for piece_id in range(1, n_cc + 1):
            if counts[piece_id] < threshold:
                crop[cc == piece_id] = 0
                n_removed += 1

    return out.astype(np.int32), n_removed


def split_label(
    labels: np.ndarray,
    target_label: int,
    n_splits: int = 2,
    sigma: float = 1.0,
    min_distance: int = 5,
) -> "tuple[np.ndarray, list[int]]":
    """
    Split one label into n_splits parts using watershed on the distance transform.

    The boundary is placed where the object is narrowest — the saddle point
    of the distance map between the local maxima.

    Speed notes
    -----------
    - All operations run on the bounding box of the target label, not the
      full volume — critical for large stacks.
    - Gaussian smoothing runs on GPU (CuPy) when available.
    - Seed assignment runs on GPU via Euclidean nearest-seed (CuPy); falls
      back to CPU watershed if GPU is unavailable or runs out of memory.

    Parameters
    ----------
    labels       : (Z, Y, X) int32 ndarray
    target_label : label value to split
    n_splits     : number of parts to produce (≥ 2)
    sigma        : Gaussian smoothing of distance map (higher = broader peaks)
    min_distance : minimum voxel distance between seed peaks

    Returns
    -------
    (new_labels, new_ids)
        new_labels — same shape as labels, blob split into n_splits parts
        new_ids    — list of n_splits-1 new label IDs created
                     (target_label is kept for part 1)

    Raises
    ------
    ValueError  if the label is not found, or fewer peaks than n_splits found
    """
    from scipy.ndimage import distance_transform_edt

    mask = labels == target_label
    if not np.any(mask):
        raise ValueError(f"Label {target_label} not found")

    # ── 1. Crop to bounding box (avoids running EDT on full volume) ────────
    nz  = np.argwhere(mask)
    lo  = nz.min(axis=0)
    hi  = nz.max(axis=0)
    pad = max(int(min_distance), int(sigma) + 2, 2)
    lo_p = np.maximum(lo - pad, 0)
    hi_p = np.minimum(hi + pad, np.array(mask.shape) - 1)
    sl   = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo_p, hi_p))

    mask_crop = mask[sl]

    # ── 2. Distance transform (CPU — not in cupyx) ─────────────────────────
    dist = distance_transform_edt(mask_crop).astype(np.float32)

    # ── 3. Gaussian smoothing — GPU if available, CPU fallback ─────────────
    if _BACKEND == "cuda" and _CP is not None:
        try:
            dist_gpu  = _CP.asarray(dist)
            dist_gpu  = _CPND.gaussian_filter(dist_gpu, sigma=float(sigma))
            dist_smooth = dist_gpu.get()
            del dist_gpu
            _free_gpu_cache()
            print(f"   Split: Gaussian smooth on GPU")
        except Exception as exc:
            print(f"   Split: GPU smooth failed ({exc}), using CPU")
            dist_smooth = cpu_gaussian(dist, sigma=float(sigma)) if sigma > 0 else dist
    else:
        dist_smooth = cpu_gaussian(dist, sigma=float(sigma)) if sigma > 0 else dist

    # ── 4. Seed detection via h-maxima (topological prominence) ────────────
    #
    #    peak_local_max uses Euclidean distance — it fails when two big chunks
    #    are spatially close (thin neck) because their centres may be within
    #    min_distance of each other.
    #
    #    h_maxima finds peaks that stand at least h ABOVE their lowest saddle
    #    to any higher peak.  The thin neck IS that saddle, so the two chunk
    #    centres are always separated regardless of their Euclidean distance.
    #
    #    We auto-reduce h (starting at 50% of max EDT) until >= n_splits
    #    topologically distinct peaks are found.  Each peak is then placed at
    #    the EDT maximum inside its h-maxima connected region.
    #
    #    min_distance is used as a final Euclidean guard: if two chosen seeds
    #    are closer than min_distance voxels, the weaker one is dropped.
    from skimage.morphology import h_maxima
    from scipy.ndimage import label as _nd_label

    dist_in_mask = dist_smooth * mask_crop.astype(np.float32)
    max_dist = float(dist_in_mask.max())
    if max_dist == 0:
        raise ValueError(f"Label {target_label}: distance transform is zero — blob too flat?")

    # Iteratively reduce h until >= n_splits prominent peaks found
    h_val   = max_dist * 0.50
    h_floor = max_dist * 0.005          # never go below 0.5 % of max EDT
    labeled_hmax = None
    n_found = 0
    while h_val >= h_floor:
        hmax = h_maxima(dist_in_mask, h=float(h_val))
        labeled_hmax, n_found = _nd_label(hmax)
        if n_found >= n_splits:
            break
        h_val *= 0.75

    if n_found < n_splits:
        raise ValueError(
            f"Only {n_found} distinct sub-volume(s) found — "
            f"try reducing Smooth σ"
        )

    # For each h-maxima region pick the voxel with the highest EDT value
    region_peaks = []
    for i in range(1, n_found + 1):
        region_dist = np.where(labeled_hmax == i, dist_in_mask, 0.0)
        coord       = np.array(np.unravel_index(region_dist.argmax(), region_dist.shape))
        peak_val    = float(dist_in_mask[tuple(coord)])
        region_vol  = int((labeled_hmax == i).sum())
        region_peaks.append((peak_val, region_vol, coord))

    # Sort by EDT peak value (thickest chunk centre first) then apply
    # Euclidean min_distance guard to avoid two seeds in the same chunk
    region_peaks.sort(key=lambda t: t[0], reverse=True)
    seeds = []
    for peak_val, _vol, coord in region_peaks:
        if all(np.linalg.norm(coord - s) >= min_distance for s in seeds):
            seeds.append(coord)
        if len(seeds) == n_splits:
            break

    if len(seeds) < n_splits:
        raise ValueError(
            f"Only {len(seeds)} well-separated peak(s) after min-distance "
            f"guard — try reducing Min distance"
        )

    # ── 5. Watershed on negative distance map (finds narrowest boundary) ───
    #    Runs on the cropped region only — fast even on CPU.
    from skimage.segmentation import watershed
    markers = np.zeros(mask_crop.shape, dtype=np.int32)
    for i, c in enumerate(seeds, start=1):
        markers[tuple(c)] = i
    split_crop = watershed(-dist_smooth, markers, mask=mask_crop)

    # ── 6. Clear only the cut interface (1 voxel each side) ──────────────
    #    Find face-adjacent voxel pairs belonging to different parts and zero
    #    both.  The outer surface of each part is left completely untouched.
    eroded_crop = split_crop.copy()
    interface   = np.zeros(split_crop.shape, dtype=bool)
    for axis in range(split_crop.ndim):
        slc_lo = [slice(None)] * split_crop.ndim
        slc_hi = [slice(None)] * split_crop.ndim
        slc_lo[axis] = slice(None, -1)
        slc_hi[axis] = slice(1, None)
        slc_lo = tuple(slc_lo)
        slc_hi = tuple(slc_hi)
        both = (
            (split_crop[slc_lo] > 0) &
            (split_crop[slc_hi] > 0) &
            (split_crop[slc_lo] != split_crop[slc_hi])
        )
        tmp_lo = np.zeros(split_crop.shape, dtype=bool)
        tmp_hi = np.zeros(split_crop.shape, dtype=bool)
        tmp_lo[slc_lo] = both
        tmp_hi[slc_hi] = both
        interface |= tmp_lo | tmp_hi
    eroded_crop[interface] = 0

    # ── 7. Write result back into full-volume label array ─────────────────
    split_full = np.zeros(mask.shape, dtype=np.int32)
    split_full[sl] = eroded_crop

    out     = labels.copy()
    new_ids = []
    max_lbl = int(labels.max())

    # Zero out the original blob first (gap voxels become background)
    out[mask] = 0
    out[split_full == 1] = target_label
    for i in range(2, n_splits + 1):
        new_id = max_lbl + (i - 1)
        out[split_full == i] = new_id
        new_ids.append(new_id)

    for i, nid in enumerate([target_label] + new_ids, start=1):
        n_vox = int((split_full == i).sum())
        print(f"   Part {i}: {n_vox:,} vox  (id {nid})")

    return out.astype(np.int32), new_ids


def join_labels(labels: np.ndarray, label_a: int, label_b: int) -> np.ndarray:
    """
    Merge label_b into label_a -- every voxel currently labeled label_b
    becomes label_a instead. The inverse of split_label(): two labels
    that are really one cell, wrongly segmented into two pieces (e.g. a
    thin neck that fooled the segmenter into cutting it in half),
    collapsed back into one. label_a survives; label_b's ID disappears.

    A single vectorized boolean assignment over the whole volume --
    unlike split_label(), there's no bounding-box crop to compute
    (nothing here depends on shape/geometry) and no GPU path needed,
    so this stays fast even on a full-fish volume without one.

    Returns new_labels (same shape, same dtype). Raises ValueError if
    either label is not found, or if label_a == label_b.
    """
    if label_a == label_b:
        raise ValueError("Label A and Label B must be different labels.")
    mask_a = labels == label_a
    mask_b = labels == label_b
    if not np.any(mask_a):
        raise ValueError(f"Label {label_a} not found")
    if not np.any(mask_b):
        raise ValueError(f"Label {label_b} not found")

    new_labels = labels.copy()
    new_labels[mask_b] = label_a
    return new_labels


def create_labels(
    volume: np.ndarray,
    sigma_xy: float = 1.0,
    sigma_z: float = 0.5,
    min_volume: int = 7500,
    min_hole_size: int = 0,
    final_min_fraction: float = 0.618,
) -> np.ndarray:
    """
    Create 3D labels from brain_only volume using true 3D connected components.

    Dispatches to the fastest available backend:
      CUDA (CuPy)  →  Apple MPS (threaded CPU)  →  CPU threaded

    Parameters
    ----------
    volume        : (Z, Y, X) ndarray — brain_only output
    sigma_xy      : Gaussian smoothing sigma in XY (voxels)
    sigma_z       : Gaussian smoothing sigma in Z (voxels)
    min_volume    : minimum 3D blob size in voxels
    min_hole_size : per-slice hole-fill floor, in voxels. A background
                    region fully enclosed by signal in a 2D slice
                    survives as real background only if its area is
                    >= this value; anything smaller is filled in as
                    noise instead of being left as a stray gap. Named
                    to match min_volume: both name the size a region
                    must clear to be trusted as real, not the size at
                    which it gets discarded/filled. <=0 (default) fills
                    every enclosed hole regardless of size -- the
                    original, unconditional behaviour, kept as the
                    default so existing callers are unaffected unless
                    they opt in.
    final_min_fraction : the actual 3D-blob deletion cutoff is
                    final_min_fraction * min_volume, not min_volume
                    itself -- same golden-ratio safety-net idea as
                    _cellpose_seg.py's final_min_size_cleanup(), applied
                    here too since this route has no merge/reattach
                    stage to leave a gray-zone object standing for later
                    reconsideration the way Cellpose-SAM's GMM/safe-
                    merge/large-contact chain does. Default 0.618 (the
                    golden ratio, 1/phi) matches that route's default;
                    pass 1.0 to recover the exact historical behaviour
                    (cutoff == min_volume, no relaxation).

    Returns
    -------
    (Z, Y, X) int32 ndarray — 0=background, 1..N=objects (1=largest)
    """
    backend_label = {
        "cuda": "CUDA (CuPy)",
        "mps":  f"Apple MPS → threaded CPU  (threads={_N_THREADS})",
        "cpu":  f"CPU threaded  (threads={_N_THREADS})",
    }[_BACKEND]
    print(f"   Backend: {backend_label}")

    if _BACKEND == "cuda":
        try:
            return _create_labels_cuda(
                volume, sigma_xy, sigma_z, min_volume, min_hole_size, final_min_fraction
            )
        except Exception as exc:
            # e.g. out-of-memory — fall back gracefully
            print(f"   CUDA error ({exc}), falling back to CPU.")
            _free_gpu_cache()

    return _create_labels_threaded(
        volume, sigma_xy, sigma_z, min_volume, min_hole_size, final_min_fraction
    )
