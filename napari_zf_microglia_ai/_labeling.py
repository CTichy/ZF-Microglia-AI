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
    sort_by  : "size" | "centroid_z" | "centroid_y" | "centroid_x" | "complexity"
    reverse  : reverse the natural sort order
                 size        — natural = descending (largest = label 1)
                 centroid    — natural = ascending  (smallest coord = label 1)
                 complexity  — natural = descending (most branched = label 1)

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
    elif sort_by == "complexity":
        # "Complexity" = skeleton branch count, the same morphology proxy
        # this project has used throughout (see microglia_cellpose.md) to
        # find genuinely branched/ramified cells rather than just large
        # ones. Branch *count* is scale-independent (pure topology), so a
        # dummy (1,1,1) spacing is fine here -- unlike Statistics' own
        # _skeleton_stats() use, no physical branch length is needed.
        from scipy.ndimage import find_objects
        from ._statistics import _skeleton_stats

        slices = find_objects(labels, max_label=max_lbl)

        def _branch_count(lbl):
            sl = slices[lbl - 1]
            if sl is None:
                return lbl, 0
            binary = labels[sl] == lbl
            n_branches, *_ = _skeleton_stats(binary, (1.0, 1.0, 1.0))
            return lbl, n_branches

        with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // 2)) as ex:
            results = list(ex.map(_branch_count, label_list))
        counts = dict(results)
        keyed  = [(counts[lbl], int(lbl)) for lbl in label_list]
        # natural: descending (most branched first → label 1)
        keyed.sort(key=lambda t: t[0], reverse=not reverse)
    else:
        axis_map = {"centroid_z": 0, "centroid_y": 1, "centroid_x": 2}
        axis     = axis_map[sort_by]
        # center_of_mass returns a proper list of tuples for a LIST index
        # (label_list always is one here), even a length-1 list -- the
        # "single bare tuple" case only applies to a scalar index, which
        # this never passes. No extra wrapping needed (a prior version of
        # this code wrapped unconditionally on unique.size==1, which was
        # wrong and would have crashed resorting a single-label volume by
        # centroid -- found and fixed while building a similar sweep that
        # copied the same mistaken assumption from here).
        raw      = _com(labels > 0, labels, label_list)
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
    from scipy.ndimage import find_objects

    labels = np.asarray(labels)
    max_lbl = int(labels.max()) if labels.size else 0
    if max_lbl == 0:
        return labels.copy(), 0

    out = labels.copy()
    objs = find_objects(labels)
    n_removed = 0
    for lbl in range(1, max_lbl + 1):
        sl = objs[lbl - 1] if lbl - 1 < len(objs) else None
        if sl is None:
            continue
        n_removed += _remove_debris_from_crop(out[sl], lbl, threshold)

    return out.astype(np.int32), n_removed


def _remove_debris_from_crop(crop: np.ndarray, lbl: int, threshold: int) -> int:
    """
    In-place: zero out connected fragments of value `lbl` within `crop`
    smaller than `threshold` voxels (26-connectivity, matching
    create_labels()'s own convention). Returns the number of fragments
    removed. Shared by remove_debris() (loops over every label in a
    layer) and remove_debris_for_label() (scoped to just one).
    """
    from scipy.ndimage import label as _cc_label
    structure = np.ones((3, 3, 3), dtype=np.int32)
    mask = crop == lbl
    if not mask.any():
        return 0
    cc, n_cc = _cc_label(mask, structure=structure)
    if n_cc <= 1:
        if int(mask.sum()) < threshold:
            crop[mask] = 0
            return 1
        return 0
    counts = np.bincount(cc.ravel())
    removed = 0
    for piece_id in range(1, n_cc + 1):
        if counts[piece_id] < threshold:
            crop[cc == piece_id] = 0
            removed += 1
    return removed


def remove_debris_for_label(labels: np.ndarray, label_id: int, threshold: int) -> "tuple[np.ndarray, int]":
    """
    Same connected-fragment volume filter as remove_debris(), but
    scoped to exactly ONE label -- every other label elsewhere in the
    volume is left completely untouched, unlike running the general
    Remove Debris tool on a whole layer (used as the automatic cleanup
    step after correct_label_from_intensity_3d(), which shouldn't have
    side effects on unrelated cells just because it corrected one).

    Returns (new_labels, n_removed_px) -- pixel count removed (not
    fragment count, unlike remove_debris() -- more informative for a
    single-label report).
    """
    mask = labels == label_id
    if not np.any(mask):
        return labels.copy(), 0
    nz = np.argwhere(mask)
    lo = nz.min(axis=0)
    hi = nz.max(axis=0)
    sl = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))

    out = labels.copy()
    crop = out[sl]
    before = int((crop == label_id).sum())
    _remove_debris_from_crop(crop, label_id, threshold)
    after = int((crop == label_id).sum())
    return out.astype(np.int32), before - after


def _watershed_split_mask(
    mask: np.ndarray,
    n_splits: int,
    sigma: float,
    min_distance: int,
    surface_full: "np.ndarray | None" = None,
) -> np.ndarray:
    """
    Core watershed-split engine, dimension-agnostic (works identically on a
    2D mask -- one Z-slice -- or a full 3D mask) -- extracted so
    split_label() can reuse the exact same pipeline for both its 3D and 2D
    modes instead of maintaining two copies.

    mask         : bool ndarray, any ndim (2D slice or full 3D volume)
    surface_full : optional ndarray, SAME shape as mask (uncropped) -- the
                   raw signal intensity to watershed on instead of the
                   mask's own shape. When None (default, used by 3D mode),
                   the surface is the mask's own EDT -- a purely geometric
                   split that cuts at the shape's thinnest neck, regardless
                   of what the underlying signal looks like there. When
                   given (used by 2D mode), the surface is the actual
                   image intensity -- seeds are placed at local brightness
                   peaks (real cell interiors) and the cut runs along the
                   dimmest ridge between them, i.e. wherever the real
                   signal is weakest, not wherever the mask's silhouette
                   happens to be geometrically narrowest. These can find
                   completely different cuts: a mask can be geometrically
                   wide at a point where the real signal already dips low
                   (e.g. a genuine cell edge, with a merged skin-residue
                   fragment sitting past it) -- EDT has no way to see that
                   dip at all, since it only looks at the mask's outline.

    Returns split_full : int32 ndarray, same shape as mask -- 0=background,
    1..n_splits=parts (part 1 is the seed with the highest surface peak,
    i.e. the thickest chunk in geometric mode, or the brightest chunk in
    intensity mode).

    Raises ValueError if fewer than n_splits distinct sub-regions/peaks are
    found (mirrors the original error messages).
    """
    from scipy.ndimage import distance_transform_edt

    # ── 1. Crop to bounding box (avoids running EDT on the full array) ─────
    nz  = np.argwhere(mask)
    lo  = nz.min(axis=0)
    hi  = nz.max(axis=0)
    pad = max(int(min_distance), int(sigma) + 2, 2)
    lo_p = np.maximum(lo - pad, 0)
    hi_p = np.minimum(hi + pad, np.array(mask.shape) - 1)
    sl   = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo_p, hi_p))

    mask_crop = mask[sl]

    # ── 2. The surface to split: signal intensity if given, else the
    #       mask's own EDT (geometric neck-finding, the original behavior)
    if surface_full is not None:
        dist = surface_full[sl].astype(np.float32)
    else:
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
        raise ValueError("distance transform is zero — blob too flat?")

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
    eroded_crop = _clear_split_interface(split_crop)

    # ── 7. Write result back into a full-size (mask.shape) array ───────────
    split_full = np.zeros(mask.shape, dtype=np.int32)
    split_full[sl] = eroded_crop
    return split_full


def _clear_split_interface(split_arr: np.ndarray) -> np.ndarray:
    """
    Given an int array with 0=background and 1..N=parts (from any
    watershed split), zero out only the 1-voxel-wide interface where two
    different parts directly touch -- the outer surface of each part is
    left completely untouched. Shared by _watershed_split_mask() (auto
    peak-found seeds) and any marker-seeded watershed split that wants
    the same "clean gap between parts, not a jagged shared boundary"
    treatment.
    """
    eroded = split_arr.copy()
    interface = np.zeros(split_arr.shape, dtype=bool)
    for axis in range(split_arr.ndim):
        slc_lo = [slice(None)] * split_arr.ndim
        slc_hi = [slice(None)] * split_arr.ndim
        slc_lo[axis] = slice(None, -1)
        slc_hi[axis] = slice(1, None)
        slc_lo = tuple(slc_lo)
        slc_hi = tuple(slc_hi)
        both = (
            (split_arr[slc_lo] > 0) &
            (split_arr[slc_hi] > 0) &
            (split_arr[slc_lo] != split_arr[slc_hi])
        )
        tmp_lo = np.zeros(split_arr.shape, dtype=bool)
        tmp_hi = np.zeros(split_arr.shape, dtype=bool)
        tmp_lo[slc_lo] = both
        tmp_hi[slc_hi] = both
        interface |= tmp_lo | tmp_hi
    eroded[interface] = 0
    return eroded


def split_label(
    labels: np.ndarray,
    target_label: int,
    n_splits: int = 2,
    sigma: float = 1.0,
    min_distance: int = 5,
    mode: str = "3d",
    z: "int | None" = None,
    image: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, list[int]]":
    """
    Split one label into n_splits parts using watershed.

    mode="3d" (default) splits the whole 3D blob using the mask's own
    distance transform -- the boundary is placed where the SHAPE is
    geometrically narrowest (the saddle point of the distance map
    between local maxima), independent of what the underlying signal
    looks like there.

    mode="2d" restricts the entire operation to ONE Z-slice (`z`) AND
    watersheds on the raw signal intensity (`image[z]`) instead of the
    mask's shape -- seeds are placed at local brightness peaks (real
    cell interiors) and the cut runs along the dimmest ridge between
    them, i.e. wherever the actual signal is weakest. This is for the
    case where two things only touch on a single cross-section -- e.g.
    real signal happening to graze a skin-residue fragment right at
    that slice -- where the mask's own outline can be geometrically
    wide right through a point the real signal already dips low at (a
    true 3D neck doesn't exist there, and even in 2D the mask's own
    EDT has no way to see an intensity dip it isn't shaped around).
    Only that one slice is touched -- every other Z-slice of the label
    is left completely untouched, and the new part(s) exist only on
    that slice (they don't extend into neighboring slices the way a
    3D split's parts would).

    Speed notes
    -----------
    - All operations run on the bounding box of the target label (or its
      footprint on the given slice, in 2D mode), not the full volume/slice
      — critical for large stacks.
    - Gaussian smoothing runs on GPU (CuPy) when available.

    Parameters
    ----------
    labels       : (Z, Y, X) int32 ndarray
    target_label : label value to split
    n_splits     : number of parts to produce (≥ 2)
    sigma        : Gaussian smoothing of the split surface (higher = broader peaks)
    min_distance : minimum voxel distance between seed peaks
    mode         : "3d" (default) or "2d"
    z            : slice index, required when mode="2d" (ignored for "3d")
    image        : (Z, Y, X) raw signal volume, same shape as labels --
                   required when mode="2d" (ignored for "3d", which never
                   looks at signal intensity)

    Returns
    -------
    (new_labels, new_ids)
        new_labels — same shape as labels, blob split into n_splits parts
        new_ids    — list of n_splits-1 new label IDs created
                     (target_label is kept for part 1)

    Raises
    ------
    ValueError  if mode is invalid, z/image is missing or the wrong shape
                for mode="2d", the label is not found (in the volume, or
                on slice z), or fewer peaks than n_splits are found
    """
    if mode not in ("3d", "2d"):
        raise ValueError(f"mode must be '3d' or '2d', got {mode!r}")

    if mode == "2d":
        if z is None:
            raise ValueError("z (slice index) is required when mode='2d'")
        if not (0 <= z < labels.shape[0]):
            raise ValueError(f"slice {z} out of range for a {labels.shape[0]}-slice volume")
        if image is None:
            raise ValueError("image (raw signal volume) is required when mode='2d'")
        if image.shape != labels.shape:
            raise ValueError(f"image shape {image.shape} != labels shape {labels.shape}")
        mask = labels[z] == target_label
        if not np.any(mask):
            raise ValueError(f"Label {target_label} not found on slice {z}")
        surface_full = image[z].astype(np.float32)
    else:
        mask = labels == target_label
        if not np.any(mask):
            raise ValueError(f"Label {target_label} not found")
        surface_full = None

    split_full = _watershed_split_mask(mask, n_splits, sigma, min_distance, surface_full=surface_full)

    out     = labels.copy()
    out_view = out[z] if mode == "2d" else out
    new_ids = []
    max_lbl = int(labels.max())

    # Zero out the original blob first (gap voxels become background)
    out_view[mask] = 0
    out_view[split_full == 1] = target_label
    for i in range(2, n_splits + 1):
        new_id = max_lbl + (i - 1)
        out_view[split_full == i] = new_id
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


def correct_label_from_intensity(
    labels: np.ndarray,
    image: np.ndarray,
    label_id: int,
    z: int,
    lo: float,
    hi: float,
    pad: int = 15,
) -> np.ndarray:
    """
    Regenerate one label's shape on one Z-slice from a raw-signal
    intensity window, instead of hand-painting it -- for the case
    where a particular contrast window (read straight from the signal
    image layer's own contrast_limits) happens to trace the cell's
    true silhouette better than the existing segmentation does on
    that slice.

    labels    : (Z, Y, X) label volume
    image     : (Z, Y, X) raw signal volume, same shape as labels
    label_id  : the label being corrected
    z         : slice index to correct -- only this slice is touched
    lo, hi    : the signal layer's own contrast_limits. Only lo is
                actually used as the foreground cutoff (candidate =
                image >= lo) -- hi is napari's display SATURATION
                ceiling, not an upper bound on what still counts as
                real signal. A band threshold (lo <= image <= hi) was
                tried first and got this backwards: with a narrow
                window like [100, 101] chosen specifically to make
                the display saturate into a clean silhouette, a
                cell's true bright interior sits well ABOVE hi (it's
                just displayed as solid white, i.e. saturated) --
                excluding it as "not candidate" emptied the label,
                while scattered background pixels that happened to
                fall inside the narrow band got pulled in instead.
                hi is still accepted (and shown in status messages)
                for context, but never restricts the mask.
    pad       : bounding-box padding in pixels around the label's
                existing footprint on this slice, XY only (matches
                extract_cellpose_crops.py's own convention)

    A naive threshold within the padded bounding box would happily
    pick up a neighboring cell that happens to fall inside the same
    box -- two real problems, not one: (a) a bright neighbor sitting
    fully inside the crop but never touching label_id at all, and (b)
    a neighbor whose thresholded pixels touch label_id's own
    candidate pixels, which a plain connected-component pass would
    then fuse into one blob. Both are handled the same way: any pixel
    already claimed by a DIFFERENT existing label is excluded from
    the candidate mask outright, before connected components ever
    run -- so a foreign label can never be grown into, merged in, or
    even present in the corrected shape, regardless of how the
    intensity window or connectivity falls.

    Returns new_labels (same shape, same dtype) with only that one
    slice's label_id footprint changed. Raises ValueError if label_id
    isn't present on slice z, or if the intensity window leaves
    nothing connected to the label's own existing footprint (a wrong
    contrast window would otherwise silently erase the label instead
    of correcting it).
    """
    if not (0 <= z < labels.shape[0]):
        raise ValueError(f"slice {z} out of range for a {labels.shape[0]}-slice volume")
    if labels.shape != image.shape:
        raise ValueError(f"labels shape {labels.shape} != image shape {image.shape}")

    labels_z = labels[z]
    image_z  = image[z]
    try:
        corrected, crop_existing, (y0, y1, x0, x1) = _intensity_correct_2d(
            labels_z, image_z, label_id, lo, pad
        )
    except ValueError as exc:
        raise ValueError(f"{exc} (slice {z})") from None

    new_labels = labels.copy()
    crop = new_labels[z, y0:y1, x0:x1]
    crop[crop_existing] = 0
    crop[corrected] = label_id
    return new_labels


def correct_label_from_intensity_3d(
    labels: np.ndarray,
    image: np.ndarray,
    label_id: int,
    lo: float,
    pad: int = 15,
    min_volume: "int | None" = None,
    final_min_fraction: float = 0.618,
    z_extent_pad: "int | None" = None,
    sigma: float = 1.0,
    resolve_adjacent: bool = True,
    auto_grow: bool = False,
    growth_step: int = 15,
    max_iterations: int = 5,
    until_stable: bool = False,
    max_stability_passes: int = 10,
) -> "tuple[np.ndarray, dict]":
    """
    3D version of correct_label_from_intensity(): corrects the WHOLE
    cell, not just one slice.

    NOT a bounding box. This loops over 2D areas, one Z slice at a
    time, each with its OWN local neighborhood -- the label's own
    CURRENT footprint on that one slice, padded -- exactly the same
    neighborhood the already-correct 2D tools use, reused directly:

    - No other label anywhere in that local area -> plain single-label
      2D correction (_intensity_correct_2d, the exact engine behind
      Correct Label's own 2D mode).
    - One or more other labels present in that local area -> the same
      joint marker-seeded watershed split Correct Adjacent Labels uses
      standalone (_correct_label_group_2d_core, scoped to label_id via
      focus_ids so only ITS OWN local neighborhood is ever touched --
      see correct_label_group_2d's own focus_ids docstring), so the
      LOCAL boundary against a genuinely adjacent cell gets properly
      re-derived at that one slice too, not just excluded/protected as
      an immovable wall the way plain foreign-exclusion alone would.

    The label's own ORIGINAL Z range [zmin, zmax] is corrected this way
    outright (the label already exists on every one of those slices) --
    but NOT in plain Z order. Slices with no foreign label anywhere
    nearby (the safely-isolated "core" of the cell) are corrected
    first; slices actually touching or near another label (a real
    neighbor, or a Protect-Skin-as-Label sentinel) are corrected LAST,
    in order of increasing proximity to that contact -- the core
    settles first and completely, so the genuinely contested boundary
    has a fully-formed interior to anchor against by the time it's
    resolved, instead of settling into an arbitrary, order-dependent
    shape before the rest of the cell is even known. Growth beyond the
    original range works by COPYING the current slice's own just-
    corrected footprint onto the next Z first (foreign-protected, same
    principle as Copy Label to Adjacent Slice), then running the exact
    same local-area correction there; if that correction finds no real
    signal to support the copied position at all, the copy is undone
    and growth in that direction stops -- the true edge has been found.
    A hard cap, z_extent_pad, still bounds how far this can ever extend
    regardless (see its own docstring below).

    (An earlier version instead sized ONE Y/X window from the label's
    OWN ENTIRE 3D extent and applied that SAME fixed window to every Z
    slice being corrected or walked through. That's a real design
    flaw, not just an inefficiency: a window sized from the label's
    whole extent can easily be much bigger, in some direction, than
    where the cell actually sits on any ONE particular slice -- and
    since whole-area-fill has no connectivity requirement within a
    window (see _intensity_grow_2d's own docstring), it would happily
    claim whatever OTHER real, not-yet-labeled signal -- skin residue,
    a different unlabeled structure -- happened to also fall inside
    that oversized window on that slice. Once absorbed, THAT signal's
    own further extent on the next slice could pull the walk along even
    further, propagating the error outward. A per-slice, always
    locally-anchored neighborhood -- re-derived fresh from wherever the
    label ACTUALLY currently sits, every single slice -- cannot do
    this: it can only ever reach as far as `pad` from the label's own
    real, already-verified position one slice ago.)

    After the walk, remove_debris_for_label() (golden-ratio-relaxed
    floor, same as Cellpose-SAM's own final safety net) cleans up any
    small disconnected fragment left over from the correction -- scoped
    to ONLY this label, unlike running the general Remove Debris tool
    on the whole layer.

    labels, image      : (Z, Y, X) volumes, same shape
    label_id            : the label being corrected
    lo                  : one-sided intensity cutoff (signal = image >= lo)
    pad                 : bbox padding in pixels, Y/X axes, applied fresh
                          around the label's own footprint on EACH slice
    min_volume          : if given (with final_min_fraction), the debris
                          floor is final_min_fraction * min_volume voxels
                          -- pass None to skip the debris-cleanup step
                          entirely (report will show n_debris_removed_px=0)
    final_min_fraction  : golden ratio (0.618) by default, matching
                          every other final-safety-net stage in this plugin
    sigma               : Gaussian smoothing before the watershed split,
                          only used on a slice where another label is
                          actually nearby AND resolve_adjacent is True
                          -- same meaning/default as Correct Adjacent
                          Labels' own sigma
    resolve_adjacent    : True (default) -- when another label is
                          present in a slice's own local area, jointly
                          re-derive the boundary against it (the normal
                          single-cell behavior described above). False
                          treats every other label purely as protected,
                          excluded territory instead -- never grown
                          into, never jointly split against, just
                          skipped -- exactly the plain single-label
                          engine's own behavior. Meant for correcting a
                          label that's deliberately expected to be
                          touching many other labels everywhere (e.g.
                          seed_skin_label()'s sentinel skin label, which
                          borders essentially every real cell in the
                          fish) -- jointly watershed-splitting against
                          every one of them would be both wrong (skin
                          isn't "the same kind of thing" as a real cell
                          whose boundary needs resolving) and far more
                          expensive than simply excluding them.
    auto_grow           : False (default) -- when True, any slice whose
                          own result touches the edge of ITS OWN local
                          working area retries with a bigger pad, for
                          THAT SLICE ONLY, up to max_iterations times --
                          growth_step/max_iterations same meaning as
                          the widget's own Auto-grow fields. Growing is
                          entirely LOCAL and per-slice, not a global
                          "redo the whole cell with a bigger pad"
                          retry: a cell with 40 well-behaved slices and
                          one long branch on slice 23 alone only ever
                          regrows slice 23, at whatever bigger pad IT
                          needs -- every other slice keeps its own
                          original, already-correct result and its own
                          (smaller) pad, untouched. This also makes
                          convergence far easier to reach: one slice
                          needing a bigger pad no longer means EVERY
                          slice must be redone at that same bigger pad
                          before the whole cell can be judged converged.
                          growth_step/max_iterations are ignored when
                          auto_grow is False (every slice uses `pad`
                          exactly once, unchanged prior behavior).
    until_stable        : False (default) -- a SEPARATE, independent
                          switch from auto_grow: growing decides how
                          much ROOM one attempt gets; this decides
                          whether to RE-RUN a slice's correction
                          repeatedly. A slice with a genuinely adjacent
                          label doesn't just get its own shape
                          corrected -- the joint watershed split also
                          updates the NEIGHBOR's boundary there, seeded
                          from each label's CURRENT shape. Re-running
                          that same slice again, now that the neighbor
                          (and this label) both hold their just-updated
                          shapes, re-seeds the watershed from a
                          slightly different marker than before, which
                          can nudge the split a little closer to a
                          mutually-consistent position each time --
                          repeated enough, this converges toward a
                          genuine fixed point where re-running the
                          slice produces the identical result again.
                          When True, EACH slice (independently, exactly
                          like auto_grow above) keeps re-correcting
                          itself -- with its own auto_grow retries
                          still applying on every one of those attempts
                          if auto_grow is also True -- until ITS OWN
                          footprint on that one slice matches what the
                          PREVIOUS attempt on that same slice produced,
                          or max_stability_passes is reached, whichever
                          comes first. A slice with no adjacent label at
                          all still needs a minimum of 2 attempts to
                          CONFIRM nothing is changing (there's no
                          baseline to compare against on the first
                          attempt) -- cheap, and the only way to be sure
                          without assuming. False runs exactly one
                          attempt per slice, unchanged prior behavior.
    max_stability_passes : hard per-slice cap on until_stable's re-runs
                          (default 10), so a slice can never loop
                          forever. Ignored when until_stable is False.

    Returns (new_labels, report). report is a dict:
        z_center            -- informative only (status-message
                               "corrected from centroid slice N") --
                               the label's own pre-correction 3D centroid,
                               snapped to its nearest real Z if the
                               centroid itself lands off the label
        slices_corrected    -- sorted list of every Z the walk actually
                               produced label_id on
        slices_trimmed      -- always [] -- growth only ever adds a
                               slice after confirming real continuation,
                               and the original range is always
                               corrected in place, so nothing needs a
                               separate trim step
        n_trimmed_px        -- always 0, kept for report-shape
                               compatibility with existing callers
        n_debris_removed_px -- pixels removed by the debris-cleanup step
        foreign_touching    -- {z: sorted [foreign label ids]} -- IDs
                               whose pixels directly border (8-connected)
                               this label's corrected footprint on that
                               slice, for slices where this is non-empty
        foreign_nearby      -- {z: sorted [foreign label ids]} -- IDs
                               present anywhere inside that slice's own
                               local working area, even where not
                               touching (exactly the set that decided
                               single-label vs. joint correction there)
        border_touching_slices, touched_border -- does the label's own
                               corrected footprint reach the edge of
                               ITS OWN local area on that slice, AFTER
                               auto_grow's own per-slice retries (if
                               any) are exhausted -- i.e. these are the
                               slices genuinely still cut off even at
                               their own biggest attempted pad, worth a
                               manual look (Y/X only -- never whether
                               Z-growth stopped early against its own
                               cap; see z_extent_pad below for why Z
                               deliberately stays out of this signal)
        slices_grown        -- {z: final pad used}, only for slices
                               whose own final pad ended up bigger than
                               the base `pad` -- i.e. where per-slice
                               auto_grow actually kicked in. Empty
                               unless auto_grow=True and at least one
                               slice needed the extra room.
        slices_stability_passes -- {z: n passes}, only for slices that
                               took MORE than 1 pass to stop changing
                               -- i.e. where per-slice until_stable
                               actually kept re-running. Empty unless
                               until_stable=True.
        stable              -- True unless at least one slice hit
                               max_stability_passes still changing.
                               Always True when until_stable=False.

    z_extent_pad        : hard cap, in slices, on how far Z-growth may
                          ever extend beyond the label's own original
                          [zmin, zmax] in either direction, regardless
                          of what the copy-and-verify step alone would
                          allow. Defaults to the same value as `pad`
                          when not given explicitly. Exists as its own
                          parameter (decoupled from `pad`) for the
                          auto-grow orchestrator (_grow_correct.py):
                          its Pass 1 always passes the FIXED initial_pad
                          here, even while `pad` itself grows across
                          retries, so a single attempt's Z-growth can
                          never cascade into a genuinely-touching-but-
                          not-yet-corrected neighbor's own real signal
                          and continue along however far THAT
                          neighbor's own signal happens to extend,
                          long before a joint/Pass-2-style correction
                          ever gets a chance to split the boundary
                          properly. touched_border/border_touching_slices
                          are deliberately never based on hitting this
                          cap -- a real cell taller than the cap
                          silently stops growing there rather than
                          being treated as "not converged" by the
                          auto-grow loop above; that's an accepted
                          trade-off there, not a retriable condition.

    Raises ValueError if label_id isn't found anywhere in the volume,
    or if the threshold leaves no signal at all anywhere in the
    label's own original Z range.
    """
    if labels.shape != image.shape:
        raise ValueError(f"labels shape {labels.shape} != image shape {image.shape}")

    mask3d = labels == label_id
    if not np.any(mask3d):
        raise ValueError(f"label {label_id} not found anywhere in the volume")

    Z_dim, Y_dim, X_dim = labels.shape
    zs = np.nonzero(mask3d)[0]

    # Informative only (status-message "corrected from centroid slice
    # N") -- the walk below neither starts from nor depends on this.
    from scipy.ndimage import center_of_mass as _com
    cz, _cy, _cx = _com(mask3d)
    z_center = int(round(cz))
    zs_with_label = np.unique(zs)
    if z_center not in zs_with_label.tolist():
        z_center = int(zs_with_label[np.argmin(np.abs(zs_with_label - z_center))])

    z_orig_min, z_orig_max = int(zs.min()), int(zs.max())
    if z_extent_pad is None:
        z_extent_pad = pad

    new_labels = labels.copy()

    from scipy.ndimage import binary_dilation
    struct2d = np.ones((3, 3), dtype=bool)

    slices_corrected: "list[int]" = []
    foreign_touching: "dict[int, list[int]]" = {}
    foreign_nearby: "dict[int, list[int]]" = {}
    border_touching_slices: "set[int]" = set()
    pad_used_by_slice: "dict[int, int]" = {}
    slices_stability_passes: "dict[int, int]" = {}
    slices_unstable: "set[int]" = set()

    def _local_bbox(z: int, use_pad: int) -> "tuple[int, int, int, int] | None":
        own = new_labels[z] == label_id
        if not own.any():
            return None
        ys, xs = np.nonzero(own)
        y0 = max(int(ys.min()) - use_pad, 0)
        y1 = min(int(ys.max()) + use_pad + 1, Y_dim)
        x0 = max(int(xs.min()) - use_pad, 0)
        x1 = min(int(xs.max()) + use_pad + 1, X_dim)
        return y0, y1, x0, x1

    def _correct_one_slice(z: int, use_pad: int) -> bool:
        """One attempt at correcting label_id on slice z from its OWN
        local neighborhood (sized from use_pad), mutating new_labels in
        place. Returns True if label_id ends up with a non-empty
        footprint there afterward, False if the correction found no
        real signal to support it there at all (nothing written; the
        caller decides what that means). Updates foreign_touching/
        foreign_nearby/border_touching_slices for z to reflect THIS
        attempt -- a later retry at a bigger pad simply overwrites/
        clears the previous attempt's entries for the same z, so only
        the final attempt's state survives into the report."""
        bbox = _local_bbox(z, use_pad)
        if bbox is None:
            return False
        y0, y1, x0, x1 = bbox
        foreign_ids = sorted(
            int(i) for i in np.unique(new_labels[z, y0:y1, x0:x1])
            if i not in (0, label_id)
        )
        try:
            if not foreign_ids or not resolve_adjacent:
                corrected, crop_existing, (cy0, cy1, cx0, cx1) = _intensity_correct_2d(
                    new_labels[z], image[z], label_id, lo, use_pad
                )
                crop = new_labels[z, cy0:cy1, cx0:cx1]
                crop[crop_existing] = 0
                crop[corrected] = label_id
            else:
                group_ids = [label_id] + foreign_ids
                (cy0, cy1, cx0, cx1, crop_existing, finals, _info) = _correct_label_group_2d_core(
                    new_labels, image, group_ids, z, lo, use_pad, sigma, focus_ids=[label_id],
                )
                crop = new_labels[z, cy0:cy1, cx0:cx1]
                for lid in group_ids:
                    crop[crop_existing[lid]] = 0
                for lid in group_ids:
                    crop[finals[lid]] = lid
        except ValueError:
            return False

        own_now = new_labels[z] == label_id
        if not np.any(own_now):
            return False

        dilated = binary_dilation(own_now, structure=struct2d)
        touching_here = new_labels[z][dilated & ~own_now]
        touching_ids = sorted(int(i) for i in np.unique(touching_here) if i not in (0, label_id))
        if touching_ids:
            foreign_touching[z] = touching_ids
        else:
            foreign_touching.pop(z, None)
        if foreign_ids:
            foreign_nearby[z] = foreign_ids
        else:
            foreign_nearby.pop(z, None)

        crop_own = own_now[y0:y1, x0:x1]
        touched = (
            (y0 > 0 and bool(crop_own[0, :].any()))
            or (y1 < Y_dim and bool(crop_own[-1, :].any()))
            or (x0 > 0 and bool(crop_own[:, 0].any()))
            or (x1 < X_dim and bool(crop_own[:, -1].any()))
        )
        if touched:
            border_touching_slices.add(z)
        else:
            border_touching_slices.discard(z)
        pad_used_by_slice[z] = use_pad
        return True

    def _correct_slice_with_growth(z: int) -> bool:
        """Correct slice z, retrying with a bigger LOCAL pad -- just
        for this one slice -- as long as auto_grow is on and this
        slice's own result keeps touching the edge of its own working
        area, up to max_iterations attempts. A slice that needs more
        room (a long branch reaching further in Y/X, say) grows on its
        own, without inflating every other slice's own pad too -- see
        the auto_grow parameter's own docstring for why."""
        use_pad = pad
        ok = False
        attempts = max_iterations if auto_grow else 1
        for _attempt in range(attempts):
            ok = _correct_one_slice(z, use_pad)
            if not ok or not auto_grow or z not in border_touching_slices:
                break
            use_pad += growth_step
        return ok

    def _correct_slice_full(z: int) -> bool:
        """Correct slice z, wrapping _correct_slice_with_growth() (per-
        slice auto_grow, above) with a further per-slice STABILITY
        loop: if until_stable is on, keep re-running THIS SAME SLICE --
        each attempt re-seeded from the previous attempt's own result
        on this slice, including whatever a joint correction just did
        to a genuinely adjacent label there too -- until this slice's
        own footprint matches what the immediately preceding attempt on
        this slice produced, or max_stability_passes is hit. Entirely
        local to z: a neighboring slice settling slowly never causes
        THIS slice to be redone, and vice versa -- see the until_stable
        parameter's own docstring for the full rationale."""
        if not until_stable:
            return _correct_slice_with_growth(z)
        prev_mask_here = None
        ok = False
        for sp in range(1, max_stability_passes + 1):
            ok = _correct_slice_with_growth(z)
            if not ok:
                return False
            cur_mask_here = new_labels[z] == label_id
            if prev_mask_here is not None and np.array_equal(cur_mask_here, prev_mask_here):
                if sp > 1:
                    slices_stability_passes[z] = sp
                slices_unstable.discard(z)
                return True
            prev_mask_here = cur_mask_here
        # exhausted max_stability_passes without two consecutive
        # attempts agreeing -- still changing, report it rather than
        # silently keeping whatever the last attempt happened to produce
        slices_stability_passes[z] = max_stability_passes
        slices_unstable.add(z)
        return True

    def _copy_forced(src_z: int, dst_z: int) -> bool:
        """Paste label_id's CURRENT shape from src_z onto dst_z,
        foreign-protected. Returns True if anything was actually
        painted (False if every pixel of the shape landed on foreign
        territory -- nothing to copy at all)."""
        src_mask = new_labels[src_z] == label_id
        dst_slice = new_labels[dst_z]
        foreign_here = (dst_slice != 0) & (dst_slice != label_id)
        paint_mask = src_mask & ~foreign_here
        if not paint_mask.any():
            return False
        dst_slice[dst_slice == label_id] = 0
        dst_slice[paint_mask] = label_id
        return True

    def _bbox_area(z: int) -> int:
        """Area of label_id's own bounding box on slice z (0 if not
        present there). Deliberately NOT voxel count: a sparse, spread-
        out shape (a branch reaching wide, say) can have FEWER voxels
        than a small dense blob while still needing a much bigger local
        working window -- since _local_bbox() sizes that window from
        the footprint's own min/max extent (+ pad), not from how many
        of its pixels are actually filled in, the bounding box's own
        area is what actually predicts how much room a slice needs,
        not raw pixel count."""
        ys, xs = np.nonzero(new_labels[z] == label_id)
        if ys.size == 0:
            return 0
        return int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))

    # 1a. Propagate the bigger of each consecutive pair's bounding-box
    #     area first, as a single SEQUENTIAL sweep over the whole known
    #     range, before any actual correction runs -- see _bbox_area()'s
    #     own docstring for why area (not voxel count), and the note
    #     below for why this stays sequential even though the actual
    #     correction pass (1b) no longer is. A single spuriously
    #     undersized original label on one slice (raw Cellpose-SAM
    #     prediction noise, not necessarily a real taper) can otherwise
    #     size that slice's own local working window too tightly,
    #     clipping real signal a properly-sized window would have
    #     caught. This only ever widens the WINDOW the correction looks
    #     within -- what actually gets painted still comes purely from
    #     the real signal threshold there, so a genuinely tapering slice
    #     still comes out correctly smaller; only the risk of a
    #     too-small window is removed, not the correction's own honesty.
    for z in range(z_orig_min, z_orig_max):
        if _bbox_area(z) > _bbox_area(z + 1):
            _copy_forced(z, z + 1)
        else:
            _copy_forced(z + 1, z)

    # 1b. Correct every slice in the known range -- but NOT in plain Z
    #     order. A lightweight lookahead first finds which slices are
    #     already near ANY foreign label (before anything is corrected,
    #     using the same local-bbox neighborhood _correct_one_slice()
    #     itself would look at); every OTHER slice is then corrected in
    #     order of DECREASING distance from the nearest such slice --
    #     i.e. the safely-isolated "core" of the cell settles first and
    #     completely, and only the genuinely contested boundary (right
    #     where a real neighbor or Protect-Skin-as-Label's sentinel
    #     label is actually touching) is corrected last, once it has a
    #     fully-settled interior to anchor against instead of one that's
    #     still mid-correction itself. Without this, a fixed min->max
    #     walk order means the FIRST slice ever corrected could be one
    #     sitting right against skin, settling its own joint watershed
    #     boundary before the rest of the cell's own true shape is even
    #     known yet -- a real risk of never stabilizing cleanly, or
    #     settling into an order-dependent boundary that isn't actually
    #     the best one, rather than a genuinely converged result.
    foreign_contact_zs: "set[int]" = set()
    for z in range(z_orig_min, z_orig_max + 1):
        bbox = _local_bbox(z, pad)
        if bbox is None:
            continue
        y0, y1, x0, x1 = bbox
        crop_lbls = new_labels[z, y0:y1, x0:x1]
        if np.any((crop_lbls != 0) & (crop_lbls != label_id)):
            foreign_contact_zs.add(z)

    all_zs = list(range(z_orig_min, z_orig_max + 1))
    if foreign_contact_zs:
        def _dist_to_contact(z: int) -> int:
            return min(abs(z - fz) for fz in foreign_contact_zs)
        order = sorted(all_zs, key=lambda z: (-_dist_to_contact(z), z))
    else:
        order = all_zs

    for z in order:
        if _correct_slice_full(z):
            slices_corrected.append(z)

    if not slices_corrected:
        raise ValueError(
            f"threshold >= {lo} leaves no signal at all anywhere in label "
            f"{label_id}'s own original Z range -- refusing to erase the "
            f"label; adjust the contrast window and try again."
        )

    # 2. Extend outward one slice at a time in each direction: copy the
    #    current footprint forward (foreign-protected), then run the
    #    same local-area correction there; undo and stop the moment it
    #    finds nothing to support that position.
    for direction, start_z, cap_z in (
        (1, z_orig_max, min(z_orig_max + z_extent_pad, Z_dim - 1)),
        (-1, z_orig_min, max(z_orig_min - z_extent_pad, 0)),
    ):
        z = start_z if start_z in slices_corrected else None
        if z is None:
            # the boundary slice itself lost the label entirely (rare)
            # -- fall back to the nearest slice inward that still has it
            search = range(start_z, z_orig_min - 1, -1) if direction == 1 else range(start_z, z_orig_max + 1)
            for zz in search:
                if zz in slices_corrected:
                    z = zz
                    break
        if z is None:
            continue  # nothing left in the original range to extend from

        while True:
            z_next = z + direction
            if z_next < 0 or z_next >= Z_dim:
                break
            if (direction == 1 and z_next > cap_z) or (direction == -1 and z_next < cap_z):
                break

            if not _copy_forced(z, z_next):
                break  # nothing of the shape could even be copied here -- true edge

            if not _correct_slice_full(z_next):
                new_labels[z_next][new_labels[z_next] == label_id] = 0  # undo -- no real signal here
                break

            slices_corrected.append(z_next)
            z = z_next

    n_debris_removed_px = 0
    if min_volume is not None:
        threshold = final_min_fraction * min_volume
        new_labels, n_debris_removed_px = remove_debris_for_label(new_labels, label_id, threshold)

    # Slices whose own final pad ended up bigger than the base pad --
    # i.e. where per-slice auto-grow actually kicked in. Empty unless
    # auto_grow=True and at least one slice needed the extra room.
    slices_grown = {z: p for z, p in pad_used_by_slice.items() if p != pad}

    report = {
        "z_center": z_center,
        "slices_corrected": sorted(slices_corrected),
        "slices_trimmed": [],
        "n_trimmed_px": 0,
        "n_debris_removed_px": n_debris_removed_px,
        "foreign_touching": foreign_touching,
        "foreign_nearby": foreign_nearby,
        "border_touching_slices": sorted(border_touching_slices),
        "touched_border": bool(border_touching_slices),
        "slices_grown": dict(sorted(slices_grown.items())),
        "slices_stability_passes": dict(sorted(slices_stability_passes.items())),
        "stable": not slices_unstable,
    }
    return new_labels.astype(np.int32), report


def sand_label(
    labels: np.ndarray,
    label_id: int,
    sigma_xy: float,
    sigma_z: float,
    pad: int = 10,
) -> "tuple[np.ndarray, dict]":
    """
    Softens one label's contour ("sanding"): 3D anisotropic Gaussian-blurs
    its own binary mask and re-thresholds at 0.5, rounding off small
    jagged/blocky voxel edges without meaningfully changing the cell's
    real shape or volume. Purely geometric -- no image/intensity involved,
    unlike Correct Label.

    Foreign-protected like every other Correct Label tool: a neighbor's
    already-claimed voxels can never be grown into, even where the blur
    would otherwise cross into them.

    labels             : (Z, Y, X) label volume
    label_id            : the label to soften
    sigma_xy, sigma_z    : Gaussian sigma in voxels (same units/meaning as
                          Create Labels' own Smooth sigma XY/Z)
    pad                  : bbox padding (voxels, all 3 axes) around the
                          label's own extent, giving the blur room to
                          round the boundary without being clipped by the
                          crop edge

    Returns (new_labels, info). info = {
        "applied": bool, "reason": str | None,
        "n_before": int, "n_after": int,
    } -- applied=False (labels returned unchanged) if smoothing collapses
    the label to nothing (usually because foreign neighbors already
    surround it tightly) or if the result loses all contact with the
    label's own original footprint.

    Raises ValueError if label_id isn't found anywhere in the volume.
    """
    mask3d = labels == label_id
    if not np.any(mask3d):
        raise ValueError(f"label {label_id} not found anywhere in the volume")

    zs, ys, xs = np.nonzero(mask3d)
    Z, Y, X = labels.shape
    z0, z1 = max(0, int(zs.min()) - pad), min(Z, int(zs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(Y, int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(X, int(xs.max()) + pad + 1)

    crop = labels[z0:z1, y0:y1, x0:x1]
    own = crop == label_id
    foreign = (crop != 0) & ~own
    n_before = int(own.sum())

    own_f = own.astype(np.float32)
    if _BACKEND == "cuda" and _CP is not None:
        try:
            own_gpu = _CP.asarray(own_f)
            blurred_gpu = _CPND.gaussian_filter(own_gpu, sigma=(sigma_z, sigma_xy, sigma_xy))
            blurred = blurred_gpu.get()
            del own_gpu, blurred_gpu
            _free_gpu_cache()
        except Exception as exc:
            print(f"   Sanding: GPU smooth failed ({exc}), using CPU")
            blurred = cpu_gaussian(own_f, sigma=(sigma_z, sigma_xy, sigma_xy))
    else:
        blurred = cpu_gaussian(own_f, sigma=(sigma_z, sigma_xy, sigma_xy))

    candidate = (blurred >= 0.5) & ~foreign
    n_after = int(candidate.sum())

    if n_after == 0:
        return labels, {
            "applied": False,
            "reason": "smoothing collapsed the label to nothing "
                      "(blocked by neighboring labels or too small for this sigma)",
            "n_before": n_before, "n_after": 0,
        }
    if not np.any(candidate & own):
        return labels, {
            "applied": False,
            "reason": "smoothed shape lost all contact with the label's original footprint",
            "n_before": n_before, "n_after": n_after,
        }

    new_labels = labels.copy()
    new_crop = new_labels[z0:z1, y0:y1, x0:x1]
    new_crop[own] = 0
    new_crop[candidate] = label_id
    return new_labels.astype(np.int32), {
        "applied": True, "reason": None,
        "n_before": n_before, "n_after": n_after,
    }


def _intensity_correct_2d(
    labels_z: np.ndarray,
    image_z: np.ndarray,
    label_id: int,
    lo: float,
    pad: int,
) -> "tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]":
    """
    Core 2D intensity-threshold engine behind correct_label_from_intensity(),
    extracted so a calibration sweep (many candidate `lo` values) can reuse
    it directly without copying/returning a full label volume on every
    candidate -- this returns just the small cropped boolean result.

    labels_z, image_z : one Z-slice each (2D), same shape
    label_id           : the label being corrected
    lo                  : one-sided intensity cutoff (signal = image >= lo)
    pad                 : bbox padding in pixels

    Returns (corrected, crop_existing, (y0, y1, x0, x1)):
        corrected     — bool array, shape (y1-y0, x1-x0): the label's new
                         footprint within the crop
        crop_existing — bool array, same crop shape: the label's OLD
                         footprint within the crop (what production code
                         needs to clear before painting `corrected`)
        (y0,y1,x0,x1) — the crop's bounds within labels_z/image_z, for the
                         caller to splice back with

    Raises ValueError if label_id isn't on this slice, or if the threshold
    leaves no signal at all in the padded region.
    """
    existing = labels_z == label_id
    if not np.any(existing):
        raise ValueError(f"label {label_id} not found")

    result = _intensity_grow_2d(image_z, labels_z, label_id, lo, pad, existing)
    if result is None:
        raise ValueError(
            f"threshold >= {lo} leaves no signal at all in the padded "
            f"region around label {label_id} -- refusing to erase "
            f"the label; adjust the contrast window and try again."
        )
    corrected, crop_seed, bbox = result
    return corrected, crop_seed, bbox


def _intensity_grow_2d(
    image_z: np.ndarray,
    labels_z: np.ndarray,
    label_id: int,
    lo: float,
    pad: int,
    seed_mask: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None":
    """
    Threshold engine behind _intensity_correct_2d() (single-slice
    correction, seed = that slice's own existing label_id footprint).
    (correct_label_from_intensity_3d() used to reuse this for its own
    per-slice Z-walk too -- it's since been redesigned around a single
    bounded 3D box fill instead, see that function's own docstring for
    why, and no longer calls this at all.)

    seed_mask : bool, same shape as image_z/labels_z -- non-empty
                (caller's responsibility; this function doesn't itself
                know whether an empty seed means "label not found" or
                something else, since that's a different error for each
                caller)

    Fills the WHOLE padded crop with candidate signal -- every pixel at
    or above `lo` that isn't already claimed by a different label
    becomes part of the result, regardless of whether it's connected to
    seed_mask. Deliberately not connected-component-restricted: real
    cell boundaries are often irregular/branchy, and a human eye
    reading the same contrast window would count a visibly-present
    blob as part of the cell whether or not it happens to touch the
    label's existing pixels at this exact threshold -- a stricter
    "must already connect" rule would silently discard signal that's
    plainly there. seed_mask itself is only used to size the crop
    (bbox around it, + pad); it does not gate which candidate pixels
    survive.

    Returns (corrected, crop_seed, (y0,y1,x0,x1)) within the crop only
    -- same convention _intensity_correct_2d() already returned.
    Returns None (not an exception) if the threshold leaves no signal
    at all in the padded crop -- the caller decides what that means.
    """
    ys, xs = np.nonzero(seed_mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, labels_z.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, labels_z.shape[1])

    crop_labels = labels_z[y0:y1, x0:x1]
    crop_image  = image_z[y0:y1, x0:x1]
    crop_seed   = seed_mask[y0:y1, x0:x1]

    candidate = crop_image >= lo  # one-sided: signal is "at/above lo", not a narrow band
    foreign = (crop_labels != 0) & (crop_labels != label_id)
    candidate &= ~foreign  # never claim another label's territory

    if not candidate.any():
        return None
    return candidate, crop_seed, (y0, y1, x0, x1)


def correct_adjacent_labels_2d(
    labels: np.ndarray,
    image: np.ndarray,
    label_a: int,
    label_b: int,
    z: int,
    lo: float,
    pad: int = 15,
    sigma: float = 1.0,
) -> "tuple[np.ndarray, dict]":
    """
    Corrects TWO adjacent/touching labels on ONE slice simultaneously
    from the raw signal's intensity threshold, with the boundary
    between them placed by watershed on the signal itself -- for the
    case where two things (two real cells, or a cell and a skin-residue
    fragment) end up merged together on a particular cross-section
    (e.g. right after Copy Label to Adjacent Slice pastes a shape that
    now touches a neighbor there).

    Correcting each label independently with correct_label_from_intensity()
    doesn't handle this: at a shared contrast threshold, both labels'
    regenerated regions can fuse into one connected blob exactly where
    they touch, and each single-label correction's own foreign-exclusion
    guard would just draw the boundary wherever the OTHER label's stale
    ORIGINAL pixels happened to already sit -- not the real signal
    boundary between them, which is usually different once both are
    corrected together.

    The cut is placed by a MARKER-SEEDED watershed, anchored at each
    label's own existing footprint -- not by Split Label's blind
    peak-finding (auto-detecting the two most prominent intensity peaks
    anywhere in the combined region and cutting between THOSE). Peak-
    finding has no notion of "these two regions already have their own
    separate identities and an existing boundary between them" -- on a
    combined region with more than one real intensity dip (e.g. a
    genuine internal texture variation inside one of the two cells, in
    addition to the real seam between them), it can just as easily lock
    onto the wrong valley entirely, one unrelated to where the two
    labels actually meet. Seeding directly from each label's own
    existing region instead means the watershed floods outward from
    each cell's own known territory and the two fronts meet wherever
    they meet -- the resulting boundary naturally tracks close to the
    ORIGINAL touching boundary's neighborhood while still following the
    real per-pixel signal (so it isn't just re-drawing the stale shape
    either), rather than being pulled toward some other, unrelated dip.

    The working rectangle is sized from label_a's OWN existing footprint
    alone (+ pad) -- NOT the union with label_b. label_a is the focus:
    this is the correction "for" label_a, using label_b only to resolve
    the local boundary between them right where they're actually
    adjacent. If label_b is larger, or its own bounding box extends far
    beyond label_a's own neighborhood, none of that distant territory is
    ever read or touched -- only whatever portion of label_b happens to
    fall inside label_a's own (padded) rectangle takes part, exactly the
    same way any other foreign label would if it happened to be there.
    (An earlier version sized the rectangle from the UNION of both
    labels' footprints instead, which could balloon the working area --
    and the cost of watershedding it -- far beyond label_a's own
    neighborhood whenever label_b was large or far-flung, the same class
    of scope problem already fixed for 3D auto-grow's Pass 1.)

    labels, image    : (Z, Y, X) volumes, same shape
    label_a, label_b : label_a is the label being corrected (its own
                       footprint sizes the working rectangle); label_b
                       is the adjacent label the boundary is resolved
                       against. Must be different, both present on
                       slice z.
    z                : slice index -- only this slice is touched
    lo               : one-sided intensity cutoff (signal = image >= lo)
    pad              : bbox padding in pixels around label_a's OWN
                       existing footprint on this slice
    sigma            : Gaussian smoothing of the signal before watershed
                       (higher = less sensitive to single-pixel noise
                       nudging the boundary around)

    Returns (new_labels, info). info is a dict:
        n_a, n_b -- final pixel counts for label_a/label_b within the
                    working rectangle after correction (label_b's own
                    territory outside the rectangle, if any, is
                    unaffected and not reflected here)
        n_lost   -- pixels that were label_a or label_b before, but ended
                    up neither after the joint correction/split (e.g. a
                    sliver the threshold no longer supports, or that
                    neither watershed part reached) -- reported, not
                    hidden, so a meaningful loss doesn't go unnoticed

    Raises ValueError if label_a == label_b, either label isn't present
    on slice z, label_b doesn't reach into label_a's own working
    rectangle at all (nothing to resolve a boundary against there), the
    joint threshold connects to neither label's existing footprint at
    all, or one label's own marker ends up with zero reachable candidate
    pixels within the combined region -- refuses to silently erase one
    label rather than returning a 1-label result.
    """
    if label_a == label_b:
        raise ValueError("label_a and label_b must be different labels")
    if not (0 <= z < labels.shape[0]):
        raise ValueError(f"slice {z} out of range for a {labels.shape[0]}-slice volume")
    if labels.shape != image.shape:
        raise ValueError(f"labels shape {labels.shape} != image shape {image.shape}")

    labels_z = labels[z]
    image_z = image[z]
    existing_a = labels_z == label_a
    existing_b = labels_z == label_b
    if not np.any(existing_a):
        raise ValueError(f"label {label_a} not found on slice {z}")
    if not np.any(existing_b):
        raise ValueError(f"label {label_b} not found on slice {z}")

    # Rectangle sized from label_a ALONE -- see the docstring above.
    ys, xs = np.nonzero(existing_a)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, labels_z.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, labels_z.shape[1])

    crop_labels = labels_z[y0:y1, x0:x1]
    crop_image  = image_z[y0:y1, x0:x1]
    crop_a      = existing_a[y0:y1, x0:x1]
    crop_b      = existing_b[y0:y1, x0:x1]
    crop_seed   = crop_a | crop_b

    if not np.any(crop_b):
        raise ValueError(
            f"label {label_b} does not reach into label {label_a}'s own "
            f"working rectangle (its bbox + {pad}px padding) on slice {z} "
            f"-- nothing of it there to resolve a boundary against; "
            f"increase padding or correct label {label_a} alone instead."
        )

    candidate = crop_image >= lo  # one-sided, same convention as every other Correct Label tool
    foreign = (crop_labels != 0) & (crop_labels != label_a) & (crop_labels != label_b)
    candidate &= ~foreign  # a THIRD label's territory is still off-limits

    # The whole padded crop's candidate signal becomes the combined
    # region to split below -- not connected-component-restricted to
    # what already touches label_a/label_b (see _intensity_grow_2d's
    # own docstring for why: real cell boundaries are often irregular,
    # and a visibly-present blob within the crop should count even if
    # it isn't already touching either label's existing pixels).
    combined = candidate
    if not combined.any():
        raise ValueError(
            f"threshold >= {lo} leaves no signal at all in the padded "
            f"region around labels {label_a}/{label_b} on slice {z} -- "
            f"refusing to erase both labels; adjust the contrast window "
            f"and try again."
        )

    # Marker-seeded watershed: markers are exactly each label's own
    # EXISTING footprint (not auto-detected peaks) -- floods outward from
    # each cell's own known territory on the (smoothed) real signal, so
    # the two fronts meet near the actual current boundary's neighborhood
    # rather than being pulled toward some other, unrelated intensity dip
    # elsewhere in the combined region. See the docstring above for why
    # Split Label's peak-finding approach was tried first and rejected.
    from skimage.segmentation import watershed
    if sigma > 0:
        crop_image_smooth = cpu_gaussian(crop_image.astype(np.float32), sigma=float(sigma))
    else:
        crop_image_smooth = crop_image.astype(np.float32)

    # Unmasked watershed: floods the WHOLE crop by nearest marker (not
    # gated by connectivity to `combined`), so a candidate island that
    # only touches label_a/label_b diagonally -- or with a real gap in
    # this one slice that's actually bridged through Z in the true 3D
    # structure -- still gets assigned an owner instead of being
    # silently dropped by a mask-connectivity requirement. The result
    # is intersected with `combined` right after, so non-signal pixels
    # never survive regardless of which side of the divide they fall on.
    markers = np.zeros(combined.shape, dtype=np.int32)
    markers[crop_a] = 1
    markers[crop_b] = 2
    split_crop = watershed(-crop_image_smooth, markers)
    split_crop = _clear_split_interface(split_crop)

    crop_final_a = (split_crop == 1) & combined
    crop_final_b = (split_crop == 2) & combined
    if not np.any(crop_final_a) or not np.any(crop_final_b):
        # A marker with no candidate pixels of its own reachable within
        # `combined` gets 0 output pixels from skimage's watershed,
        # silently -- e.g. label_a's own existing footprint doesn't meet
        # the new threshold at all. Refuse rather than erase one label
        # outright, matching every other Correct Label tool's guard.
        empty_lbl = label_a if not np.any(crop_final_a) else label_b
        raise ValueError(
            f"threshold >= {lo} leaves label {empty_lbl} with nothing -- "
            f"refusing to erase it; adjust the contrast window and try again."
        )

    new_labels = labels.copy()
    crop = new_labels[z, y0:y1, x0:x1]
    crop[crop_a] = 0
    crop[crop_b] = 0
    crop[crop_final_a] = label_a
    crop[crop_final_b] = label_b

    n_lost = int((crop_seed & ~(crop_final_a | crop_final_b)).sum())
    info = {
        "n_a": int(crop_final_a.sum()),
        "n_b": int(crop_final_b.sum()),
        "n_lost": n_lost,
    }
    return new_labels.astype(np.int32), info


def correct_label_group_2d(
    labels: np.ndarray,
    image: np.ndarray,
    label_ids: "list[int]",
    z: int,
    lo: float,
    pad: int = 15,
    sigma: float = 1.0,
    focus_ids: "list[int] | None" = None,
) -> "tuple[np.ndarray, dict]":
    """
    N-label generalization of correct_adjacent_labels_2d(): jointly
    corrects an arbitrary GROUP of mutually-touching labels on ONE
    slice from the raw signal, via the same marker-seeded watershed
    (markers = each label's own existing footprint, never an
    auto-detected peak) -- see correct_adjacent_labels_2d()'s own
    docstring for why that anchoring matters. Built for
    auto_contrast_correct_stack()'s slice-by-slice pass, where more
    than two cells can end up mutually touching on a given slice after
    an independent cell-by-cell 3D correction pass, not just pairs.

    label_ids : 2 or more distinct label IDs, all present on slice z.
                A single-ID call degenerates to plain single-label
                intensity correction (one marker floods everything the
                threshold connects it to) -- supported for uniformity,
                though the pipeline that drives this only ever calls it
                with real touching groups (size >= 2).
    focus_ids : optional subset of label_ids. When given, the working
                rectangle is sized from ONLY these labels' own existing
                footprints (+ pad) -- not the whole group -- mirroring
                correct_adjacent_labels_2d()'s own label_a-only scoping
                (see its docstring). Any OTHER member of label_ids only
                takes part via whatever portion of its own territory
                happens to fall inside that smaller rectangle; territory
                elsewhere is never read or touched. None (default) uses
                the UNION of the whole group instead, matching
                auto_contrast_correct_stack()'s own Pass 2 use, where
                every group member is a genuine peer already independently
                corrected in Pass 1 -- there's no single "label we're
                concerned about" to scope around there.

    Returns (new_labels, info). info is a dict:
        n_lost   -- pixels that were one of label_ids before, but ended
                    up belonging to NONE of them after the joint
                    correction/split (reported, not hidden)
        <label_id>: <final pixel count>  -- one entry per label in the
                    group

    Raises ValueError if label_ids has duplicates, fewer than 1 label
    is found on slice z, any non-focus member of label_ids doesn't
    reach into the focus rectangle at all (when focus_ids is given),
    the joint threshold connects to none of the group's existing
    footprint at all, or any one label's own marker ends up with zero
    reachable pixels within the combined region -- refuses to silently
    erase a label rather than returning a result missing one.
    """
    (y0, y1, x0, x1, crop_existing, finals, info) = _correct_label_group_2d_core(
        labels, image, label_ids, z, lo, pad, sigma, focus_ids,
    )

    new_labels = labels.copy()
    crop = new_labels[z, y0:y1, x0:x1]
    for lid in label_ids:
        crop[crop_existing[lid]] = 0
    for lid in label_ids:
        crop[finals[lid]] = lid

    return new_labels.astype(np.int32), info


def _correct_label_group_2d_core(
    labels: np.ndarray,
    image: np.ndarray,
    label_ids: "list[int]",
    z: int,
    lo: float,
    pad: int,
    sigma: float,
    focus_ids: "list[int] | None",
):
    """
    Crop-only core behind correct_label_group_2d() -- same validation,
    candidate/foreign-exclusion, and marker-seeded watershed logic, but
    returns just the crop-level result (bbox + each label's existing
    and final crop masks + the info dict) instead of writing into a
    full (Z, Y, X) COPY of `labels`. correct_label_group_2d() itself is
    now a thin wrapper around this that does that one copy, for its own
    standalone/2D use; correct_label_from_intensity_3d()'s per-slice
    walk calls this directly so a tall cell's Z-walk isn't paying for a
    full-volume copy on every single slice it touches -- on a large
    fish that cost alone could dominate the whole operation.

    Returns (y0, y1, x0, x1, crop_existing, finals, info) -- crop_existing
    and finals are {label_id: bool array, shape (y1-y0, x1-x0)}; info is
    exactly the dict correct_label_group_2d() itself returns.
    """
    if len(label_ids) < 1:
        raise ValueError("label_ids must contain at least one label")
    if len(set(label_ids)) != len(label_ids):
        raise ValueError(f"label_ids must not contain duplicates: {label_ids}")
    if not (0 <= z < labels.shape[0]):
        raise ValueError(f"slice {z} out of range for a {labels.shape[0]}-slice volume")
    if labels.shape != image.shape:
        raise ValueError(f"labels shape {labels.shape} != image shape {image.shape}")

    labels_z = labels[z]
    image_z = image[z]

    existing: "dict[int, np.ndarray]" = {}
    seed = np.zeros(labels_z.shape, dtype=bool)
    for lid in label_ids:
        m = labels_z == lid
        if not np.any(m):
            raise ValueError(f"label {lid} not found on slice {z}")
        existing[lid] = m
        seed |= m

    # Rectangle sized from focus_ids alone when given -- see
    # correct_label_group_2d()'s own focus_ids docstring -- otherwise
    # the whole group (unchanged, existing behavior).
    scope_seed = seed
    if focus_ids is not None:
        scope_seed = np.zeros(labels_z.shape, dtype=bool)
        for lid in focus_ids:
            scope_seed |= existing[lid]

    ys, xs = np.nonzero(scope_seed)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, labels_z.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, labels_z.shape[1])

    if focus_ids is not None:
        for lid in label_ids:
            if lid in focus_ids:
                continue
            if not np.any(existing[lid][y0:y1, x0:x1]):
                raise ValueError(
                    f"label {lid} does not reach into the focus label(s) "
                    f"{sorted(focus_ids)}'s own working rectangle (bbox + "
                    f"{pad}px padding) on slice {z} -- nothing of it there "
                    f"to resolve a boundary against."
                )

    crop_labels = labels_z[y0:y1, x0:x1]
    crop_image  = image_z[y0:y1, x0:x1]
    crop_seed   = seed[y0:y1, x0:x1]
    crop_existing = {lid: m[y0:y1, x0:x1] for lid, m in existing.items()}

    candidate = crop_image >= lo  # one-sided, same convention as every other Correct Label tool
    foreign = (crop_labels != 0) & (~np.isin(crop_labels, list(label_ids)))
    candidate &= ~foreign  # any label OUTSIDE this group is still off-limits

    # The whole padded crop's candidate signal becomes the combined
    # region to split below -- not connected-component-restricted to
    # what already touches the group's existing footprints (see
    # _intensity_grow_2d's own docstring for why).
    combined = candidate
    if not combined.any():
        raise ValueError(
            f"threshold >= {lo} leaves no signal at all in the padded "
            f"region around {list(label_ids)} on slice {z} -- refusing "
            f"to erase the whole group; adjust the contrast window and "
            f"try again."
        )

    # Marker-seeded watershed: markers are exactly each label's own
    # EXISTING footprint (not auto-detected peaks) -- see
    # correct_adjacent_labels_2d()'s docstring for the full rationale.
    from skimage.segmentation import watershed
    if sigma > 0:
        crop_image_smooth = cpu_gaussian(crop_image.astype(np.float32), sigma=float(sigma))
    else:
        crop_image_smooth = crop_image.astype(np.float32)

    # Unmasked watershed -- see correct_adjacent_labels_2d()'s own
    # comment on this exact same pattern: floods the whole crop by
    # nearest marker so a candidate island not connected (even
    # diagonally, even at all within this one slice) to any group
    # member's existing footprint still gets assigned an owner, then
    # gets intersected with `combined` so only real signal survives.
    markers = np.zeros(combined.shape, dtype=np.int32)
    for i, lid in enumerate(label_ids, start=1):
        markers[crop_existing[lid]] = i
    split_crop = watershed(-crop_image_smooth, markers)
    split_crop = _clear_split_interface(split_crop)

    finals = {}
    empties = []
    for i, lid in enumerate(label_ids, start=1):
        finals[lid] = (split_crop == i) & combined
        if not np.any(finals[lid]):
            empties.append(lid)
    if empties:
        raise ValueError(
            f"threshold >= {lo} leaves label(s) {empties} with nothing -- "
            f"refusing to erase; adjust the contrast window and try again."
        )

    any_final = np.zeros(combined.shape, dtype=bool)
    for lid in label_ids:
        any_final |= finals[lid]
    n_lost = int((crop_seed & ~any_final).sum())

    # Same border-touch signal as correct_label_from_intensity_3d's own
    # per-slice check -- only a PADDED (not true image) edge counts.
    # Reported BOTH per-label and as a group-wide OR: the auto-grow
    # orchestrator only wants to keep growing because of labels it was
    # actually asked to correct, not a neighbor that got folded in
    # purely to protect its own territory -- per_label_touched_border
    # lets it filter to just the labels it cares about, while
    # touched_border (any member) stays for any caller that doesn't
    # need that distinction.
    Y_dim, X_dim = labels_z.shape

    def _touches_border(mask: np.ndarray) -> bool:
        return bool(
            (y0 > 0 and mask[0, :].any())
            or (y1 < Y_dim and mask[-1, :].any())
            or (x0 > 0 and mask[:, 0].any())
            or (x1 < X_dim and mask[:, -1].any())
        )

    per_label_touched_border = {lid: _touches_border(finals[lid]) for lid in label_ids}
    touched_border = any(per_label_touched_border.values())

    # Genuine physical adjacency (dilated-by-1) to a label OUTSIDE this
    # group -- distinct from touched_border above and from the
    # "anything present in the crop" notion correct_label_from_intensity_3d's
    # own foreign_nearby uses. The auto-grow orchestrator needs exactly
    # this stricter signal to decide whether to fold a new label into
    # the group: "present somewhere in an ever-growing padded box" is
    # far too permissive once the box gets big, and would keep pulling
    # in unrelated labels indefinitely.
    from scipy.ndimage import binary_dilation as _dilate2d
    struct2d = np.ones((3, 3), dtype=bool)
    per_label_foreign_touching: "dict[int, list[int]]" = {}
    for lid in label_ids:
        dilated = _dilate2d(finals[lid], structure=struct2d)
        # Read from crop_labels (the pre-write labels crop) rather than
        # a post-write crop -- a label OUTSIDE this group's own identity
        # never changes from this call, so this is exactly equivalent
        # and avoids needing the (now not-yet-written) output crop here.
        touching_here = crop_labels[dilated & ~finals[lid]]
        per_label_foreign_touching[lid] = sorted(
            int(i) for i in np.unique(touching_here) if i != 0 and i not in label_ids
        )

    info = {
        "n_lost": n_lost,
        "touched_border": touched_border,
        "per_label_touched_border": per_label_touched_border,
        "per_label_foreign_touching": per_label_foreign_touching,
    }
    for lid in label_ids:
        info[lid] = int(finals[lid].sum())
    return y0, y1, x0, x1, crop_existing, finals, info


def _touching_pairs_on_slice(labels_z: np.ndarray) -> "set[frozenset]":
    """
    Every pair of distinct, nonzero labels that directly (8-connected)
    border each other on this one 2D slice. Bbox-cropped per label (via
    find_objects) so cost scales with how many cells are actually on
    this slice, not the slice's full size.
    """
    from scipy.ndimage import binary_dilation, find_objects

    present = np.unique(labels_z)
    present = present[present > 0]
    pairs: "set[frozenset]" = set()
    if present.size < 2:
        return pairs

    objs = find_objects(labels_z, max_label=int(present.max()))
    struct = np.ones((3, 3), dtype=bool)
    for lid in present.tolist():
        sl = objs[lid - 1] if lid - 1 < len(objs) else None
        if sl is None:
            continue
        pad_sl = tuple(
            slice(max(s.start - 1, 0), min(s.stop + 1, labels_z.shape[i]))
            for i, s in enumerate(sl)
        )
        crop = labels_z[pad_sl]
        own = crop == lid
        if not np.any(own):
            continue
        dilated = binary_dilation(own, structure=struct)
        neighbor_ids = np.unique(crop[dilated & ~own])
        for nid in neighbor_ids.tolist():
            if nid != 0 and nid != lid:
                pairs.add(frozenset((int(lid), int(nid))))
    return pairs


def _union_find_groups(elements: "list[int]", pairs: "set[frozenset]") -> "list[list[int]]":
    parent = {e: e for e in elements}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pair in pairs:
        a, b = tuple(pair)
        if a in parent and b in parent:
            union(a, b)

    groups: "dict[int, list[int]]" = {}
    for e in elements:
        r = find(e)
        groups.setdefault(r, []).append(e)
    return list(groups.values())


def touching_groups_for_stack(labels: np.ndarray) -> "dict[int, list[list[int]]]":
    """
    For every Z slice, groups the labels present there into
    mutually-touching (8-connected) clusters. Returns {z: [group, ...]}
    -- slices with no group of size >= 2 (every label on that slice has
    no neighbor there) are omitted entirely, since they need no joint
    correction. Each group is a sorted list of label IDs.

    Used by auto_contrast_correct_stack()'s second pass: an independent
    cell-by-cell correction can leave two (or more) cells directly
    touching wherever their newly-recalibrated shapes meet, and which
    cell "wins" a contested boundary pixel there is an arbitrary
    accident of processing order -- this is how that pass finds exactly
    which (slice, cell-group) combinations need to be re-derived
    jointly instead of left as that greedy artifact.
    """
    result: "dict[int, list[list[int]]]" = {}
    for z in range(labels.shape[0]):
        labels_z = labels[z]
        present = np.unique(labels_z)
        present = present[present > 0]
        if present.size < 2:
            continue
        pairs = _touching_pairs_on_slice(labels_z)
        if not pairs:
            continue
        groups = _union_find_groups([int(p) for p in present.tolist()], pairs)
        multi = [sorted(g) for g in groups if len(g) >= 2]
        if multi:
            result[z] = multi
    return result


def copy_label_to_adjacent_slice(
    labels: np.ndarray,
    label_id: int,
    z_src: int,
    direction: int,
) -> "tuple[np.ndarray, int]":
    """
    Copy label_id's 2D footprint from slice z_src onto the adjacent
    slice z_src + direction -- e.g. to patch a slice where this cell's
    cross-section is missing or broken by reusing a good neighboring
    slice's shape.

    labels    : (Z, Y, X) label volume
    label_id  : the label whose shape is being copied
    z_src     : source slice index (the currently-viewed slice)
    direction : +1 (copy to the next slice) or -1 (copy to the
                previous slice)

    label_id's own OLD footprint on the target slice is cleared first,
    then the copied shape is painted in -- re-running this doesn't
    leave a stale double outline behind. Any pixel on the target slice
    already claimed by a DIFFERENT existing label is excluded from the
    copy outright and left completely untouched (same foreign-label
    protection principle as correct_label_from_intensity() and
    rerun_single_cell()'s crop-splice guard) -- natural shape drift
    between slices means the copied footprint can land partly on a
    neighboring cell's territory on the target slice, and that
    territory must never be overwritten.

    Returns (new_labels, n_excluded_px) -- new_labels is the same
    shape/dtype with only the target slice's label_id footprint
    changed; n_excluded_px is how many of the source shape's pixels
    were dropped because they landed on a foreign label (0 for a
    clean copy), so the caller can report a partial-overlap copy
    honestly instead of silently pretending it was exact. Raises
    ValueError if direction isn't +-1, if the source or target slice
    is out of range, if label_id isn't present on z_src, or if every
    pixel would be dropped to foreign-label territory (refuses to
    paste nothing).
    """
    if direction not in (-1, 1):
        raise ValueError(f"direction must be -1 or +1, got {direction}")
    if not (0 <= z_src < labels.shape[0]):
        raise ValueError(f"slice {z_src} out of range for a {labels.shape[0]}-slice volume")
    z_dst = z_src + direction
    if not (0 <= z_dst < labels.shape[0]):
        raise ValueError(
            f"target slice {z_dst} out of range for a {labels.shape[0]}-slice volume "
            f"-- there is no {'next' if direction > 0 else 'previous'} slice"
        )

    src_mask = labels[z_src] == label_id
    if not np.any(src_mask):
        raise ValueError(f"label {label_id} not found on slice {z_src}")

    dst_slice = labels[z_dst]
    foreign = (dst_slice != 0) & (dst_slice != label_id)
    paint_mask = src_mask & ~foreign  # never claim another label's territory
    n_excluded_px = int(np.sum(src_mask & foreign))

    if not np.any(paint_mask):
        raise ValueError(
            f"every pixel of label {label_id}'s shape on slice {z_src} lands on "
            f"a different label on slice {z_dst} -- refusing to paste nothing."
        )

    new_labels = labels.copy()
    new_dst = new_labels[z_dst]
    new_dst[new_dst == label_id] = 0  # clear this label's own old footprint on the target slice
    new_dst[paint_mask] = label_id
    return new_labels, n_excluded_px


def seed_skin_label(
    labels: np.ndarray,
    brain_mask: np.ndarray,
    skin_label_id: int = -1,
) -> "tuple[np.ndarray, int]":
    """
    Bulk-fills every BACKGROUND voxel (labels == 0) OUTSIDE the brain
    mask with a new sentinel label, skin_label_id -- turning "skin"
    (everything the brain mask didn't keep) into a real, ordinary
    label that every other Correct Label / Correct Adjacent Labels /
    auto-grow / auto-correct call already protects against by
    construction (foreign-label exclusion), instead of relying on
    those tools' own intensity threshold happening to stay below
    skin's own signal everywhere -- which it doesn't always, and a
    real cell's own correction bleeding into skin residue is exactly
    the failure this exists to structurally rule out.

    Never overwrites an existing real label's own territory (only
    fills background voxels) and never touches anything INSIDE the
    brain mask, labeled or not.

    Defaults to -1, not max(labels)+1: a fixed sentinel below zero
    rather than a moving target that depends on however many real
    cells happen to exist right now. Every place elsewhere in this
    plugin that enumerates "real" labels already does so via
    `present[present > 0]` (or an equivalent positive-only filter) --
    -1 is automatically excluded from all of those without any special
    casing, so labels 1..N keep meaning exactly what they always have.
    The foreign-exclusion checks that actually matter for protection
    (`labels != 0` and `labels != label_id`) don't care about sign at
    all, so -1 is excluded from another label's candidate signal just
    as reliably as any positive ID would be.

    This is only a bulk SEED -- it fills the WHOLE outside-brain
    region unconditionally, including empty background far from any
    real tissue. Follow up with correct_label_from_intensity_3d() on
    the returned skin_label_id (same as any other label, but pass
    resolve_adjacent=False -- see that parameter's own docstring for
    why) to trim it down to just the real, signal-supported skin
    territory.

    labels, brain_mask : (Z, Y, X) arrays, same shape. brain_mask is
                          boolean-like (nonzero = brain, kept).
    skin_label_id       : -1 by default. Pass a different value only
                          if -1 is already in use for something else;
                          it must not be 0 or already present in `labels`.

    Returns (new_labels, skin_label_id).

    Raises ValueError if brain_mask covers the entire volume (nothing
    outside it to protect), if labels/brain_mask shapes don't match,
    or if skin_label_id is 0 or already an existing label.
    """
    if labels.shape != brain_mask.shape:
        raise ValueError(f"labels shape {labels.shape} != brain_mask shape {brain_mask.shape}")

    outside_brain = ~brain_mask.astype(bool)
    if not np.any(outside_brain):
        raise ValueError("brain mask covers the entire volume -- nothing outside it to protect as skin")

    if skin_label_id == 0:
        raise ValueError("skin_label_id must not be 0 (that's background)")
    if np.any(labels == skin_label_id):
        raise ValueError(
            f"skin_label_id {skin_label_id} is already an existing label in "
            f"this volume -- pick a different value"
        )

    new_labels = labels.copy()
    fill_mask = outside_brain & (new_labels == 0)
    new_labels[fill_mask] = skin_label_id
    return new_labels.astype(np.int32), skin_label_id


def remove_label(labels: np.ndarray, label_id: int) -> "tuple[np.ndarray, int]":
    """
    Clears every voxel currently equal to label_id back to background
    (0) -- a plain bulk clear, NOT connected-component-restricted (the
    label doesn't need to be one contiguous blob for this to remove it
    entirely, unlike remove_debris_for_label()'s size filter). Used to
    remove the sentinel skin label (seed_skin_label()) once it's no
    longer needed, but works on any label.

    Returns (new_labels, n_removed_px) -- (labels.copy(), 0) if
    label_id isn't present at all.
    """
    mask = labels == label_id
    n = int(mask.sum())
    if n == 0:
        return labels.copy(), 0
    new_labels = labels.copy()
    new_labels[mask] = 0
    return new_labels.astype(np.int32), n


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
