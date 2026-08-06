"""
_gt_annotation.py — ported from skin_annotation_tool/polygon_annotation_tool.py.

Polygon-based GT annotation for brain/skin masks: hand-draw polygons on
key slices, interpolate along Z (point-to-point, with propagation past a
reference slice), rasterize to brain/skin masks.

Ported as plain viewer-taking functions rather than a standalone-Viewer
tool -- same porting treatment _cellpose_seg.py gave krendl_do3d.py -- so
this runs on the plugin's existing shared viewer instead of launching a
second one. Metadata/IMS-loading helpers (extract_tif_metadata,
parse_metadata, find_best_metadata_match, load_ims_file) are deliberately
NOT ported: _io.py already implements identical logic (same regexes, same
TIF-embedded/external-txt/IMS priority order), so the plugin's existing
Open-file flow is reused instead of a second metadata code path.

Three deliberate deviations from the original standalone script, all
fixing footguns that only matter once this runs in a shared, long-lived
viewer session (the original always ran in a fresh, single-purpose
napari.Viewer(), where none of these were problems):
  - the target Image layer is passed in explicitly rather than picked by
    "first Image layer in viewer.layers" list order (a real correctness
    risk once Tabs 1-3 have added several other layers to the same
    viewer)
  - the 'brain_polygons' Shapes layer is auto-created on demand
    (ensure_brain_polygons_layer) instead of crashing with KeyError if
    the user forgot to create it manually
  - generate_masks() no longer hides every other layer in the viewer —
    that made sense in a disposable single-purpose viewer, not here,
    where it would silently hide the user's in-progress Tab 1-3 work
  - errors are raised (ValueError) rather than printed-and-returned, so
    the GUI can surface them in a status label instead of the console
"""

from pathlib import Path

import napari
import numpy as np
import tifffile
from scipy.ndimage import median_filter

KEY_SHAPES_LAYER = "brain_polygons"
N_POLY_PTS = 96
SLICE_86_PROPAGATION = True
PROPAGATION_SLICE = 90
DO_Z_SMOOTH = True
Z_SMOOTH_WIN = 3


def save_shapes_layer(layer, filepath):
    """Save a napari shapes layer to disk."""
    filepath = Path(filepath)
    data_list = [np.asarray(d) for d in layer.data]
    types_list = list(layer.shape_type)
    np.savez_compressed(
        filepath,
        num_shapes=len(data_list),
        shape_types=types_list,
        **{f"shape_{i}": data_list[i] for i in range(len(data_list))},
    )
    print(f"   Saved {len(data_list)} shapes to {filepath.name}")


def load_shapes_layer(filepath):
    """Load a napari shapes layer from disk. Returns (data_list, types_list) or (None, None)."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None, None
    data = np.load(filepath, allow_pickle=True)
    num_shapes = int(data["num_shapes"])
    shape_types = data["shape_types"].tolist()
    shapes_data = [data[f"shape_{i}"] for i in range(num_shapes)]
    print(f"   Loaded {num_shapes} shapes from {filepath.name}")
    return shapes_data, shape_types


def resample_polygon_preserve_order(pts, n_points=N_POLY_PTS):
    """
    Resample a polygon to a consistent number of points, preserving
    drawing order (point #1 stays the first point drawn).
    """
    pts = np.asarray(pts)
    if pts.shape[1] == 3:
        z = pts[0, 0]
        yx = pts[:, 1:3]
    else:
        z = 0
        yx = pts[:, 0:2]

    if not np.allclose(yx[0], yx[-1]):
        yx = np.vstack([yx, yx[0:1]])

    dists = np.sqrt(((yx[1:] - yx[:-1]) ** 2).sum(axis=1))
    cumulative = np.concatenate([[0], np.cumsum(dists)])
    total_length = cumulative[-1]

    sample_lengths = np.linspace(0, total_length, n_points, endpoint=False)
    y_new = np.interp(sample_lengths, cumulative, yx[:, 0])
    x_new = np.interp(sample_lengths, cumulative, yx[:, 1])

    return np.column_stack([np.full(n_points, z), y_new, x_new])


def is_polygon_clockwise(yx):
    """Shoelace-formula orientation test. True if clockwise."""
    x = yx[:, 1]
    y = yx[:, 0]
    signed_area = 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return signed_area < 0


def standardize_polygon_direction(poly, reference_is_cw):
    """Reverse point order if poly's direction doesn't match reference_is_cw."""
    yx = poly[:, 1:3]
    if is_polygon_clockwise(yx) != reference_is_cw:
        return poly[::-1].copy()
    return poly


def interpolate_point_arrays(poly1, poly2, alpha):
    """Point-to-point linear interpolation between two aligned polygons."""
    assert poly1.shape == poly2.shape, "Polygons must have the same number of points"
    return (1 - alpha) * poly1 + alpha * poly2


def ensure_brain_polygons_layer(viewer, scale=(1.0, 1.0, 1.0)):
    """Return the existing 'brain_polygons' Shapes layer, creating it if missing."""
    if KEY_SHAPES_LAYER in viewer.layers:
        return viewer.layers[KEY_SHAPES_LAYER]
    return viewer.add_shapes(
        name=KEY_SHAPES_LAYER,
        edge_color="yellow",
        face_color=[1, 1, 0, 0.15],
        edge_width=2,
        ndim=3,
        scale=scale,
    )


def interpolate_shapes(viewer, image_layer):
    """
    Interpolate hand-drawn key-slice polygons along Z (point-to-point,
    with optional propagation of the PROPAGATION_SLICE polygon to all
    slices beyond it). Returns the created 'brain_polygons_interpolated'
    Shapes layer. Raises ValueError on any of the original tool's error
    conditions (missing layer, too few polygons).
    """
    print("\n" + "=" * 80)
    print("INTERPOLATING POLYGONS")
    print("=" * 80)

    if KEY_SHAPES_LAYER not in viewer.layers:
        raise ValueError(f"No '{KEY_SHAPES_LAYER}' layer — draw polygons first.")
    shapes_layer = viewer.layers[KEY_SHAPES_LAYER]

    if len(shapes_layer.data) < 2:
        raise ValueError("Need at least 2 polygons on different slices.")

    by_z = {}
    for verts, _stype in zip(shapes_layer.data, shapes_layer.shape_type):
        arr = np.asarray(verts)
        z_here = int(round(np.mean(arr[:, 0]))) if arr.shape[1] == 3 else 0
        if len(arr) >= 3:
            print(f"   Processing polygon on slice {z_here}: {len(arr)} points -> {N_POLY_PTS} points")
            by_z[z_here] = resample_polygon_preserve_order(arr, n_points=N_POLY_PTS)

    print(f"Found {len(by_z)} key polygons on slices: {sorted(by_z.keys())}")

    sorted_zs = sorted(by_z.keys())
    reference_z = sorted_zs[0]
    reference_is_cw = is_polygon_clockwise(by_z[reference_z][:, 1:3])
    corrections = 0
    for z in sorted_zs[1:]:
        standardized = standardize_polygon_direction(by_z[z], reference_is_cw)
        if not np.array_equal(by_z[z], standardized):
            by_z[z] = standardized
            corrections += 1
    print(f"Standardized polygon directions ({corrections} corrected)")

    if SLICE_86_PROPAGATION and PROPAGATION_SLICE not in by_z:
        print(f"WARNING: slice {PROPAGATION_SLICE} polygon not found — propagation disabled.")
        use_propagation = False
    else:
        use_propagation = SLICE_86_PROPAGATION

    zs = np.array(sorted_zs, float)
    zs_for_interp = zs[zs <= PROPAGATION_SLICE] if use_propagation else zs

    Z = image_layer.data.shape[0]
    datas, types = [], []
    keys_set = set(int(z) for z in zs.tolist())

    for z in range(Z):
        if z in keys_set:
            datas.append(by_z[z].copy())
        elif use_propagation and z > PROPAGATION_SLICE:
            ref_poly = by_z[PROPAGATION_SLICE].copy()
            ref_poly[:, 0] = z
            datas.append(ref_poly)
        else:
            zs_below = zs_for_interp[zs_for_interp <= z]
            zs_above = zs_for_interp[zs_for_interp >= z]
            if len(zs_below) == 0:
                poly = by_z[int(zs_for_interp[0])].copy()
                poly[:, 0] = z
                datas.append(poly)
            elif len(zs_above) == 0:
                poly = by_z[int(zs_for_interp[-1])].copy()
                poly[:, 0] = z
                datas.append(poly)
            else:
                z_low, z_high = int(zs_below[-1]), int(zs_above[0])
                if z_low == z_high:
                    poly = by_z[z_low].copy()
                    poly[:, 0] = z
                    datas.append(poly)
                else:
                    alpha = (z - z_low) / (z_high - z_low)
                    poly = interpolate_point_arrays(by_z[z_low], by_z[z_high], alpha)
                    poly[:, 0] = z
                    datas.append(poly)
        types.append("polygon")

    print(f"Generated {Z} interpolated polygons")

    if "brain_polygons_interpolated" in viewer.layers:
        viewer.layers.remove("brain_polygons_interpolated")

    layer_scale = image_layer.scale if hasattr(image_layer, "scale") else (1.0, 1.0, 1.0)
    result = viewer.add_shapes(
        datas, shape_type=types, name="brain_polygons_interpolated",
        edge_color="cyan", face_color=[0, 1, 1, 0.1], edge_width=1.5,
        ndim=3, scale=layer_scale,
    )
    print("INTERPOLATION COMPLETE")
    return result


def generate_masks(viewer, image_layer, input_path):
    """
    Rasterize the interpolated polygons into brain/skin masks, save
    brain_mask/skin_mask/original/brain_only/skin_only TIFFs plus the two
    polygon .npz files, all under <input_path.parent>/<input_path.stem>/
    (same convention as the plugin's own _output_dir()). Adds brain_mask,
    skin_mask, and brain_only as new viewer layers. Returns the output
    directory Path. Raises ValueError on error conditions.
    """
    print("\n" + "=" * 80)
    print("GENERATING MASKS")
    print("=" * 80)

    if "brain_polygons_interpolated" not in viewer.layers:
        raise ValueError("No interpolated polygons — run Interpolate Polygons first.")

    brain_shapes_interp = viewer.layers["brain_polygons_interpolated"]
    print(f"Using {len(brain_shapes_interp.data)} interpolated polygons")
    print(f"Image shape: {image_layer.data.shape}")

    try:
        lab_obj = brain_shapes_interp.to_labels(image_layer.data.shape)
        lab_arr = lab_obj.data if hasattr(lab_obj, "data") else lab_obj
        brain_mask = (np.asarray(lab_arr) > 0).astype(np.uint8)
    except Exception as exc:
        raise ValueError(f"Rasterization failed: {exc}")

    if brain_mask.sum() == 0:
        raise ValueError("Mask is empty after rasterization.")

    if DO_Z_SMOOTH and Z_SMOOTH_WIN >= 3 and Z_SMOOTH_WIN % 2 == 1:
        brain_mask = median_filter(brain_mask, size=(Z_SMOOTH_WIN, 1, 1))

    skin_mask = (1 - brain_mask).astype(np.uint8)
    print(f"Brain voxels: {brain_mask.sum():,} ({100 * brain_mask.sum() / brain_mask.size:.2f}%)")
    print(f"Skin voxels:  {skin_mask.sum():,} ({100 * skin_mask.sum() / skin_mask.size:.2f}%)")

    if "brain_mask" in viewer.layers:
        viewer.layers.remove("brain_mask")
    if "skin_mask" in viewer.layers:
        viewer.layers.remove("skin_mask")
    viewer.add_labels(brain_mask, name="brain_mask")
    viewer.add_labels(skin_mask, name="skin_mask")

    input_path = Path(input_path)
    output_dir = input_path.parent / input_path.stem
    output_dir.mkdir(exist_ok=True, parents=True)

    if KEY_SHAPES_LAYER in viewer.layers:
        save_shapes_layer(viewer.layers[KEY_SHAPES_LAYER], output_dir / f"{input_path.stem}_brain_polygons.npz")
    save_shapes_layer(brain_shapes_interp, output_dir / f"{input_path.stem}_brain_polygons_interpolated.npz")

    brain_path = output_dir / f"{input_path.stem}_brain_mask.tif"
    tifffile.imwrite(brain_path, (brain_mask * 255).astype(np.uint8), compression="zlib")
    print(f"   {brain_path.name}")

    skin_path = output_dir / f"{input_path.stem}_skin_mask.tif"
    tifffile.imwrite(skin_path, (skin_mask * 255).astype(np.uint8), compression="zlib")
    print(f"   {skin_path.name}")

    original_path = output_dir / f"{input_path.stem}_original.tif"
    tifffile.imwrite(original_path, image_layer.data)
    print(f"   {original_path.name}")

    brain_only = image_layer.data.astype("float32") * brain_mask.astype("float32")
    brain_only = brain_only.astype(image_layer.data.dtype)
    brain_only_path = output_dir / f"{input_path.stem}_brain_only.tif"
    tifffile.imwrite(brain_only_path, brain_only)
    print(f"   {brain_only_path.name}")

    skin_only = image_layer.data.astype("float32") * skin_mask.astype("float32")
    skin_only = skin_only.astype(image_layer.data.dtype)
    skin_only_path = output_dir / f"{input_path.stem}_skin_only.tif"
    tifffile.imwrite(skin_only_path, skin_only)
    print(f"   {skin_only_path.name}")

    layer_scale = image_layer.scale if hasattr(image_layer, "scale") else (1.0, 1.0, 1.0)
    if f"{input_path.stem}_brain_only" in viewer.layers:
        viewer.layers.remove(f"{input_path.stem}_brain_only")
    viewer.add_image(brain_only, name=f"{input_path.stem}_brain_only", scale=layer_scale, colormap="gray")

    print("GENERATION COMPLETE")
    print(f"Outputs saved in: {output_dir}")
    return output_dir
