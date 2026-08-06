"""
_io.py — File loading with metadata-aware anisotropy extraction.

Metadata priority:
  1. External *_metadata.txt (Leica format: camera pixel + magnification)
  2. TIF ImageJ embedded metadata (spacing, XResolution/YResolution tags)
  3. IMS embedded resolution (f.resolution)
  4. Default (1.0, 1.0, 1.0)
"""

import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import tifffile

try:
    from imaris_ims_file_reader import ims as ImsReader
    HAS_IMS = True
except ImportError:
    HAS_IMS = False

_DEFAULT_META = {
    "scale":      (1.0, 1.0, 1.0),
    "voxel_z_um": 1.0,
    "voxel_y_um": 1.0,
    "voxel_x_um": 1.0,
    "anisotropy": 1.0,
    "source":     "Default (no metadata)",
}


def _calc_anisotropy(voxel_z_um, voxel_x_um, voxel_y_um):
    xy = (voxel_x_um + voxel_y_um) / 2.0
    return voxel_z_um / xy if xy > 0 else 1.0


def extract_tif_metadata(tif_path):
    """
    Extract voxel metadata from TIF ImageJ embedded tags.
    Returns a metadata dict, or None on failure.
    """
    try:
        with tifffile.TiffFile(str(tif_path)) as tif:
            if not tif.imagej_metadata:
                return None
            md = tif.imagej_metadata
            print("   Found ImageJ metadata in TIF.")

            z_spacing = float(md.get("spacing", md.get("finterval", 1.0)))

            x_um = y_um = 1.0
            page = tif.pages[0]
            if hasattr(page, "tags"):
                xr = page.tags.get("XResolution")
                yr = page.tags.get("YResolution")
                if xr and yr:
                    xv = xr.value
                    yv = yr.value
                    if isinstance(xv, tuple):
                        x_um = xv[1] / xv[0] if xv[0] else 1.0
                    else:
                        x_um = 1.0 / xv if xv else 1.0
                    if isinstance(yv, tuple):
                        y_um = yv[1] / yv[0] if yv[0] else 1.0
                    else:
                        y_um = 1.0 / yv if yv else 1.0

            result = {
                "voxel_x_um": float(x_um),
                "voxel_y_um": float(y_um),
                "voxel_z_um": z_spacing,
                "scale":      (z_spacing, float(y_um), float(x_um)),
                "source":     "TIF ImageJ metadata",
            }
            result["anisotropy"] = _calc_anisotropy(
                result["voxel_z_um"], result["voxel_x_um"], result["voxel_y_um"]
            )
            print(
                f"   Voxel: Z={z_spacing:.4f}  Y={y_um:.4f}  X={x_um:.4f} µm"
                f"  (anisotropy {result['anisotropy']:.2f}:1)"
            )
            return result

    except Exception as exc:
        print(f"   Could not extract TIF metadata: {exc}")
        return None


def parse_metadata(metadata_path):
    """
    Parse a Leica *_metadata.txt file for voxel dimensions.
    Returns a metadata dict, or None if the file cannot be parsed.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return None

    print(f"   Reading metadata: {metadata_path.name}")
    content = metadata_path.read_text(encoding="utf-8")

    pixel_w = re.search(
        r'\{DisplayName=Pixel Width \(\xb5m\), Value=(\d+\.?\d*)\}', content
    )
    pixel_h = re.search(
        r'\{DisplayName=Pixel Height \(\xb5m\), Value=(\d+\.?\d*)\}', content
    )
    step = re.search(r'StepSize=(\d+\.?\d*)', content)
    mags = re.findall(
        r'\{DisplayName=TotalConsolidatedOpticalMagnification, Value=(\d+\.?\d*)\}',
        content,
    )

    if not (pixel_w and pixel_h and step and mags):
        print("   Could not extract voxel dims from metadata file.")
        return None

    cam_x    = float(pixel_w.group(1))
    cam_y    = float(pixel_h.group(1))
    voxel_z  = float(step.group(1))
    mag_vals = [float(m) for m in mags]

    obj_mags  = [m for m in mag_vals if m >= 10]
    zoom_mags = [m for m in mag_vals if m < 10]

    if not (obj_mags and zoom_mags):
        print("   Could not find both objective and zoom magnification.")
        return None

    total_mag = obj_mags[0] * zoom_mags[0]
    vx = cam_x / total_mag
    vy = cam_y / total_mag

    result = {
        "voxel_x_um": vx,
        "voxel_y_um": vy,
        "voxel_z_um": voxel_z,
        "scale":      (voxel_z, vy, vx),
        "source":     "External metadata file",
    }
    result["anisotropy"] = _calc_anisotropy(voxel_z, vx, vy)
    print(
        f"   Voxel: Z={voxel_z:.4f}  Y={vy:.4f}  X={vx:.4f} µm"
        f"  (anisotropy {result['anisotropy']:.2f}:1)"
    )
    return result


def find_best_metadata_match(file_path):
    """
    Search the parent directory for the best-matching *_metadata.txt file.
    Returns the Path to the best match, or None.
    """
    file_path = Path(file_path)
    parent = file_path.parent

    exact = parent / f"{file_path.stem}_metadata.txt"
    if exact.exists():
        print(f"   Found exact metadata match: {exact.name}")
        return exact

    candidates = list(parent.glob("*_metadata.txt"))
    if not candidates:
        return None

    stem = file_path.stem.lower()
    best, best_r = None, 0.0
    for c in candidates:
        r = SequenceMatcher(
            None, stem, c.stem.replace("_metadata", "").lower()
        ).ratio()
        if r > best_r:
            best_r, best = r, c

    if best and best_r > 0.3:
        print(f"   Using closest metadata match ({best_r:.0%}): {best.name}")
        return best
    return None


def load_file(path):
    """
    Load a TIF or IMS confocal stack with metadata-aware voxel scaling.

    All channels are returned as separate entries so the caller can open each
    as its own napari layer — letting the user pick which channel to process.

    Returns
    -------
    list of (volume, name, metadata) tuples — one per channel.
      volume   : np.ndarray  3-D (Z, Y, X)
      name     : str         e.g. "stem" (single channel) or "stem_ch0"
      metadata : dict        keys: scale, voxel_z_um, voxel_y_um, voxel_x_um,
                                   anisotropy, source
    """
    path   = Path(path)
    suffix = path.suffix.lower()
    stem   = path.stem

    if suffix in (".tif", ".tiff"):
        raw = tifffile.imread(str(path))
        print(f"   Loaded TIF  shape={raw.shape}  dtype={raw.dtype}")

        # Metadata priority: external txt > TIF embedded > default
        meta = None
        ext_path = find_best_metadata_match(path)
        if ext_path:
            meta = parse_metadata(ext_path)
        if meta is None:
            meta = extract_tif_metadata(path)
        if meta is None:
            print("   No metadata found — using default scale (1, 1, 1).")
            meta = dict(_DEFAULT_META)

        if raw.ndim == 3:
            channels = [(raw, stem, meta)]
        elif raw.ndim == 4:
            n = raw.shape[0]
            print(f"   Multi-channel TIF: {n} channels (C, Z, Y, X)")
            channels = [(raw[c], f"{stem}_ch{c}", meta) for c in range(n)]
        else:
            raise ValueError(f"Expected 3-D or 4-D TIF, got shape {raw.shape}")

    elif suffix == ".ims":
        if not HAS_IMS:
            raise ImportError(
                "imaris_ims_file_reader not installed. "
                "Run: pip install imaris_ims_file_reader"
            )
        f = ImsReader(str(path))
        n_ch = f.Channels
        print(f"   IMS  channels={n_ch}  shape={f.shape}")

        # Metadata priority: external txt > IMS embedded
        meta = None
        ext_path = find_best_metadata_match(path)
        if ext_path:
            meta = parse_metadata(ext_path)
        if meta is None:
            res = f.resolution
            vz, vy, vx = float(res[0]), float(res[1]), float(res[2])
            meta = {
                "voxel_z_um": vz,
                "voxel_y_um": vy,
                "voxel_x_um": vx,
                "scale":      (vz, vy, vx),
                "anisotropy": _calc_anisotropy(vz, vx, vy),
                "source":     "IMS file",
            }
            print(f"   Using IMS embedded resolution: {res}")

        channels = []
        for c in range(n_ch):
            vol = f.get_Volume_At_Specific_Resolution(f.resolution, 0, c)
            name = stem if n_ch == 1 else f"{stem}_ch{c}"
            print(f"   ch{c}: shape={vol.shape}  dtype={vol.dtype}")
            channels.append((vol, name, meta))

    else:
        raise ValueError(
            f"Unsupported file type: {suffix!r}  (expected .tif, .tiff, or .ims)"
        )

    return channels
