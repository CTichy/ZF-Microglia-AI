"""
_reader.py — npe2 reader contribution.

Registers napari-zf-microglia-ai as a reader for .tif/.tiff/.ims files so
they can be opened from napari's File menu with correct voxel scale applied.
"""

from pathlib import Path


def get_reader(path):
    """
    Return _read_file if the path is a supported confocal format, else None.
    Called by napari to decide whether this plugin can open the given file.
    """
    if isinstance(path, list):
        path = path[0]
    if Path(path).suffix.lower() in (".tif", ".tiff", ".ims"):
        return _read_file
    return None


_CH_COLORMAPS = ["gray", "green", "magenta", "cyan"]


def _read_file(path):
    """
    Load a TIF or IMS stack and return napari layer data with correct scale.
    All channels are returned as separate Image layers.

    Returns
    -------
    List of (data, add_kwargs, layer_type) tuples as required by napari.
    """
    if isinstance(path, list):
        path = path[0]

    from ._io import load_file

    channels = load_file(Path(path))

    meta = channels[0][2]
    scale = meta["scale"]
    print(
        f"   napari-zf-microglia-ai reader: {len(channels)} channel(s)"
        f"  scale Z={scale[0]:.4f}  Y={scale[1]:.4f}  X={scale[2]:.4f} µm"
        f"  ({meta['source']})"
    )

    layers = []
    for i, (volume, name, metadata) in enumerate(channels):
        cmap = _CH_COLORMAPS[i % len(_CH_COLORMAPS)]
        layers.append((
            volume,
            {"name": name, "colormap": cmap, "scale": metadata["scale"]},
            "image",
        ))
    return layers
