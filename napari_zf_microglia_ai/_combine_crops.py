"""
_combine_crops.py — pools one fish's own Extract-XZYZ-Patches crop
folder into a single shared training folder via symlinks, so a
multi-fish Cellpose-SAM training run can point its own Data dir at one
place regardless of how many fish have been added to it.

Every symlink is prefixed with the fish's own full, already-unique
data-folder name (e.g. "NT39-3dpf-Crispr-ctrl-D1F4_2024-09-05_15.38.01"
-- includes the NT-number, age, dish/fish, and acquisition timestamp),
never a short Dish/Fish tag like "D1F4" alone. That shorthand repeats
across different experiments and ages -- this project has two
genuinely different fish both called "D1F4" (NT36-3dpf and NT39-3dpf)
-- so it is not a safe identifier for a shared pool on its own (see
feedback_fish_naming). Reuses find_crop_pairs() from
_crop_truncation.py so the {xy,xz,yz}_NNN_NN pair-matching logic stays
in exactly one place.

Safe to call repeatedly as new fish become available: an existing
symlink in the combined folder is never overwritten, so re-running
after adding a new fish's own crop folder only ever adds that fish's
new links, regardless of how many other fish are already pooled there.
"""

from pathlib import Path

from ._crop_truncation import find_crop_pairs


def combine_crop_folder(fish_crop_dir, combined_dir, fish_stem=None,
                         progress_cb=None, cancel_event=None):
    """
    fish_crop_dir : a train_cellpose_512-style folder ({xy,xz,yz}_NNN_NN
                    naming) for ONE fish, already produced by Extract
                    X/Y/Z Patches.
    combined_dir  : the shared training folder every fish's crops get
                    pooled into -- created if it doesn't exist yet.
    fish_stem     : prefix for this fish's symlinks in combined_dir.
                    Defaults to fish_crop_dir's own parent folder name
                    -- the fish's full, already-unique data-folder name
                    -- see this module's own docstring for why a short
                    Dish/Fish tag is not used instead.

    Returns dict: {fish_stem, n_pairs_found, n_links_created,
    n_already_existed, cancelled}.
    """
    def _report(msg):
        if progress_cb:
            progress_cb(msg)

    fish_crop_dir = Path(fish_crop_dir)
    combined_dir = Path(combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)

    if fish_stem is None:
        fish_stem = fish_crop_dir.parent.name

    pairs = find_crop_pairs(fish_crop_dir)
    _report(f"{fish_stem}: {len(pairs)} crop pairs found in {fish_crop_dir}")

    n_created = 0
    n_existed = 0
    for i, (_stem, img_path, mask_path) in enumerate(pairs):
        if cancel_event is not None and cancel_event.is_set():
            _report(f"{fish_stem}: cancelled after {i}/{len(pairs)} pairs")
            return {
                "fish_stem": fish_stem, "n_pairs_found": len(pairs),
                "n_links_created": n_created, "n_already_existed": n_existed,
                "cancelled": True,
            }
        for src in (img_path, mask_path):
            link_path = combined_dir / f"{fish_stem}_{src.name}"
            if link_path.exists() or link_path.is_symlink():
                n_existed += 1
                continue
            link_path.symlink_to(src.resolve())
            n_created += 1
        if progress_cb and (i % 50 == 0):
            _report(f"{fish_stem}: {i + 1}/{len(pairs)} pairs linked...")

    _report(f"{fish_stem}: done — {n_created} symlink(s) created, {n_existed} already existed")
    return {
        "fish_stem": fish_stem, "n_pairs_found": len(pairs),
        "n_links_created": n_created, "n_already_existed": n_existed,
        "cancelled": False,
    }
