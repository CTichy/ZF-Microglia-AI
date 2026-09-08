"""
_grow_correct.py -- auto-growing wrapper around Correct Label / Correct
Adjacent Labels: retries the same threshold-based correction with a
progressively larger padded working region whenever the result's own
footprint touches the edge of that region -- catching real signal that
too small a pad would otherwise cut off. Also auto-detects when growth
reveals a neighboring label and expands into a joint multi-label
correction so a neighbor's territory is never wrongly consumed -- this
never grows silently into someone else's cell.

2D mode reuses correct_label_group_2d() directly (a single-ID call
degenerates to plain single-label correction, so one function already
covers both the 1-label and N-label case uniformly).

3D mode reuses the same two-pass architecture auto_contrast_correct_stack()
already uses for a whole fish (Pass 1: correct_label_from_intensity_3d()
independently per label; Pass 2: touching_groups_for_stack() +
correct_label_group_2d() wherever the independently-corrected labels now
touch) -- just scoped to the growing local group instead of every label
in the fish, and re-run from the ORIGINAL labels each attempt rather than
compounding one attempt on top of the last. One deliberate difference
from auto_contrast_correct_stack(): Pass 1's own expensive whole-cell
walk only ever runs for the ORIGINALLY-REQUESTED label(s), never for a
neighbor folded in purely to protect its own territory -- that neighbor
was never asked to be corrected, and independently re-deriving its
entire 3D extent (which can be large/complex/sprawling) just so Pass 2
has a marker for the slice or two where it actually touches the target
is wasted work Pass 2 doesn't need: a folded-in neighbor's own
PRE-correction footprint is already a perfectly valid watershed marker
(exactly how Correct Adjacent Labels works standalone, with no Pass-1
pre-correction on either label at all).

Neighbor discovery (both modes) is deliberately narrow on two axes, to
stop it ever cascading into an unrelated part of the fish:

1. Only GENUINE TOUCHING adjacency counts, never mere presence nearby
   -- a "how much padding is there room for" search would otherwise
   keep finding *something* within an ever-growing box indefinitely.
2. Only the ORIGINALLY-REQUESTED label(s)' own touches are ever
   examined -- once a neighbor is folded into the group purely to
   protect its own territory near the target, that neighbor's own
   touches elsewhere are never looked at. A large/sprawling label
   folded in this way can easily touch several other, completely
   unrelated cells somewhere else in the fish; without this
   restriction, discovering it would cascade the group into all of
   those too, and then whatever THEY touch, and so on.

In 3D, Pass 2 only ever runs once a given attempt found zero new
neighbors this way; if it does find one, the whole attempt is redone
from scratch with the bigger group instead of applying Pass 2 to a
group that might still be missing a member.
"""

from __future__ import annotations

import numpy as np

from ._labeling import (
    correct_label_from_intensity_3d,
    correct_label_group_2d,
    touching_groups_for_stack,
    remove_debris_for_label,
)


def grow_correct_label_2d(
    labels: np.ndarray,
    image: np.ndarray,
    label_ids: "int | list[int]",
    z: int,
    lo: float,
    initial_pad: int = 15,
    growth_step: int = 15,
    max_iterations: int = 5,
    sigma: float = 1.0,
    progress_cb=None,
) -> "tuple[np.ndarray, dict]":
    """
    Auto-grows Correct Label's 2D (single-slice) correction until the
    result no longer touches the edge of its own padded working
    region, expanding the padding by growth_step each time it does.
    If the growing region starts overlapping a different label, that
    label is folded into the correction (via correct_label_group_2d(),
    the joint marker-seeded watershed correction) instead of being
    encroached on by the target label's own growth.

    labels, image  : (Z, Y, X) volumes, same shape
    label_ids       : the label(s) to correct -- a single int (Correct
                      Label's own use) or a list of 2+ ints, FIRST
                      element = "label A" (Correct Adjacent Labels' own
                      use -- [label_a, label_b], seeding the group with
                      both from the start instead of just one). The
                      working rectangle is always scoped to just this
                      first id's own extent (see correct_label_group_2d's
                      own focus_ids parameter) -- label B and any
                      further folded-in neighbor still fully take part
                      in discovery/convergence/the joint watershed, only
                      the rectangle itself stays anchored on A.
    z               : slice index -- only this slice is touched
    lo              : one-sided intensity cutoff (signal = image >= lo)
    initial_pad, growth_step, max_iterations : padding starts at
                      initial_pad, grows by growth_step each attempt
                      that still touches the border, up to
                      max_iterations attempts total
    sigma           : passed through to correct_label_group_2d()'s
                      watershed smoothing (irrelevant for a 1-label group)
    progress_cb      : optional callable(str)

    Returns (new_labels, report). report:
        group            -- sorted list of label ids corrected together
        pad_used          -- the padding of the last attempt actually made
        n_iterations       -- how many attempts were made
        converged          -- True if the last attempt didn't touch the border
        group_grew         -- True if growth ever pulled in a neighbor
        info               -- the underlying correct_label_group_2d()'s own
                              info dict from the last attempt

    Raises ValueError only if even the first attempt (initial_pad,
    single label) fails outright -- same errors correct_label_group_2d()
    itself raises (label not found, threshold connects to nothing).
    """
    def _report(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    if isinstance(label_ids, (list, tuple)):
        ids_list = [int(i) for i in label_ids]
    elif isinstance(label_ids, set):
        ids_list = sorted(int(i) for i in label_ids)  # a set has no defined order -- deterministic fallback
    else:
        ids_list = [int(label_ids)]
    group = set(ids_list)
    original_group = frozenset(group)  # convergence/discovery judged on these, never a folded-in neighbor
    # The working RECTANGLE, though, is scoped to just the FIRST id as
    # given -- "label A" in Correct Adjacent Labels' own terms (its
    # widget call always passes [label_a, label_b] in that order; for
    # plain Correct Label, label_ids is a single int, so this is
    # trivially that same label). Label B (and any further folded-in
    # neighbor) still fully participates in discovery/convergence/the
    # joint watershed -- only the RECTANGLE stays anchored on A, exactly
    # as correct_adjacent_labels_2d() itself now does standalone.
    rect_focus = frozenset({ids_list[0]})
    pad = int(initial_pad)
    group_grew = False
    last_new_labels = labels
    last_info = None
    converged = False
    used_pad = pad
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        used_pad = pad
        _report(f"Attempt {iteration}: pad={used_pad}px, group={sorted(group)}")
        new_labels, info = correct_label_group_2d(
            labels, image, sorted(group), z, lo, pad=used_pad, sigma=sigma,
            # Rectangle scoped to label A alone (rect_focus), never label
            # B or a folded-in neighbor -- same principle as 3D Pass 1
            # (see the module docstring above): a large/far-flung label
            # must never balloon the working area (or the watershed's own
            # cost) beyond what's actually needed near A's own neighborhood.
            focus_ids=sorted(rect_focus),
        )
        last_new_labels, last_info = new_labels, info

        # Discovery is driven ONLY by the originally-requested label(s)'
        # own GENUINE TOUCHING adjacency (per_label_foreign_touching),
        # never by "any label merely present somewhere in the padded
        # box" (too permissive once the box grows large -- would keep
        # finding *something* nearby indefinitely) and never by an
        # already-folded-in neighbor's own touches (which could cascade
        # the group into everything THAT label happens to touch,
        # regardless of relevance to what was actually asked to be
        # corrected).
        new_neighbors: "set[int]" = set()
        for lid in original_group:
            new_neighbors.update(info["per_label_foreign_touching"].get(lid, []))
        new_neighbors -= group
        if new_neighbors:
            group |= new_neighbors
            group_grew = True
            _report(f"Growing group to include neighbor(s) {sorted(new_neighbors)} -> {sorted(group)}, redoing this attempt")
            continue  # redo with the bigger group, same pad

        # Convergence is judged ONLY on label A (rect_focus) -- NOT on
        # label B or any folded-in neighbor, even though both are members
        # of original_group. The working rectangle is sized from A's own
        # extent alone (see correct_label_group_2d's focus_ids), so B
        # -- being the adjacent, usually larger/further-reaching label --
        # will almost always end up touching the edge of A's own small
        # rectangle; that's expected and not a sign the correction needs
        # a bigger box, since B was never meant to be grown to its own
        # true extent here in the first place. Judging convergence on B
        # too would mean auto-grow essentially never converges whenever
        # B is bigger than A, which defeats the point of scoping the
        # rectangle to A at all.
        relevant_touched = any(
            info["per_label_touched_border"][lid] for lid in rect_focus
        )
        if not relevant_touched:
            converged = True
            break
        pad += growth_step

    report = {
        "group": sorted(group),
        "pad_used": used_pad,
        "n_iterations": iteration,
        "converged": converged,
        "group_grew": group_grew,
        "info": last_info,
    }
    return last_new_labels, report


def grow_correct_label_3d(
    labels: np.ndarray,
    image: np.ndarray,
    label_ids: "int | list[int]",
    lo: float,
    initial_pad: int = 15,
    growth_step: int = 15,
    max_iterations: int = 5,
    sigma: float = 1.0,
    min_volume: "int | None" = None,
    final_min_fraction: float = 0.618,
    progress_cb=None,
) -> "tuple[np.ndarray, dict]":
    """
    3D analogue of grow_correct_label_2d(). See the module docstring
    for the two-pass architecture (independent-per-label 3D walk, then
    joint re-derivation wherever the group ends up touching).

    Neighbor discovery here is reactive: nothing is known about a
    label's true 3D extent ahead of actually running Pass 1 on it, so
    each attempt runs Pass 1 first and looks at what came out of it. A
    neighbor is folded in only when it is GENUINELY TOUCHING one of the
    originally-requested label(s)' own corrected shape -- from Pass 1's
    own foreign_touching report (never foreign_nearby, which flags
    anything merely present somewhere in the padded crop and gets more
    permissive, not less, as the pad grows) or from a Pass 2 touching
    group that includes a label outside the current group. Either way
    causes the WHOLE attempt to be redone from the original labels with
    the group expanded -- Pass 2 never runs on a group that might still
    be missing a member. An already-folded-in neighbor's own touches
    elsewhere are never examined -- see grow_correct_label_3d's own
    inline comments for why that matters (a large/sprawling neighbor
    could otherwise cascade the group into everything IT happens to
    touch, unrelated to what was actually asked to be corrected).

    Returns (new_labels, report): group, pad_used, n_iterations,
    converged, group_grew, per_label_reports (keyed by label id --
    only the ORIGINALLY-REQUESTED label(s) get an entry, each its own
    correct_label_from_intensity_3d() report from the final attempt; a
    folded-in neighbor never gets one, since it never goes through
    Pass 1 at all -- see the module docstring for why).

    Raises ValueError only if even the first attempt's single-label
    correction fails outright (same errors correct_label_from_intensity_3d()
    itself raises).
    """
    def _report(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    group = set(label_ids) if isinstance(label_ids, (list, tuple, set)) else {int(label_ids)}
    original_group = frozenset(group)  # convergence only ever judged on these, never a folded-in neighbor
    pad = int(initial_pad)
    group_grew = False
    last_new_labels = labels
    last_per_label_reports: "dict[int, dict]" = {}
    converged = False
    used_pad = pad
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        used_pad = pad
        working = labels
        per_label_reports: "dict[int, dict]" = {}
        new_neighbors: "set[int]" = set()

        # Pass 1 only ever runs the expensive independent whole-cell 3D
        # walk for the ORIGINALLY-REQUESTED label(s) -- never for a
        # neighbor that got folded in purely to protect its own
        # territory near the target. A folded-in neighbor was never
        # asked to be corrected: recorrecting its ENTIRE 3D extent from
        # its own centroid (which can be a large, complex, sprawling
        # cell spanning most of the fish's own Z range) purely so Pass 2
        # has a marker for the one or two slices where it actually
        # touches the target is pure wasted work -- and, on a real
        # production fish, was the dominant cost of this whole
        # operation. Pass 2 below only needs each group member's
        # CURRENT existing footprint as a watershed marker (exactly how
        # Correct Adjacent Labels already works standalone, with no
        # Pass-1-style pre-correction on either label at all) -- a
        # folded-in neighbor's pre-correction footprint is a perfectly
        # valid marker for that.
        _report(f"Attempt {iteration}: pad={used_pad}px, group={sorted(group)} -- Pass 1 (independent, original label(s) only)...")
        for lid in sorted(group & original_group):
            working, rep = correct_label_from_intensity_3d(
                working, image, lid, lo, pad=used_pad,
                min_volume=min_volume, final_min_fraction=final_min_fraction,
                # Deliberately the FIXED initial_pad, not the growing
                # used_pad: this bound exists only to stop a walk from
                # leaking into a genuinely-touching-but-not-yet-corrected
                # neighbor's own real signal and cascading along however
                # far THAT signal extends -- legitimate Z-growth for this
                # label's own real signal is already handled by the
                # walk's own natural stop-when-nothing-connects logic, no
                # extra room needed for that. If z_extent_pad grew in
                # lockstep with used_pad (needed for genuinely large XY
                # padding), it would eventually relax enough to reach the
                # very neighbor it was meant to guard against.
                z_extent_pad=initial_pad,
            )
            per_label_reports[lid] = rep
            # Discovery is driven ONLY by the originally-requested
            # label(s)' own GENUINE TOUCHING adjacency (foreign_touching)
            # -- never foreign_nearby, which flags anything merely
            # PRESENT somewhere in the padded crop and gets more
            # permissive, not less, as the pad grows (it would keep
            # finding *something* nearby indefinitely). A folded-in
            # neighbor never gets its own Pass 1 report at all now (see
            # above), so this loop naturally can't cascade off one --
            # this filter is kept anyway as the single source of truth
            # for "whose touching-adjacency matters here".
            for ids in rep["foreign_touching"].values():
                new_neighbors.update(ids)
        new_neighbors -= group

        _report(f"Attempt {iteration}: Pass 2 (touching-group check)...")
        groups_by_z = touching_groups_for_stack(working)
        pass2_jobs: "list[tuple[int, list[int]]]" = []
        for z, tgroups in groups_by_z.items():
            for tgroup in tgroups:
                # Same restriction as Pass 1 above: only a touching
                # group that actually involves an originally-requested
                # label is relevant here at all -- two already-folded-in
                # (or wholly unrelated) labels touching each other
                # somewhere else in the fish is simply none of this
                # operation's business.
                if not (set(tgroup) & original_group):
                    continue
                extra = set(tgroup) - group
                if extra:
                    new_neighbors.update(extra)
                    continue
                pass2_jobs.append((z, sorted(tgroup)))

        if new_neighbors:
            group |= new_neighbors
            group_grew = True
            _report(f"Growing group to include neighbor(s) {sorted(new_neighbors)} -> {sorted(group)}, redoing this attempt")
            last_new_labels = working
            last_per_label_reports = per_label_reports
            continue

        for z, tgroup in pass2_jobs:
            try:
                working, _info = correct_label_group_2d(
                    working, image, tgroup, z, lo, pad=used_pad, sigma=sigma,
                    # Same scoping principle as Pass 1 and the 2D
                    # orchestrator above: the rectangle is sized from
                    # only the originally-requested label(s) actually
                    # present in this touching group, never a folded-in
                    # neighbor's own extent.
                    focus_ids=sorted(set(tgroup) & original_group),
                )
            except ValueError:
                pass  # same tolerant skip auto_contrast_correct_stack's own Pass 2 uses

        last_new_labels = working
        last_per_label_reports = per_label_reports

        # Convergence is judged ONLY on the originally-requested label(s)
        # -- a neighbor folded in purely to protect its own territory was
        # never asked to be grown to its own true extent, so its border
        # status must not keep this looping (see grow_correct_label_2d's
        # own matching comment).
        any_touched_border = any(
            per_label_reports[lid]["touched_border"] for lid in original_group
        )
        if not any_touched_border:
            converged = True
            break
        pad += growth_step

    report = {
        "group": sorted(group),
        "pad_used": used_pad,
        "n_iterations": iteration,
        "converged": converged,
        "group_grew": group_grew,
        "per_label_reports": last_per_label_reports,
    }

    # Final whole-group debris cleanup -- same golden-ratio safety net as
    # every other final-safety-net stage in this plugin (see
    # auto_contrast_correct_stack()'s own step 5). Pass 1's own per-label
    # cleanup already ran, but Pass 2's watershed cuts can still leave a
    # small disconnected sliver behind at a cut that Pass 1 never saw
    # (it ran before Pass 2 touched that boundary). Scoped to just this
    # group's own labels, not the whole fish.
    n_debris_removed_total = 0
    if min_volume is not None:
        threshold = final_min_fraction * min_volume
        for lid in sorted(group):
            last_new_labels, n_removed = remove_debris_for_label(last_new_labels, lid, threshold)
            n_debris_removed_total += n_removed
    report["n_debris_removed_px"] = n_debris_removed_total

    return last_new_labels, report


def format_grow_report(report: dict, mode: str) -> str:
    lines = []
    group = report["group"]
    lines.append(
        f"Auto-grow ({mode}): {report['n_iterations']} attempt(s), final pad={report['pad_used']}px, "
        f"group={group}{' (grew from neighbor discovery)' if report['group_grew'] else ''}"
    )
    if report["converged"]:
        lines.append("  Converged -- no part of the result touches the padded region's own edge.")
    else:
        lines.append(
            "  NOT converged -- signal still reaches the padded region's edge after "
            f"{report['n_iterations']} attempt(s). Real signal may extend further; "
            "consider a larger starting pad or more max iterations, or correct this cell by hand."
        )
    if "n_debris_removed_px" in report:
        lines.append(f"  Debris removed: {report['n_debris_removed_px']} px")
    return "\n".join(lines)
