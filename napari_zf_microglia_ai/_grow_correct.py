"""
_grow_correct.py -- auto-growing wrapper around Correct Label / Correct
Adjacent Labels: retries the same threshold-based correction with a
progressively larger padded working region whenever the result's own
footprint touches the edge of that region -- catching real signal that
too small a pad would otherwise cut off.

2D mode (grow_correct_label_2d) reuses correct_label_group_2d() directly
(a single-ID call degenerates to plain single-label correction, so one
function already covers both the 1-label and N-label case uniformly).
Also auto-detects when growth reveals a neighboring label and expands
into a joint multi-label correction so a neighbor's territory is never
wrongly consumed -- this never grows silently into someone else's cell.
Neighbor discovery here is deliberately narrow on two axes, to stop it
ever cascading into an unrelated part of the fish: (1) only GENUINE
TOUCHING adjacency counts, never mere presence nearby -- a "how much
padding is there room for" search would otherwise keep finding
*something* within an ever-growing box indefinitely; (2) only the
ORIGINALLY-REQUESTED label(s)' own touches are ever examined -- once a
neighbor is folded into the group purely to protect its own territory
near the target, that neighbor's own touches elsewhere are never looked
at, since a large/sprawling label folded in this way could otherwise
cascade the group into everything else it happens to touch.

3D mode (grow_correct_label_3d) is now just a single, thin pass-
through to correct_label_from_intensity_3d(auto_grow=True) -- no
separate neighbor-discovery/group-folding pass, unlike 2D (that's
because correct_label_from_intensity_3d() itself already resolves any
genuinely adjacent label ENTIRELY ON ITS OWN, per slice, as part of its
own walk -- see its own docstring in _labeling.py), and no outer
retry-the-whole-cell-at-a-bigger-pad loop either any more: growth is
PER-SLICE, entirely internal to that function's own walk. A cell with
one long branch on a single slice only ever regrows that one slice's
own pad -- every other slice keeps its own already-correct result and
its own (smaller) pad untouched, which is both cheaper and converges
far more easily than redoing the whole cell whenever any one slice
needs more room.
"""

from __future__ import annotations

import numpy as np

from ._labeling import (
    correct_label_from_intensity_3d,
    correct_label_group_2d,
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
    Auto-grows Correct Label's 3D whole-cell correction. UNLIKE the 2D
    orchestrator above, this is now just a single, thin pass-through to
    correct_label_from_intensity_3d(auto_grow=True) -- growth itself
    happens entirely INSIDE that function's own per-slice walk, not
    here. A cell with 40 well-behaved slices and one long branch on
    slice 23 only ever regrows slice 23's own pad, at whatever size IT
    needs -- every other slice keeps its own already-correct result and
    its own (smaller) pad. This is both cheaper (no reason to reprocess
    30+ already-fine slices just because one needed more room) and
    converges far more easily (one slice needing extra room no longer
    means the WHOLE cell has to be redone at that bigger pad before it
    can be judged converged) -- see correct_label_from_intensity_3d()'s
    own auto_grow docstring for the full rationale. An earlier version
    of this function instead redid the entire cell from scratch with a
    bigger GLOBAL pad on every retry, exactly like the 2D orchestrator
    still does (2D genuinely needs that, since Correct Adjacent Labels'
    2-label case has no "per slice" concept at all -- it only ever
    touches the one slice the user is looking at).

    No separate neighbor-discovery/group-folding pass is needed here
    either (an even earlier version had one, mirroring the 2D
    orchestrator's Pass 1 + Pass 2 split): correct_label_from_intensity_3d()
    itself now resolves any genuinely adjacent label ENTIRELY ON ITS
    OWN, per slice, as part of its own walk (see its own docstring) --
    there's nothing left over for a second pass to discover or fix.

    label_ids : a single int in every real use today (3D-mode "Correct
                Label" only ever corrects one label; "Correct Adjacent
                Labels" is 2D-only). Accepted as int | list[int] for
                signature parity with the 2D orchestrator -- if a list
                is ever given, only its first element (sorted) is used,
                matching that same "first = the label actually being
                corrected" convention.

    Returns (new_labels, report): group (the corrected label plus every
    OTHER label reported as foreign_nearby by the final attempt --
    informational, for sanding/reporting, not something this function
    itself grows into), pad_used (the BASE pad -- growth is per-slice
    now, see slices_grown), n_iterations (the CAP each slice may use,
    not an actual global attempt count), converged, group_grew (always
    False now -- kept for report-shape compatibility), slices_grown
    ({z: final pad used}, only for slices that actually needed more
    than the base pad), per_label_reports ({label_id: the underlying
    correct_label_from_intensity_3d() report}).

    Raises ValueError only if even the first attempt fails outright
    (same errors correct_label_from_intensity_3d() itself raises).
    """
    def _report(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    if isinstance(label_ids, (list, tuple, set)):
        label_id = int(sorted(int(i) for i in label_ids)[0])
    else:
        label_id = int(label_ids)

    _report(f"Correcting label={label_id}, base pad={initial_pad}px, per-slice auto-grow up to {max_iterations} attempt(s)...")
    new_labels, rep = correct_label_from_intensity_3d(
        labels, image, label_id, lo, pad=initial_pad,
        min_volume=min_volume, final_min_fraction=final_min_fraction,
        # Deliberately the FIXED initial_pad, not a growing value: this
        # bound stops the walk's own Z-extension from leaking into a
        # genuinely-touching-but-different structure and cascading
        # along however far THAT signal extends -- legitimate Z-growth
        # for this label's own real signal is already handled by the
        # walk's own copy-and-verify logic, no extra room needed for
        # that. If it grew in lockstep with the per-slice Y/X pad, it
        # would eventually relax enough to reach whatever it was meant
        # to guard against.
        z_extent_pad=initial_pad,
        sigma=sigma,
        # Growth is now entirely INSIDE the walk, per slice -- see
        # correct_label_from_intensity_3d()'s own auto_grow docstring.
        # No outer retry loop needed here any more: a single call
        # already lets each slice grow only as much as IT individually
        # needs, instead of redoing the whole cell at a bigger pad
        # every time any one slice touches its own edge.
        auto_grow=True, growth_step=growth_step, max_iterations=max_iterations,
    )
    converged = not rep["touched_border"]

    group = sorted({label_id} | {i for ids in rep.get("foreign_nearby", {}).values() for i in ids})

    report = {
        "group": group,
        "pad_used": initial_pad,  # the BASE pad -- growth is per-slice now, see slices_grown
        "n_iterations": max_iterations,  # the CAP each slice may use, not an actual global count
        "converged": converged,
        "group_grew": False,
        "per_label_reports": {label_id: rep},
        "slices_grown": rep.get("slices_grown", {}),
    }
    report["n_debris_removed_px"] = rep.get("n_debris_removed_px", 0)
    return new_labels, report


def format_grow_report(report: dict, mode: str) -> str:
    lines = []
    group = report["group"]
    is_3d = mode.upper().startswith("3D")
    if is_3d:
        # Growth is per-slice in 3D now -- there's no single "final
        # pad" or "attempt count" for the whole cell any more, just a
        # base pad and a per-slice cap; see slices_grown below for what
        # actually happened.
        lines.append(
            f"Auto-grow (3D, per-slice): base pad={report['pad_used']}px, "
            f"up to {report['n_iterations']} attempt(s) per slice, group={group}"
        )
        slices_grown = report.get("slices_grown", {})
        if slices_grown:
            grown_txt = ", ".join(f"{z}: {p}px" for z, p in sorted(slices_grown.items()))
            lines.append(f"  Slice(s) that needed a bigger pad: {grown_txt}")
        else:
            lines.append("  No slice needed more than the base pad.")
    else:
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
        # 3D mode only: per_label_reports carries each label's own
        # border_touching_slices (2D's own report has no per-slice
        # concept at all -- it only ever touches the ONE slice the
        # caller gave it, already known to whoever's reading this).
        # Naming exactly which slice(s) are still cut off (even after
        # THEIR OWN per-slice growth was exhausted) lets a user go
        # correct those individually (a bigger pad on just that slice
        # via Correct Label 2D, or by hand) instead of guessing.
        still_touching = sorted({
            z for rep in report.get("per_label_reports", {}).values()
            for z in rep.get("border_touching_slices", [])
        })
        if still_touching:
            lines.append(
                f"  Still touching the edge on slice(s) {still_touching} -- "
                f"correct those individually (e.g. a bigger pad on just that "
                f"slice via Correct Label 2D, or by hand) rather than growing "
                f"the whole cell further."
            )
    if "n_debris_removed_px" in report:
        lines.append(f"  Debris removed: {report['n_debris_removed_px']} px")
    return "\n".join(lines)
