"""
_sieve.py — generic coarse-to-fine narrowing ("sieve") engine, shared by
every GT-sweep tool that has a continuous primary axis (MONAI Threshold,
BG Threshold, Smooth sigma XY, Cellprob).

Born directly out of a real manual run: sweeping cellprob -6..6 at step
1.0, then narrowing +-1.0 at step 0.25 around the winner, then +-0.5 at
step 0.1 -- done by hand once, then generalized here so any sweep tool
can offer it as a checkbox instead of the user re-running the tool by
hand three times and manually re-centering the range each time.

This module only implements the narrowing loop itself -- it knows
nothing about MONAI, Cellpose, or any specific sweep's parameters. Each
caller supplies a run_stage_fn closure that knows how to run ONE 1D
grid of primary-axis values (holding everything else fixed) and return
that stage's winning primary-axis value plus whatever result object the
underlying run_*_sweep() function normally returns -- run_sieve() never
inspects that result beyond the two things every one of this plugin's
sweep tools already exposes: 'best_point' (from which the caller's
closure extracts the primary-axis value) and 'cancelled'.
"""
from __future__ import annotations

import numpy as np


def sieve_grid(lo: float, hi: float, center: float, half_width: float, step: float) -> list:
    """Build one stage's 1D grid: [center-half_width, center+half_width]
    clipped to [lo, hi], at the given step. Always includes at least one
    point even if the clipped range collapses to a single value."""
    stage_lo = max(lo, center - half_width)
    stage_hi = min(hi, center + half_width)
    if stage_hi <= stage_lo:
        return [max(lo, min(hi, center))]
    return list(np.round(np.arange(stage_lo, stage_hi + step / 2, step), 4))


def run_sieve(run_stage_fn, lo: float, hi: float, coarse_step: float,
              refine_steps: "list[tuple[float, float]]",
              progress_cb=None, cancel_check=None):
    """
    Run a full coarse-to-fine sieve.

    run_stage_fn(values: list) -> (best_value: float, stage_result: dict)
        Runs one stage's grid of primary-axis values (secondary axis /
        everything else held fixed by the caller's own closure) and
        returns the winning primary-axis value plus that stage's raw
        result dict (whatever the underlying run_*_sweep() returns --
        opaque to this function, kept only for reporting/history).
        stage_result is expected to carry a 'cancelled' key; if absent,
        treated as not cancelled.

    lo, hi        : hard bounds for every stage's grid (e.g. -6.0/6.0
                    for cellprob, or a tool's own slider range).
    coarse_step   : stage 1's step across the full [lo, hi] range.
    refine_steps  : list of (half_width, step) for stage 2, 3, ... --
                    e.g. [(1.0, 0.25), (0.5, 0.1)] is exactly the
                    cellprob narrowing validated by hand on real GT
                    data before this engine existed.
    progress_cb(str), cancel_check() -> bool: same contracts as every
        other sweep tool in this plugin. cancel_check is polled between
        stages (a stage's own internal cancel_event, if any, is the
        caller's responsibility -- this loop only decides whether to
        start the *next* stage).

    Returns dict: {
      'stages': [{'values': [...], 'best_value': float, 'result': dict}, ...],
      'final_best_value': float or None,
      'final_result': dict or None,   # last completed stage's result
      'cancelled': bool,
    }
    """
    stages = []
    best_value = None
    final_result = None
    cancelled = False

    values = list(np.round(np.arange(lo, hi + coarse_step / 2, coarse_step), 4))
    stage_specs = [(None, None)] + list(refine_steps)  # stage 1 has no width/step of its own

    for i, (half_width, step) in enumerate(stage_specs):
        if cancel_check is not None and cancel_check():
            cancelled = True
            break

        if i > 0:
            values = sieve_grid(lo, hi, best_value, half_width, step)

        if progress_cb:
            progress_cb(f"Sieve stage {i+1}/{len(stage_specs)}: {len(values)} points, "
                        f"range [{values[0]}, {values[-1]}]")

        best_value, stage_result = run_stage_fn(values)
        stages.append(dict(values=values, best_value=best_value, result=stage_result))
        final_result = stage_result

        if progress_cb:
            progress_cb(f"Sieve stage {i+1}/{len(stage_specs)} winner: {best_value}")

        if isinstance(stage_result, dict) and stage_result.get("cancelled"):
            cancelled = True
            break

    return dict(stages=stages, final_best_value=best_value,
                final_result=final_result, cancelled=cancelled)
