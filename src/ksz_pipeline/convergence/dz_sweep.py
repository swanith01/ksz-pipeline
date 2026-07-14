"""
Redshift-sampling-density (dz) convergence sweeps, for both coeval
(Cain + Georgiev) and stitched-lightcone methods.

Adapted from the project's older kSZ_boxes_dz_convergence.py (different,
pre-migration repo): compute the FINEST z grid once, then coarser grids
are literal subsets of it -- reusing cached per-redshift results with
zero extra py21cmfast runs, rather than treating each dz as an
independent simulation. That subset-reuse happens automatically here:
coeval_sweep.run_one_config's cache check (`missing = [z for z in
z_snapshots if z not in results]`) comes back empty for any subset of an
already-cached grid, as long as the SAME cache tag is used across dz
variants for a given (BOX_LEN, HII_DIM).

Unlike box-size/resolution sweeps, dz variants are subsampling factors
of one reference z-grid (coeval_ksz.z_snapshots in the config), not an
independently-specified list -- the reference grid isn't perfectly
uniform in z (finer near the reionization transition, coarser at the
tails, same as the original script's dz=0.25/0.50 convention was for a
uniform grid), so "every Nth point" is the honest generalization rather
than assuming a fixed dz value applies globally.
"""

import numpy as np


def build_dz_subsets(z_fine, dz_multiples):
    """
    Build coarser redshift-grid subsets from a fine reference grid.

    Parameters
    ----------
    z_fine       : sequence of float, the finest available z grid
                   (e.g. coeval_ksz.z_snapshots from the config)
    dz_multiples : sequence of int, subsampling factors -- 1 keeps every
                   point (the fine grid itself), 2 keeps every other
                   point, etc.

    Returns
    -------
    dict {multiple: subset_list}, each subset sorted descending (highest
    z first, matching this codebase's usual convention)
    """
    z_sorted = sorted(z_fine, reverse=True)
    return {m: z_sorted[::m] for m in dz_multiples}


def run_dz_sweep_coeval(z_fine, dz_multiples, BOX_LEN, HII_DIM, cache_dir,
                         tag, force=False, N_THREADS=None):
    """
    Coeval Cain+Georgiev D_ell at multiple redshift-sampling densities,
    all subsets of z_fine, at FIXED (BOX_LEN, HII_DIM) -- isolates dz
    sensitivity from box-size/resolution effects.

    Parameters
    ----------
    z_fine       : sequence of float, finest reference z grid
    dz_multiples : sequence of int, e.g. [1, 2, 4]
    BOX_LEN, HII_DIM : float, int -- held fixed across all dz variants
    cache_dir    : str
    tag          : str -- SAME tag used for every dz variant here, so
                   they share one cache file and only the finest grid
                   (dz_multiples.min()) triggers real py21cmfast runs
    force        : bool, passed to the first (finest) call only --
                   forcing every subsequent subset call would defeat the
                   whole point of this function
    N_THREADS    : int, optional -- see coeval/fields.py. Only the first
                   (finest-grid) call below actually triggers new
                   py21cmfast runs; passed to both anyway for safety in
                   case dz_multiples doesn't fully cover what's cached.

    Returns
    -------
    dict keyed by f"dz_x{multiple}" -> coeval_sweep.run_one_config's
    return dict
    """
    from .coeval_sweep import run_one_config

    subsets = build_dz_subsets(z_fine, dz_multiples)
    finest_m = min(dz_multiples)

    # Populate the cache with the finest grid FIRST -- every coarser
    # subset below then finds everything it needs already cached.
    print(f"  Populating fine-grid cache (dz_x{finest_m}, "
          f"{len(subsets[finest_m])} redshifts)...", flush=True)
    _ = run_one_config(BOX_LEN, HII_DIM, subsets[finest_m], cache_dir, tag,
                        force=force, N_THREADS=N_THREADS)

    results = {}
    for m in sorted(dz_multiples):
        label = f"dz_x{m}"
        print(f"  {label}: {len(subsets[m])} redshifts "
              f"(subset of the cached fine grid, no new py21cmfast runs)",
              flush=True)
        results[label] = run_one_config(BOX_LEN, HII_DIM, subsets[m],
                                         cache_dir, tag, force=False,
                                         N_THREADS=N_THREADS)
        print(f"    D_3000 direct={results[label]['D3000_direct']:.4g} uK^2  "
              f"georgiev={results[label]['D3000_georgiev']:.4g} uK^2", flush=True)
    return results


def run_dz_sweep_stitched(z_fine, dz_multiples, BOX_LEN, HII_DIM, z_min, z_max,
                           cache_dir, angle_deg=0.0):
    """
    Stitched-lightcone D_ell at multiple SNAPSHOT-sampling densities
    (how many coeval boxes get interpolated between), at fixed
    (BOX_LEN, HII_DIM) and fixed LOS pixel grid (build_los_z_grid, tied
    to cell_size, unaffected by this sweep). This is a genuinely
    different question from the coeval dz test above: not "how finely
    sampled does the C_ell Limber integral need to be" but "how many
    snapshots does the stitcher need to interpolate an accurate
    continuous field from."

    Reuses py21cmfast's own cache (direc=cache_dir) across dz variants
    for efficiency -- run_coeval_fields calls for redshifts already
    simulated (from a finer dz variant run earlier) are read from cache,
    not recomputed, same principle as the coeval version above but via
    a different cache layer.

    Parameters
    ----------
    z_fine, dz_multiples : see build_dz_subsets
    BOX_LEN, HII_DIM      : fixed across all dz variants
    z_min, z_max          : LOS redshift range (independent of dz --
                             see module docstring)
    cache_dir              : str
    angle_deg               : float, default 0

    Returns
    -------
    dict keyed by f"dz_x{multiple}" -> stitched_sweep.run_one_config's
    return dict
    """
    from .stitched_sweep import run_one_config

    subsets = build_dz_subsets(z_fine, dz_multiples)
    results = {}
    for m in sorted(dz_multiples):
        label = f"dz_x{m}"
        print(f"  {label}: {len(subsets[m])} coeval snapshots to stitch from",
              flush=True)
        results[label] = run_one_config(BOX_LEN, HII_DIM, subsets[m],
                                         z_min, z_max, cache_dir, label,
                                         angle_deg=angle_deg)
        print(f"    D_3000={results[label]['D3000']:.4g} uK^2", flush=True)
    return results
