"""
Stitched-lightcone convergence sweeps (box-size and resolution).

Mirrors coeval_sweep.py's structure: one sweep function handles both
convergence types via the (BOX_LEN, HII_DIM, tag) list callers build.
See scripts/05_convergence_stitched.py.

Reuses, unchanged: ksz.stitch_from_coeval (stitch_lightcone_from_coeval,
build_los_z_grid), ksz.optical_depth (compute_tau, compute_visibility,
analytic_tau_below, compute_patchy_mask), ksz.lightcone_integral
(compute_ksz_map, ksz_map_to_Dl) -- exactly the pipeline
03_stitched_lightcone_crosscheck.py runs for a single configuration,
looped here over a parameter list.

CHANGES (2026-07-22):
1. run_one_config now accepts optional z_lo/z_hi/chi_Mpc, used by the dz
   sweep (script 13) to match the closure test's unified window and
   chi_eff. Defaults to None -- OLD BEHAVIOR (stitched's own native
   window, chi_Mpc=7800 default) is preserved for existing callers
   (script 05's box-size/resolution sweep, via run_sweep) that don't
   pass these.
2. compute_ksz_map now explicitly passes ne0=ne0_cgs() (the
   helium-inclusive value, commit 6fd040e), fixing a previously
   undiscovered instance of the ne0 convention mismatch -- this call
   was silently using compute_ksz_map's NE0_HYDROGEN_ONLY default.
   This is UNCONDITIONAL (not gated behind z_lo/z_hi/chi_Mpc), so any
   future rerun of script 05's sweep will also shift by the same
   confirmed ~0.4% effect.
"""

import os

import numpy as np

# stitch_lightcone_from_coeval is imported lazily inside run_one_config,
# not here -- same reasoning as coeval_sweep.py: it pulls in py21cmfast
# transitively, and this module's own logic doesn't need to.
from ..ksz.stitch_from_coeval import build_los_z_grid, comoving_distance_mpc
from ..ksz.optical_depth import (compute_tau, compute_visibility,
                                  analytic_tau_below, compute_patchy_mask)
from ..ksz.lightcone_integral import compute_ksz_map, ksz_map_to_Dl
from ..utils.constants import MPC_CM, ne0_cgs


def run_one_config(BOX_LEN, HII_DIM, z_snapshots, z_min, z_max, cache_dir,
                    tag, angle_deg=0.0, N_THREADS=None, random_seed=None,
                    z_lo=None, z_hi=None, chi_Mpc=None):
    """
    Stitch a lightcone and compute D_ell for one (BOX_LEN, HII_DIM)
    configuration. No pickle-level caching of its own here (unlike
    coeval_sweep.run_one_config) -- the expensive part, the underlying
    py21cmfast coeval boxes, is already cached via cache_dir/direc as
    usual; re-running the stitch+map+FFT on top of those is cheap.

    Parameters
    ----------
    BOX_LEN, HII_DIM : float, int
    z_snapshots       : sequence of float, coeval snapshot redshifts
    z_min, z_max      : float, LOS redshift range (should bracket
                        z_snapshots with margin)
    cache_dir         : str, py21cmfast cache directory
    tag               : str, label for this configuration (used only in
                        print statements here, not a cache key)
    angle_deg         : float, default 0 -- see stitch_from_coeval's
                        module docstring for why
    N_THREADS         : int, optional -- see coeval/fields.py; pass
                        explicitly (e.g. config's 21cmfast.N_THREADS)
                        rather than relying solely on OMP_NUM_THREADS
                        being set in whatever context this runs in
    z_lo, z_hi        : float, optional. If BOTH given, the LOS integral
                        is truncated to this z-range BEFORE building the
                        map (matching the unified/matched window from
                        the closure test, script 14) instead of relying
                        on this function's own patchy_mask_3D threshold
                        over the full z_min-z_max range. Default None --
                        old behavior (stitched's own native window).
    chi_Mpc           : float, optional. If given, passed through to
                        ksz_map_to_Dl in place of its hardcoded
                        chi_Mpc=7800 default. Default None -- old
                        behavior (7800 default).

    Returns
    -------
    dict: ell, Dl, D3000, ksz_map_rms
    """
    from ..ksz.stitch_from_coeval import stitch_lightcone_from_coeval

    cell_size = BOX_LEN / HII_DIM
    z_arr = build_los_z_grid(z_min, z_max, cell_size)

    stitched = stitch_lightcone_from_coeval(
        z_snapshots=z_snapshots, z_arr=z_arr,
        HII_DIM=HII_DIM, BOX_LEN=BOX_LEN,
        cache_dir=cache_dir, angle_deg=angle_deg,
        N_THREADS=N_THREADS, random_seed=random_seed,
    )

    density_1plus = 1.0 + stitched['density']
    x_HII_field   = 1.0 - stitched['xH_box']
    v_los_Mpc_s   = stitched['velocity_z'] / MPC_CM

    x_e_interp = 1.0 - stitched['xH_box'].mean(axis=(0, 1))
    pos_axis   = stitched['pos_axis']

    tau0 = analytic_tau_below(z_arr.min())
    z_mid, ds, dtau, tau = compute_tau(x_e_interp, z_arr, pos_axis, tau0=tau0)
    tau_at_lc, visibility, visibility_3D = compute_visibility(tau, z_arr, z_mid)
    patchy_mask, patchy_mask_3D = compute_patchy_mask(x_e_interp)

    if z_lo is not None and z_hi is not None:
        # Matched-window mode (script 14 convention): truncate to the
        # SAME unified window used for the closure test, rather than
        # this variant's own native patchy_mask_3D threshold on the
        # full z_min-z_max range. Isolates the interpolation-density
        # variable from window-definition drift across dz variants.
        i0 = np.searchsorted(z_arr, z_lo)
        i1 = np.searchsorted(z_arr, z_hi)
        density_1plus  = density_1plus[:, :, i0:i1]
        x_HII_field    = x_HII_field[:, :, i0:i1]
        v_los_Mpc_s    = v_los_Mpc_s[:, :, i0:i1]
        z_arr_used     = z_arr[i0:i1]
        ds_used        = ds[i0:i1 - 1]
        visibility_3D  = visibility_3D[:, :, i0:i1]
        patchy_mask_3D = patchy_mask_3D[:, :, i0:i1]
    else:
        z_arr_used, ds_used = z_arr, ds

    ksz_map = compute_ksz_map(density_1plus, x_HII_field, v_los_Mpc_s,
                               z_arr_used, ds_used, visibility_3D,
                               ne0=ne0_cgs(), patchy_mask_3D=patchy_mask_3D)

    if chi_Mpc is not None:
        ell, Dl, Dl_err = ksz_map_to_Dl(ksz_map, BOX_LEN, chi_Mpc=chi_Mpc)
    else:
        ell, Dl, Dl_err = ksz_map_to_Dl(ksz_map, BOX_LEN)

    D3000 = float(np.interp(3000, ell, Dl)) if len(ell) else float('nan')

    return dict(ell=ell, Dl=Dl, Dl_err=Dl_err, D3000=D3000,
                ksz_map_rms=float(np.sqrt(np.mean(ksz_map**2))),
                n_lc_pix=len(z_arr_used), cell_size=cell_size)


def run_sweep(param_list, z_snapshots, z_min, z_max, cache_dir,
              angle_deg=0.0, N_THREADS=None, random_seed=None):
    """
    Run run_one_config for each configuration in param_list.

    Unchanged behavior: does not pass z_lo/z_hi/chi_Mpc, so this
    continues to use stitched's own native window and the 7800 default
    -- box-size/resolution sweep results are only affected by the
    unconditional ne0 fix inside run_one_config (~0.4%, confirmed small).

    Parameters
    ----------
    param_list  : list of (BOX_LEN, HII_DIM, tag) tuples
    z_snapshots : sequence of float, same for every configuration
    z_min, z_max, cache_dir, angle_deg, N_THREADS : see run_one_config

    Returns
    -------
    dict keyed by tag -> run_one_config's return dict
    """
    out = {}
    for BOX_LEN, HII_DIM, tag in param_list:
        dx = BOX_LEN / HII_DIM
        print(f"=== {tag}: BOX_LEN={BOX_LEN} Mpc, HII_DIM={HII_DIM} "
              f"(dx={dx:.3f} Mpc) ===", flush=True)
        out[tag] = run_one_config(BOX_LEN, HII_DIM, z_snapshots, z_min, z_max,
                                   cache_dir, tag, angle_deg=angle_deg,
                                   N_THREADS=N_THREADS, random_seed=random_seed)
        print(f"  n_lc_pix={out[tag]['n_lc_pix']}  "
              f"D_3000={out[tag]['D3000']:.4g} uK^2", flush=True)
    return out
