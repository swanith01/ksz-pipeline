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
from ..utils.constants import MPC_CM


def run_one_config(BOX_LEN, HII_DIM, z_snapshots, z_min, z_max, cache_dir,
                    tag, angle_deg=0.0, N_THREADS=None, random_seed=None):
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

    ksz_map = compute_ksz_map(density_1plus, x_HII_field, v_los_Mpc_s,
                               z_arr, ds, visibility_3D,
                               patchy_mask_3D=patchy_mask_3D)
    ell, Dl, Dl_err = ksz_map_to_Dl(ksz_map, BOX_LEN)
    D3000 = float(np.interp(3000, ell, Dl)) if len(ell) else float('nan')

    return dict(ell=ell, Dl=Dl, Dl_err=Dl_err, D3000=D3000,
                ksz_map_rms=float(np.sqrt(np.mean(ksz_map**2))),
                n_lc_pix=len(z_arr), cell_size=cell_size)


def run_sweep(param_list, z_snapshots, z_min, z_max, cache_dir,
              angle_deg=0.0, N_THREADS=None, random_seed=None):
    """
    Run run_one_config for each configuration in param_list.

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
