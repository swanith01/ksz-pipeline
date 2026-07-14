#!/usr/bin/env python
"""
Script: v4 angular vs. rectilinear lightcone kSZ D_ell, same parameters.

Read this before running: this is a genuinely first-pass script against
a newer/less-audited API (py21cmfast 4.1.0) than everything else in this
pipeline. Every field-access assumption below is either confirmed via
inspect.signature probes (see lightcone_v4.py's module docstring for
exactly what was and wasn't confirmed) or flagged with a [check] print
that reports both plausible interpretations so the right one can be
picked empirically from real output -- same discipline that caught the
H0-division and density-double-count bugs in the v3 pipeline, applied
here from the start instead of after the fact.

Specifically NOT yet resolved, flagged inline where relevant:
  - Whether 'density' is raw delta (needs +1) or already (1+delta) --
    v3's native lightcone.density turned out to be the latter (a
    skewed_los.py/rotation artifact); v4's field extraction is a
    different code path and there's no a priori reason to expect the
    same artifact, but don't assume either way -- read the printed mean.
  - velocity_z's units -- printed under both possible interpretations
    (already cm/s vs already Mpc/s) against the same ~1e-17 to 1e-18
    Mpc/s target used throughout this session; whichever matches wins.
  - The angular pixel scale isn't read from any LightCone attribute
    (none confirmed to expose it) -- computed directly from
    like_rectilinear's documented behavior (pixel size matched to the
    rectilinear resolution AT match_at_z), which should be
    self-consistent by construction, but isn't independently verified
    against the library's own internal value.

Deliberately small defaults (HII_DIM=32, BOX_LEN=100 Mpc) so a first run
that turns out wrong in some assumption above costs minutes, not hours.
"""

import argparse

import numpy as np
from astropy.cosmology import Planck18 as cosmo

from ksz_pipeline.ksz.lightcone_v4 import (build_inputs, build_node_redshifts,
                                            run_rectilinear, run_angular)
from ksz_pipeline.ksz.lightcone_integral import compute_ksz_map, ksz_map_to_Dl
from ksz_pipeline.ksz.angular_lightcone import angular_ksz_map_to_Dl
from ksz_pipeline.ksz.optical_depth import (compute_tau, compute_visibility,
                                             analytic_tau_below, compute_patchy_mask)
from ksz_pipeline.utils.constants import MPC_CM


def _diagnose_velocity(raw_velocity_z):
    """Print both unit interpretations; return neither -- caller picks."""
    rms_raw = float(np.sqrt(np.mean(raw_velocity_z**2)))
    print(f"  [check] rms(velocity_z), raw values         = {rms_raw:.4e}")
    print(f"  [check]   if raw is already cm/s -> /MPC_CM = "
          f"{rms_raw / MPC_CM:.4e} Mpc/s")
    print(f"  [check]   if raw is already Mpc/s (no conv) = "
          f"{rms_raw:.4e} Mpc/s")
    print(f"  [check]   target from the rest of this pipeline: ~1e-17 to 1e-18 Mpc/s")
    print(f"  [check]   -> pick whichever line above lands near that target")


def process_common(fields, red_axis, pos_axis, patchy_check_label):
    """
    Shared post-extraction pipeline: density/velocity interpretation,
    tau/visibility/patchy-mask -- identical logic for both lightconer
    types, only the final map->Dl step differs (handled by the caller).
    """
    print(f"  [check] mean(density) = {np.mean(fields['density']):.4f}  "
          f"(if ~0: raw delta, use 1.0+density below (current default). "
          f"If ~1: already (1+delta), do NOT add 1 -- edit density_1plus "
          f"below if so)")
    density_1plus = 1.0 + fields['density']   # ASSUMES raw delta -- see check above
    x_HII_field   = 1.0 - fields['neutral_fraction']

    _diagnose_velocity(fields['velocity_z'])
    v_los_Mpc_s = fields['velocity_z'] / MPC_CM   # ASSUMES raw cm/s -- see check above

    x_e_interp = 1.0 - fields['neutral_fraction'].mean(axis=(0, 1))
    tau0 = analytic_tau_below(red_axis.min())
    z_mid, ds, dtau, tau = compute_tau(x_e_interp, red_axis, pos_axis, tau0=tau0)
    tau_at_lc, visibility, visibility_3D = compute_visibility(tau, red_axis, z_mid)
    patchy_mask, patchy_mask_3D = compute_patchy_mask(x_e_interp)

    z_patchy = red_axis[patchy_mask.astype(bool)]
    if z_patchy.size > 0:
        print(f"  [{patchy_check_label}] patchy regime: z = "
              f"{z_patchy.max():.2f} -> {z_patchy.min():.2f} "
              f"({z_patchy.size}/{len(red_axis)} slices)")

    ksz_map = compute_ksz_map(density_1plus, x_HII_field, v_los_Mpc_s,
                               red_axis, ds, visibility_3D,
                               patchy_mask_3D=patchy_mask_3D)
    return ksz_map


def main(HII_DIM, BOX_LEN, z_min, z_max, match_at_z, random_seed,
         HII_EFF_FACTOR, cache_dir, n_threads, n_nodes):
    node_redshifts = build_node_redshifts(z_min, z_max, n_nodes=n_nodes)
    print(f"node_redshifts: {n_nodes} points, log-(1+z)-spaced, "
          f"{node_redshifts.min():.2f} -> {node_redshifts.max():.2f}")
    inputs = build_inputs(random_seed, HII_DIM, BOX_LEN, node_redshifts,
                           HII_EFF_FACTOR=HII_EFF_FACTOR, N_THREADS=n_threads)
    cell_size = BOX_LEN / HII_DIM

    print("=" * 60)
    print("RECTILINEAR")
    print("=" * 60)
    lc_rect, fields_rect = run_rectilinear(inputs, z_min, z_max, cache_dir,
                                            resolution_mpc=cell_size)
    red_axis_rect = np.asarray(lc_rect.lightcone_redshifts)
    pos_axis_rect = np.asarray(lc_rect.lightcone_distances)
    ksz_map_rect = process_common(fields_rect, red_axis_rect, pos_axis_rect, "rect")
    ell_rect, Dl_rect, Dlerr_rect = ksz_map_to_Dl(ksz_map_rect, BOX_LEN)
    D3000_rect = float(np.interp(3000, ell_rect, Dl_rect)) if len(ell_rect) else float('nan')
    print(f"  ksz_map shape: {ksz_map_rect.shape}")
    print(f"  D_3000 (rectilinear) = {D3000_rect:.4g} uK^2")

    print()
    print("=" * 60)
    print("ANGULAR")
    print("=" * 60)
    lc_ang, fields_ang = run_angular(inputs, match_at_z, z_max, cache_dir)
    red_axis_ang = np.asarray(lc_ang.lightcone_redshifts)
    pos_axis_ang = np.asarray(lc_ang.lightcone_distances)
    ksz_map_ang = process_common(fields_ang, red_axis_ang, pos_axis_ang, "ang")

    chi_match = cosmo.comoving_distance(match_at_z).value
    pixel_scale_rad = cell_size / chi_match
    print(f"  [check] pixel_scale_rad = cell_size/chi(match_at_z) = "
          f"{cell_size:.4f}/{chi_match:.1f} = {pixel_scale_rad:.4e} rad "
          f"(computed from like_rectilinear's documented matching "
          f"behavior, not read from a LightCone attribute -- see script "
          f"docstring)")
    ell_ang, Dl_ang, Dlerr_ang = angular_ksz_map_to_Dl(ksz_map_ang, pixel_scale_rad)
    D3000_ang = float(np.interp(3000, ell_ang, Dl_ang)) if len(ell_ang) else float('nan')
    print(f"  ksz_map shape: {ksz_map_ang.shape}")
    print(f"  D_3000 (angular) = {D3000_ang:.4g} uK^2")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  D_3000 rectilinear : {D3000_rect:.4g} uK^2")
    print(f"  D_3000 angular     : {D3000_ang:.4g} uK^2")
    if D3000_rect and not np.isnan(D3000_rect):
        print(f"  ratio angular/rect : {D3000_ang/D3000_rect:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--HII_DIM", type=int, default=32)
    parser.add_argument("--BOX_LEN", type=float, default=100.0)
    parser.add_argument("--z_min", type=float, default=4.0)
    parser.add_argument("--z_max", type=float, default=20.0)
    parser.add_argument("--match_at_z", type=float, default=7.5)
    parser.add_argument("--random_seed", type=int, default=37)
    parser.add_argument("--HII_EFF_FACTOR", type=float, default=30.0)
    parser.add_argument("--cache_dir", default="data/cache_v4_quicktest")
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--n_nodes", type=int, default=30)
    args = parser.parse_args()
    main(args.HII_DIM, args.BOX_LEN, args.z_min, args.z_max, args.match_at_z,
         args.random_seed, args.HII_EFF_FACTOR, args.cache_dir, args.n_threads,
         args.n_nodes)
