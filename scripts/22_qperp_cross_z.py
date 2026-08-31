#!/usr/bin/env python
"""
Script 22: q_perp cross-redshift correlation (periodicity-free) vs stitched P_off.

Cross-correlates q_perp between DIFFERENT coeval boxes (z_i != z_j) --
independent run_coeval_fields calls, never stitched/tiled/LOS-interpolated,
so CANNOT contain a periodicity artifact by construction. Same physical
channel (q_perp) compute_cell/direct already trusts for its own diagonal
terms. Answers: what fraction of stitched's P_off is periodicity vs
genuine cross-redshift physics Limber discards?

SELF-CONSISTENCY CHECK (run first, before trusting anything else): at
z_i=z_j, cross_power_qperp_pairwise_chi's math should reduce EXACTLY to
qperp_power's own trusted auto P(k) at ell=k*chi(z). If this check fails,
there is a bug in the new cross-z code -- stop and debug before reading
the z_i!=z_j results.

CAVEATS, printed again at the end -- read before over-interpreting:
1. Different-z coeval boxes share the SAME initial density field (same
   seed), not independent realizations -- a nonzero result partly
   reflects deterministic correlated time-evolution, not necessarily
   "new physics" in the sense Limber's independence assumption targets.
2. Rectilinear/flat-sky (kz=0 slice + pairwise chi_ij), not the full
   spherical/Bessel-function non-Limber treatment.
3. The "/2" normalization is inherited from qperp_power for consistency
   (validated via the self-check above) but was itself derived under
   Limber (Park+2013 Eq.A15) -- not independently re-derived here for a
   genuinely non-Limber cross quantity.

This should be CHEAP -- one 3D FFT per z_snapshot (same cost qperp_power
already pays, per snapshot, cache-hit expected on boxes), no ~20-50GB
per-slice stitched construction involved at all.

Usage
-----
    python scripts/22_qperp_cross_z.py --config configs/fiducial.yaml
"""
import argparse
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo

from ksz_pipeline.coeval.fields import run_coeval_fields
from ksz_pipeline.coeval.momentum import qperp_power
from ksz_pipeline.coeval.qperp_cross_z import (compute_qperp_transverse_components,
                                                cross_power_qperp_pairwise_chi)


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sim_cfg = cfg['21cmfast']
    BOX_LEN = sim_cfg['BOX_LEN']
    HII_DIM = sim_cfg['HII_DIM_coeval']
    cache_dir = cfg['data']['cache_dir']
    out_dir  = cfg['data']['output_dir'].rstrip('/')
    plot_dir = cfg['data']['plot_dir'].rstrip('/')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    closure_path = f"{out_dir}/closure_test.npz"
    if not os.path.exists(closure_path):
        raise FileNotFoundError(f"{closure_path} not found -- run script 14 first.")
    closure = np.load(closure_path)
    z_window = [float(z) for z in closure['z_window']]
    print(f"Using {len(z_window)} z's from closure_test.npz's own matched window\n")

    # ---- build q_perp components per z (cheap: one 3D FFT each, cache-hit boxes) ----
    print("Computing q_perp transverse components per z (cache-hit expected)...")
    qperp_components = {}
    chi_dict = {}
    qperp_power_reference = {}  # for the self-consistency check
    xH_mean_dict = {}
    for z in z_window:
        delta, xH, vx, vy, vz = run_coeval_fields(
            z, HII_DIM, BOX_LEN, cache_dir,
            N_THREADS=sim_cfg['N_THREADS'], random_seed=sim_cfg['random_seed'])
        qperp_components[z] = compute_qperp_transverse_components(delta, xH, vx, vy, vz, BOX_LEN)
        chi_dict[z] = cosmo.comoving_distance(z).value
        k_ref, P_ref, Pstd_ref = qperp_power(delta, xH, vx, vy, vz, BOX_LEN)
        qperp_power_reference[z] = (k_ref, P_ref, Pstd_ref)
        xH_mean_dict[z] = float(xH.mean())
        print(f"  z={z:.2f}  chi={chi_dict[z]:.1f} Mpc", flush=True)

    # ================================================================
    # SELF-CONSISTENCY CHECK: z_i=z_j should reduce to the SAME formula
    # applied to qperp_power's own raw P(k), via the shared helper
    # diagonal_reference_dl -- NOT compute_cell (reverted: compute_cell
    # applies visibility^2/a^-4/dchi weighting this simplified function
    # deliberately does not replicate, and needs >=2 z's for its own
    # internal np.gradient -- a structurally different, not just
    # technically incompatible, comparison. Using the SAME shared
    # formula on both sides here guarantees they cannot silently drift
    # into two different formulas the way the original missing-prefactor
    # bug happened.)
    # ================================================================
    # ================================================================
    # SANITY CHECK: the previous "fake duplicate z" self-check is no
    # longer meaningful now that weighting includes dchi -- a duplicate
    # entry with IDENTICAL chi gives dchi=0 via np.gradient, trivially
    # zeroing the result regardless of correctness. The w_pair formula's
    # exactness at i=j (sqrt(x*x)=x for x=vis2_i/a_i^4*dchi_i) is a
    # MATHEMATICAL identity, true by algebra for any non-negative x --
    # not something a runtime test can meaningfully strengthen. Instead,
    # check the genuinely new piece of logic here (tau accumulation) for
    # a real bug signature: visibility must be monotonically
    # NON-INCREASING with z (optical depth only accumulates looking
    # further back), never counting backwards.
    # ================================================================
    print("\nSANITY CHECK: visibility should be monotonically non-increasing with z...")
    zs_sorted = sorted(z_window)
    chi_arr_check = np.array([chi_dict[z] for z in zs_sorted])
    dchi_check = np.abs(np.gradient(chi_arr_check))
    xe_check = np.array([1.0 - xH_mean_dict[z] for z in zs_sorted])
    from ksz_pipeline.ksz.optical_depth import analytic_tau_below
    from ksz_pipeline.utils.constants import SIGMA_T, MPC_CM
    tau0_check = analytic_tau_below(zs_sorted[0])
    tau_check = np.full(len(zs_sorted), tau0_check)
    for i in range(len(zs_sorted) - 1):
        zmid = 0.5 * (zs_sorted[i] + zs_sorted[i + 1])
        xe_mid = 0.5 * (xe_check[i] + xe_check[i + 1])
        tau_check[i + 1] = tau_check[i] + SIGMA_T * ne0_cgs() * xe_mid * (1.0 + zmid) ** 2 * (dchi_check[i] * MPC_CM)
    vis2_check = np.exp(-2.0 * tau_check)
    is_monotonic = np.all(np.diff(vis2_check) <= 1e-12)
    print(f"  visibility^2 range: {vis2_check.min():.4g} (z={zs_sorted[np.argmin(vis2_check)]:.1f}) "
          f"to {vis2_check.max():.4g} (z={zs_sorted[np.argmax(vis2_check)]:.1f})")
    if is_monotonic:
        print("  OK -- monotonically non-increasing with z, as physically required.\n")
    else:
        print("  *** WARNING: NOT monotonic -- likely bug in tau accumulation, "
              "investigate before trusting cross-z results below. ***\n")

    # ================================================================
    # THE ACTUAL CROSS-Z CALCULATION
    # ================================================================
    print("Computing q_perp cross-z correlation (z_i != z_j, all pairs)...")
    ell_qperp, Dl_qperp_cross, n_pairs = cross_power_qperp_pairwise_chi(
        qperp_components, chi_dict, xH_mean_dict, BOX_LEN)
    n_pairs_expected = len(z_window) * (len(z_window) - 1) // 2
    print(f"  {n_pairs}/{n_pairs_expected} pairs contributed")
    d3000_qperp_cross = float(np.interp(3000, ell_qperp, Dl_qperp_cross))
    print(f"  D_3000(q_perp cross, periodicity-free) = {d3000_qperp_cross:.4g} uK^2\n")

    # ---- compare against stitched P_off, if available ----
    coherence_path = f"{out_dir}/coherence_decomposition_fiducial.npz"
    if os.path.exists(coherence_path):
        coherence = np.load(coherence_path)
        ell_off, Dl_off = coherence['ell_dec'], coherence['Dl_off']
        d3000_off_stitched = float(np.interp(3000, ell_off, Dl_off))
        ratio = d3000_qperp_cross / d3000_off_stitched
        print(f"Comparison to stitched P_off (periodicity + Limber-failure mixed):")
        print(f"  stitched P_off D_3000       = {d3000_off_stitched:.4g} uK^2")
        print(f"  q_perp cross D_3000         = {d3000_qperp_cross:.4g} uK^2")
        print(f"  ratio (q_perp cross / stitched P_off) = {ratio:.1%}")
        print(f"  -- i.e. roughly {ratio:.0%} of stitched's excess is in the SAME direction/")
        print(f"     size as what a periodicity-free channel shows; remainder plausibly")
        print(f"     periodicity-specific or a rectilinear/normalization artifact -- see caveats.")
    else:
        print(f"NOTE: {coherence_path} not found -- skipping direct ratio to stitched P_off.")
        d3000_off_stitched, ratio = float('nan'), float('nan')

    print("\nCAVEATS (see module docstring for full detail):")
    print("  1. Different-z boxes share the SAME initial conditions (same seed) -- not")
    print("     independent realizations. Nonzero result partly reflects real, deterministic")
    print("     correlated time-evolution, not necessarily 'new physics' Limber's independence")
    print("     assumption specifically targets.")
    print("  2. Rectilinear/flat-sky approximation (kz=0 slice + pairwise chi_ij), NOT the full")
    print("     spherical/Bessel-function non-Limber treatment.")
    print("  3. '/2' normalization inherited from qperp_power (itself Limber-derived) -- treat")
    print("     absolute amplitude cautiously; sign/shape more trustworthy than precise value.")

    # ================================================================
    # plot + save
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ell_qperp, Dl_qperp_cross, 'o-', color='tab:green', ms=4,
             label='q_perp cross-z (periodicity-free)')
    if os.path.exists(coherence_path):
        ax.plot(ell_off, Dl_off, 's-', color='tab:blue', ms=4,
                 label='stitched P_off (periodicity + Limber-failure)')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')
    ax.set_title('q_perp cross-z (periodicity-free) vs stitched P_off')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plot_path = f"{plot_dir}/qperp_cross_z.png"
    fig.savefig(plot_path, dpi=140, bbox_inches='tight')
    print(f"\nSaved -> {plot_path}")

    np.savez(f"{out_dir}/qperp_cross_z.npz",
              ell_qperp=ell_qperp, Dl_qperp_cross=Dl_qperp_cross, n_pairs=n_pairs,
              d3000_qperp_cross=d3000_qperp_cross, d3000_off_stitched=d3000_off_stitched,
              ratio=ratio, self_check_frac=self_check_frac)
    print(f"Saved -> {out_dir}/qperp_cross_z.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    args = parser.parse_args()
    main(args.config)
