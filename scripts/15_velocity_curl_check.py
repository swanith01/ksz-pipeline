#!/usr/bin/env python
"""
Script 15: velocity curl-fraction diagnostic.

Checks whether the raw coeval velocity field (vx, vy, vz -- BEFORE any
(1+delta)*chi momentum weighting) has any genuine rotational (perp-to-k)
power, or whether it is exactly curl-free as expected for a pure
Zel'dovich-approximation displacement field.

Motivation: P_qparallel (momentum.py) stays comparable to P_qperp much
further into small scales (high ell) than the Q_ma+2018-style reference
figure's q_parallel mode does. Leading hypothesis: 21cmFAST's Zel'dovich
velocities are curl-free by construction (v = grad(phi)), so any q_perp
power at all is coming purely from momentum.py's (1+delta)*chi weighting
convolving an irrotational v -- not from real vorticity the way an
N-body-sourced reference field would have. This script tests that
directly on v itself, independent of the momentum weighting.

Usage
-----
    python scripts/15_velocity_curl_check.py --config configs/fiducial.yaml [--z ZVAL]
"""
import argparse
import numpy as np
import yaml

from ksz_pipeline.coeval.fields import run_coeval_fields


def velocity_curl_fraction(vx, vy, vz, BOX_LEN, nbins=20):
    """
    Fraction of the velocity field's OWN Fourier power that is
    rotational (perp-to-k) vs irrotational (parallel-to-k), by k bin.
    If this comes out ~0 (floating-point noise) at every k, it confirms
    v is exactly curl-free (as expected for pure Zel'dovich) -- meaning
    ANY q_perp signal in qperp_power's output is coming purely from the
    (1+delta)*chi weighting's convolution, not from real vorticity in v.
    If instead frac_perp is non-negligible (e.g. >1e-4) at high k, that
    points to a bug introducing spurious curl somewhere in the
    displacement -> velocity conversion (velocity.py), not a physical
    limitation.

    Returns
    -------
    k_bin_edges : ndarray
    frac_perp   : ndarray  perp/(perp+par) power fraction per bin
    """
    N = vx.shape[0]; L = float(BOX_LEN); d = L / N
    kfreq = np.fft.fftfreq(N, d=d) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k_mag = np.sqrt(k2)
    k2_safe = np.where(k2 == 0.0, np.inf, k2)

    Vx = np.fft.fftn(vx) * d**3
    Vy = np.fft.fftn(vy) * d**3
    Vz = np.fft.fftn(vz) * d**3

    kdotV_k2 = (Vx * kx + Vy * ky + Vz * kz) / k2_safe
    Vx_par, Vy_par, Vz_par = kdotV_k2 * kx, kdotV_k2 * ky, kdotV_k2 * kz
    Vx_perp, Vy_perp, Vz_perp = Vx - Vx_par, Vy - Vy_par, Vz - Vz_par

    P_par  = (np.abs(Vx_par)**2  + np.abs(Vy_par)**2  + np.abs(Vz_par)**2 ).ravel()
    P_perp = (np.abs(Vx_perp)**2 + np.abs(Vy_perp)**2 + np.abs(Vz_perp)**2).ravel()
    k_flat = k_mag.ravel()

    bins = np.geomspace(k_flat[k_flat > 0].min(), k_flat.max(), nbins)
    digit = np.digitize(k_flat, bins)

    frac_perp = np.full(len(bins) - 1, np.nan)
    print(f"{'k range [Mpc^-1]':28s} {'perp/(perp+par) fraction':>28s}")
    for i in range(1, len(bins)):
        mask = digit == i
        if not np.any(mask):
            continue
        par_i, perp_i = P_par[mask].sum(), P_perp[mask].sum()
        frac_perp[i - 1] = perp_i / (par_i + perp_i)
        print(f"{bins[i-1]:10.4f} - {bins[i]:10.4f}{'':>10s}{frac_perp[i-1]:.3e}")

    return bins, frac_perp


def main(config_path, z_override=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sim_cfg, coeval_cfg = cfg['21cmfast'], cfg['coeval_ksz']
    cache_dir = cfg['data']['cache_dir']
    HII_DIM, BOX_LEN = sim_cfg['HII_DIM_coeval'], sim_cfg['BOX_LEN']
    out_dir = cfg['data']['output_dir'].rstrip('/')

    ZS = sorted(coeval_cfg['z_snapshots'])
    z_rep = z_override if z_override is not None else min(ZS, key=lambda z: abs(z - 7.0))
    print(f"Using z={z_rep} (representative snapshot, cache-hit expected if already generated)")

    delta_r, xH_r, vx_r, vy_r, vz_r = run_coeval_fields(
        z_rep, HII_DIM, BOX_LEN, cache_dir,
        N_THREADS=sim_cfg['N_THREADS'], random_seed=sim_cfg['random_seed'])

    print(f"\n=== Velocity curl-fraction check at z={z_rep} ===")
    print("(raw v field, BEFORE (1+delta)*chi momentum weighting)\n")
    bins, frac_perp = velocity_curl_fraction(vx_r, vy_r, vz_r, BOX_LEN)

    max_frac = np.nanmax(frac_perp)
    print(f"\nMax perp fraction across all k bins: {max_frac:.3e}")
    if max_frac < 1e-6:
        print("INTERPRETATION: v is curl-free to floating-point precision -- "
              "consistent with pure Zel'dovich. Any q_perp power downstream in "
              "qperp_power() is coming from the (1+delta)*chi weighting's "
              "convolution, not real vorticity in v. This supports the "
              "hypothesis that P_qparallel's slow decay (vs. the N-body-sourced "
              "reference figure) is a physical limitation of Zel'dovich "
              "velocities, not a bug.")
    else:
        print("INTERPRETATION: non-negligible rotational power found in v itself -- "
              "this is NOT expected for pure Zel'dovich. Worth checking "
              "velocity.py's displacement->velocity conversion for spurious "
              "curl (e.g. inconsistent finite differencing, grid aliasing) "
              "before concluding this is a physical limitation.")

    np.savez(f"{out_dir}/velocity_curl_check.npz",
              z_rep=z_rep, k_bin_edges=bins, frac_perp=frac_perp, max_frac=max_frac)
    print(f"\nSaved -> {out_dir}/velocity_curl_check.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    parser.add_argument("--z", type=float, default=None,
                         help="Override representative redshift (default: nearest z_snapshot to 7.0)")
    args = parser.parse_args()
    main(args.config, z_override=args.z)
