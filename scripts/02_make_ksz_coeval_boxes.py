#!/usr/bin/env python
"""
Script 02: Run coeval boxes -> P_{q_perp}(k) -> D_ell via Limber.

Produces
--------
data/products/ksz_Dl_coeval.npz          -- ell, Dl, sigma_Dl [uK^2]
data/products/ksz_Dl_coeval_georgiev.npz -- same, but built from the
                                             Eq.10 Pee/Pvv/Pev reconstruction
                                             instead of the direct measurement
data/products/coeval_reion.npz           -- ZS_asc, xe, tau
data/products/qperp_power.npz            -- k, Pqperp, Pstd per redshift
data/products/pee_pvv_pev.npz            -- k, Pee, Pvv, Pev, and the
                                             direct/reconstructed ratio,
                                             per redshift

Heavy runs (512^3) -> submit via jobs/array_scan.pbs on cluster,
copy qperp_power.pkl back to desktop, then run Limber step locally.

Usage
-----
    python scripts/02_make_ksz_coeval_boxes.py
    python scripts/02_make_ksz_coeval_boxes.py --force
    python scripts/02_make_ksz_coeval_boxes.py --config configs/local.yaml
"""

import argparse, os, pickle
import yaml
import numpy as np
import py21cmfast as p21c

from ksz_pipeline.coeval.velocity     import velocity_conversion_factor
from ksz_pipeline.coeval.momentum     import qperp_power
from ksz_pipeline.coeval.limber       import compute_cell
from ksz_pipeline.coeval.pee_pvv_pev  import (measure_pee_pvv_pev,
                                               compare_direct_vs_reconstructed)


def run_coeval_fields(z, HII_DIM, BOX_LEN, cache_dir):
    coeval = p21c.run_coeval(
        redshift    = float(z),
        user_params = {"HII_DIM": int(HII_DIM), "BOX_LEN": float(BOX_LEN)},
        write       = False,
        direc       = cache_dir,
    )
    fac   = velocity_conversion_factor(z)
    delta = coeval.density
    xH    = coeval.xH_box
    vx    = coeval.lowres_vx * fac * 1e5
    vy    = coeval.lowres_vy * fac * 1e5
    vz    = coeval.lowres_vz * fac * 1e5
    return delta, xH, vx, vy, vz


def main(config_path, force=False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sim_cfg    = cfg['21cmfast']
    coeval_cfg = cfg['coeval_ksz']
    cache_dir  = cfg['data']['cache_dir']
    HII_DIM    = sim_cfg['HII_DIM_coeval']
    BOX_LEN    = sim_cfg['BOX_LEN']
    ZS         = coeval_cfg['z_snapshots']

    os.makedirs("data/products", exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    pkl_path = os.path.join(cache_dir, "qperp_power.pkl")
    if not force and os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            results_qperp = pickle.load(f)
        print(f"Loaded P_{{q_perp}} cache: {len(results_qperp)} redshifts")
    else:
        results_qperp = {}

    missing = [z for z in ZS if z not in results_qperp]
    if missing:
        print(f"Computing P_{{q_perp}} for {len(missing)} redshifts...")
        for z in missing:
            print(f"  z={z:.1f}...", end=' ', flush=True)
            delta, xH, vx, vy, vz = run_coeval_fields(
                z, HII_DIM, BOX_LEN, cache_dir)
            k_q, P_q, P_std = qperp_power(delta, xH, vx, vy, vz, BOX_LEN)
            k_pee, Pee_z, Pvv_z, Pev_z = measure_pee_pvv_pev(
                delta, xH, vx, vy, vz, BOX_LEN)
            results_qperp[z] = dict(k=k_q, Pqperp=P_q, Pstd=P_std,
                                    xH_mean=float(xH.mean()),
                                    k_pee=k_pee, Pee=Pee_z, Pvv=Pvv_z, Pev=Pev_z)
            print(f"<xH>={xH.mean():.3f}")
        with open(pkl_path, 'wb') as f:
            pickle.dump(results_qperp, f)
        print(f"Cache saved -> {pkl_path}")
    elif any('k_pee' not in results_qperp[z] for z in ZS):
        print("NOTE: cached qperp_power.pkl predates the Pee/Pvv/Pev "
              "measurement -- the Georgiev comparison below will skip "
              "redshifts missing it. Re-run with --force to fill it in.")

    print("Running Limber projection...")
    (ells, D_ell, sigma_D, C_ell, sigma_C,
     (ZS_asc, tau), (_, xe)) = compute_cell(results_qperp)

    print("Building Georgiev Eq.10 reconstruction from Pee/Pvv/Pev "
          "and comparing against the direct measurement...")
    results_qperp_georgiev = {}
    georgiev_ratios = {}
    for z in results_qperp:
        entry = results_qperp[z]
        if 'k_pee' not in entry:
            print(f"  z={z:.1f}: no Pee/Pvv/Pev cached -- skipping "
                  f"(re-run with --force to fill in)")
            continue
        P_recon, ratio = compare_direct_vs_reconstructed(
            entry['k'], entry['Pqperp'],
            entry['k_pee'], entry['Pee'], entry['Pvv'], entry['Pev'],
            verbose=False)
        print(f"  z={z:.1f}: direct/reconstructed ratio spans "
              f"[{ratio.min():.2f}, {ratio.max():.2f}], mean {ratio.mean():.2f}"
              f"  (expect roughly flat ~1 if this matches Lopez/D'Aloisio/Cain "
              f"2025's ~10-20% non-Gaussianity undershoot; a flat ratio near "
              f"a clean value like 2 instead means a normalization mismatch "
              f"-- see pee_pvv_pev.py's module docstring)")
        georgiev_ratios[z] = ratio
        # Pstd left at zero: no formal error propagation through the Eq.10
        # convolution yet -- this curve's error bars are not meaningful,
        # only its central value, until that's added.
        results_qperp_georgiev[z] = dict(k=entry['k'], Pqperp=P_recon,
                                          Pstd=np.zeros_like(P_recon),
                                          xH_mean=entry['xH_mean'])

    if results_qperp_georgiev:
        (ells_g, D_ell_g, sigma_D_g, C_ell_g, sigma_C_g,
         (ZS_asc_g, tau_g), (_, xe_g)) = compute_cell(results_qperp_georgiev)
        np.savez("data/products/ksz_Dl_coeval_georgiev.npz",
                 ell=ells_g, Dl=D_ell_g, sigma_Dl=sigma_D_g)
        print(f"  Saved Georgiev-reconstructed D_ell -> "
              f"data/products/ksz_Dl_coeval_georgiev.npz")
    else:
        print("  No redshifts had Pee/Pvv/Pev cached -- skipped Georgiev "
              "D_ell entirely. Re-run with --force.")

    np.savez("data/products/ksz_Dl_coeval.npz",
             ell=ells, Dl=D_ell, sigma_Dl=sigma_D)
    np.savez("data/products/coeval_reion.npz",
             z=ZS_asc, xe=xe, tau=tau)

    zs_sorted = sorted(results_qperp.keys())
    save_dict = {"z": np.array(zs_sorted)}
    for i, z in enumerate(zs_sorted):
        save_dict[f"k_{i}"]    = results_qperp[z]['k']
        save_dict[f"Pq_{i}"]   = results_qperp[z]['Pqperp']
        save_dict[f"Pstd_{i}"] = results_qperp[z]['Pstd']
    np.savez("data/products/qperp_power.npz", **save_dict)

    pee_dict = {"z": np.array(zs_sorted)}
    for i, z in enumerate(zs_sorted):
        entry = results_qperp[z]
        if 'k_pee' not in entry:
            continue
        pee_dict[f"k_pee_{i}"] = entry['k_pee']
        pee_dict[f"Pee_{i}"]   = entry['Pee']
        pee_dict[f"Pvv_{i}"]   = entry['Pvv']
        pee_dict[f"Pev_{i}"]   = entry['Pev']
        if z in georgiev_ratios:
            pee_dict[f"ratio_{i}"] = georgiev_ratios[z]
    np.savez("data/products/pee_pvv_pev.npz", **pee_dict)

    print(f"Saved to data/products/")
    print(f"  D_3000 = {float(np.interp(3000, ells, D_ell)):.4f} uK^2"
          f"  (Reichardt+2021: 1.1 +1.0/-0.7 uK^2)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()
    main(args.config, force=args.force)
