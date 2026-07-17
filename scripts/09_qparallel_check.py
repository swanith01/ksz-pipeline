#!/usr/bin/env python
"""
Script 09: measure P_q_parallel(k) per redshift and Limber-project it,
testing whether q_perp + q_parallel narrows the low-ell gap between
coeval-direct (q_perp-only Limber, by construction) and stitched (direct
real-space map, no component split -- closer to a reference figure's
"2-D Maps" curve, which itself exceeds q_perp+q_parallel Limber at low
ell, so full closure isn't expected -- partial narrowing is the honest
test here).

Deliberately separate from scripts 02/03 (frozen/trusted). Reuses
run_coeval_fields (cache-hit, same pattern as script 08) and
limber.compute_cell UNCHANGED -- results_par below is shaped identically
to results_qperp so the existing, already-validated Limber integral
applies with no modification.

Produces
--------
data/products/qparallel_power.npz            -- k, Pqpar, Pstd per redshift
data/products/ksz_Dl_coeval_qpar.npz          -- Limber D_ell from
                                                  q_parallel ALONE
data/products/ksz_Dl_coeval_qperp_plus_qpar.npz -- q_perp + q_parallel,
                                                  summed at the P(k,z)
                                                  level BEFORE the Limber
                                                  integral

Usage
-----
    python scripts/09_qparallel_check.py --config configs/fiducial.yaml
"""
import argparse, os, pickle
import numpy as np
import yaml

from ksz_pipeline.coeval.fields   import run_coeval_fields
from ksz_pipeline.coeval.momentum import qparallel_power
from ksz_pipeline.coeval.limber   import compute_cell


def main(config_path, force=False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sim_cfg    = cfg['21cmfast']
    coeval_cfg = cfg['coeval_ksz']
    cache_dir  = cfg['data']['cache_dir']
    HII_DIM    = sim_cfg['HII_DIM_coeval']
    BOX_LEN    = sim_cfg['BOX_LEN']
    ZS         = coeval_cfg['z_snapshots']
    out_dir    = cfg['data']['output_dir'].rstrip('/')
    os.makedirs(out_dir, exist_ok=True)

    pkl_path = os.path.join(cache_dir, "qparallel_power.pkl")
    if not force and os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            results_par = pickle.load(f)
        print(f"Loaded P_qparallel cache: {len(results_par)} redshifts")
    else:
        results_par = {}

    missing = [z for z in ZS if z not in results_par]
    if missing:
        print(f"Computing P_qparallel for {len(missing)} redshifts "
              f"(reusing cached boxes via run_coeval_fields)...")
        for z in missing:
            print(f"  z={z:.1f}...", end=' ', flush=True)
            delta, xH, vx, vy, vz = run_coeval_fields(
                z, HII_DIM, BOX_LEN, cache_dir, N_THREADS=sim_cfg['N_THREADS'],
                random_seed=sim_cfg['random_seed'])
            k_par, P_par, P_par_std = qparallel_power(delta, xH, vx, vy, vz, BOX_LEN)
            # NOTE: key is 'Pqperp' (not 'Pqpar') deliberately -- compute_cell
            # reads entry['Pqperp'] internally, and reusing that exact key
            # lets us call compute_cell UNCHANGED rather than touching
            # validated code. The array actually holds P_q_parallel here --
            # don't be misled by the key name, it's a reuse-of-interface
            # choice, not a labeling bug.
            results_par[z] = dict(k=k_par, Pqperp=P_par, Pstd=P_par_std,
                                   xH_mean=float(xH.mean()))
            print(f"<xH>={xH.mean():.3f}")
        with open(pkl_path, 'wb') as f:
            pickle.dump(results_par, f)
        print(f"Cache saved -> {pkl_path}")

    print("Limber-projecting q_parallel alone...")
    (ells_par, D_ell_par, sigma_par, C_par, sigma_C_par,
     (ZS_par, tau_par), (_, xe_par)) = compute_cell(results_par)
    np.savez(f"{out_dir}/ksz_Dl_coeval_qpar.npz",
             ell=ells_par, Dl=D_ell_par, sigma_Dl=sigma_par)
    d3000_par = float(np.interp(3000, ells_par, D_ell_par))
    print(f"  D_3000 (q_parallel alone) = {d3000_par:.4f} uK^2")

    # -- q_perp + q_parallel, summed at the P(k,z) level BEFORE Limber --
    qperp_pkl = os.path.join(cache_dir, "qperp_power.pkl")
    if not os.path.exists(qperp_pkl):
        print(f"WARNING: {qperp_pkl} not found -- cannot build the "
              f"q_perp+q_parallel sum (script 02 should have created this "
              f"for the fiducial config already; check cache_dir).")
        return
    with open(qperp_pkl, 'rb') as f:
        results_qperp = pickle.load(f)

    results_sum = {}
    for z in ZS:
        if z not in results_qperp or z not in results_par:
            continue
        # Both were binned identically (same nbins formula, same
        # BOX_LEN/HII_DIM), so k arrays SHOULD match -- confirm rather
        # than assume before summing, since a silent shape/binning
        # mismatch here would be exactly the class of bug this pipeline
        # has already hit before.
        k_perp = results_qperp[z]['k']
        k_par  = results_par[z]['k']
        if len(k_perp) != len(k_par) or not np.allclose(k_perp, k_par, rtol=1e-6):
            print(f"  z={z:.1f}: k-binning MISMATCH between qperp and "
                  f"qparallel results -- skipping sum for this redshift")
            continue
        P_sum = results_qperp[z]['Pqperp'] + results_par[z]['Pqperp']  # 'Pqperp' key holds P_par here, see note above
        results_sum[z] = dict(k=k_perp, Pqperp=P_sum,
                               Pstd=results_qperp[z]['Pstd'],  # approximate -- not a rigorous propagated error
                               xH_mean=results_qperp[z]['xH_mean'])

    if results_sum:
        print("Limber-projecting q_perp + q_parallel (summed before Limber)...")
        (ells_sum, D_ell_sum, sigma_sum, C_sum, sigma_C_sum,
         (ZS_sum, tau_sum), (_, xe_sum)) = compute_cell(results_sum)
        np.savez(f"{out_dir}/ksz_Dl_coeval_qperp_plus_qpar.npz",
                 ell=ells_sum, Dl=D_ell_sum, sigma_Dl=sigma_sum)
        d3000_sum = float(np.interp(3000, ells_sum, D_ell_sum))
        print(f"  D_3000 (q_perp + q_parallel) = {d3000_sum:.4f} uK^2")
        print(f"  compare: coeval-direct (q_perp only) = 1.8441 uK^2")
        print(f"  compare: stitched (real-space map)    = 0.80689 uK^2")
    else:
        print("No overlapping redshifts between qperp and qparallel caches "
              "-- cannot build the sum.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()
    main(args.config, force=args.force)
