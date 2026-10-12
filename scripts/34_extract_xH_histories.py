#!/usr/bin/env python
"""
Script 34: pull the reionisation history <xH>(z) of every convergence-sweep
configuration out of the coeval sweeps' own pickle caches, into one small
file that can be committed and plotted anywhere.

Why: scripts 04/06 save only D_ell curves and D_3000 into data/products/
(convergence_*_full.npz). The per-snapshot mean neutral fraction of each
configuration was stored only in the pickles
    <cache_dir>/convergence_boxsize_coeval/qperp_<tag>.pkl   (box-size sweep)
    <cache_dir>/qperp_<tag>.pkl                              (resolution sweep)
    <cache_dir>/qperp_dz_fiducial.pkl                        (dz sweep)
(see convergence/coeval_sweep.py: results[z]['xH_mean']). This script reads
those pickles and writes data/products/convergence_xH_histories.npz. No
py21cmfast, no simulation, seconds to run; safe on a login node. It reads
the pickles only, never the large .h5 boxes.

Keys written:  <sweep>__<tag>__z   and   <sweep>__<tag>__xH
  sweep in {boxsize, resolution, dz}; tag e.g. box400_N64, res256, dz_x2.
Missing pickles are reported and skipped (that configuration then has no
history panel entry; its boxes would have to be regenerated).

Usage:  python scripts/34_extract_xH_histories.py --config configs/fiducial.yaml
"""
import argparse
import os
import pickle

import numpy as np
import yaml


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _history(results, zs):
    zs = [z for z in zs if z in results]
    if not zs:
        return None
    z = np.array(sorted(zs), dtype=float)
    return z, np.array([results[zz]['xH_mean'] for zz in z], dtype=float)


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    conv = cfg['convergence']
    cache = cfg['data']['cache_dir']
    out_dir = cfg['data']['output_dir'].rstrip('/')
    z_fine = sorted(cfg['coeval_ksz']['z_snapshots'], reverse=True)
    out, missing = {}, []

    # box-size sweep: tag box{L}_N{N}, L as written in the config (int)
    dx = conv['fixed_dx_mpc']
    for L in conv['box_lens']:
        N = max(4, int(round(L / dx)))
        tag = f"box{L}_N{N}"
        res = _load(os.path.join(cache, 'convergence_boxsize_coeval', f"qperp_{tag}.pkl"))
        h = _history(res, z_fine) if res else None
        if h is None:
            missing.append(('boxsize', tag)); continue
        out[f"boxsize__{tag}__z"], out[f"boxsize__{tag}__xH"] = h

    # resolution sweep: lives in the MAIN cache dir
    for N in conv['hii_dims']:
        tag = f"res{N}"
        res = _load(os.path.join(cache, f"qperp_{tag}.pkl"))
        h = _history(res, z_fine) if res else None
        if h is None:
            missing.append(('resolution', tag)); continue
        out[f"resolution__{tag}__z"], out[f"resolution__{tag}__xH"] = h

    # dz sweep: one pickle, subsets of the fine grid
    res = _load(os.path.join(cache, "qperp_dz_fiducial.pkl"))
    for m in conv['dz_multiples']:
        tag = f"dz_x{m}"
        h = _history(res, z_fine[::m]) if res else None
        if h is None:
            missing.append(('dz', tag)); continue
        out[f"dz__{tag}__z"], out[f"dz__{tag}__xH"] = h

    path = f"{out_dir}/convergence_xH_histories.npz"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(path, **out)
    print(f"Saved {len(out)//2} histories -> {path}")
    for k in sorted(out):
        if k.endswith('__xH'):
            x = out[k]; z = out[k.replace('__xH', '__z')]
            zm = float(np.interp(0.5, x, z))        # z ascending, xH rises with z
            print(f"  {k[:-4]:<28s} n_z={len(z):2d}  z(xH=0.5)~{zm:5.2f}")
    if missing:
        print("\nMISSING (no pickle found, no history for these):")
        for s, t in missing:
            print(f"  {s}: {t}")
        print("Check:  ls", cache, " and ", os.path.join(cache, 'convergence_boxsize_coeval'))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fiducial.yaml")
    main(ap.parse_args().config)
