#!/usr/bin/env python
"""
Script 38: Do native py21cmfast v4 lightcones (rectilinear AND angular) repeat
with the box length along the line of sight?

No kSZ here -- only the lightcone fields. For each field (density, velocity_z)
we compute the LOS autocorrelation of the per-slice-normalised field,

    rho(lag) = < f(x, chi) * f(x, chi + lag) >_{pixels, chi}

where f is the field minus its slice mean, divided by its slice rms. A box that
is tiled periodically gives bumps at lag = L, 2L, 3L, ... (L = BOX_LEN).
A lightcone with no periodicity decays smoothly with lag.

For the angular lightcone we also split pixels by off-axis angle (inner vs
outer ring), because a ray at angle theta is displaced by ~ L*tan(theta)
transverse to itself after one box length and so should decorrelate more.

Reuses ksz_pipeline.ksz.lightcone_v4 (same inputs object for both lightcones,
so they run on the same IC / same cache).

Usage (swarm, env p21c_v3 is py21cmfast 3.3.1 -- THIS NEEDS A v4 ENV, the same
one used for scripts/07_v4_angular_vs_rectilinear.py):
    python scripts/38_native_periodicity_check.py --HII_DIM 128 --BOX_LEN 800 \
        --z_min 5.0 --z_max 12.0 --cache_dir data/cache_v4_periodicity
Outputs: data/products/native_periodicity_check.npz and a PNG next to it.
"""
import argparse
import os

import numpy as np

from ksz_pipeline.ksz.lightcone_v4 import (build_inputs, build_node_redshifts,
                                            run_rectilinear, run_angular)


def uniform_along_chi(f, chi):
    """Resample (Nx,Ny,Nz) onto a uniform chi grid if the spacing is not uniform."""
    d = np.diff(chi)
    if np.max(np.abs(d - np.median(d))) < 0.01 * abs(np.median(d)):
        return f, chi, abs(float(np.median(d)))
    chi_u = np.linspace(chi.min(), chi.max(), len(chi))
    out = np.empty_like(f)
    flat = f.reshape(-1, f.shape[-1])
    order = np.argsort(chi)
    for i in range(flat.shape[0]):
        out.reshape(-1, f.shape[-1])[i] = np.interp(chi_u, chi[order], flat[i][order])
    return out, chi_u, float(chi_u[1] - chi_u[0])


def los_autocorr(f, max_lag, mask=None):
    """f: (Npix, Nz), already per-slice normalised. Returns rho[0..max_lag]."""
    if mask is not None:
        f = f[mask]
    nz = f.shape[1]
    rho = np.full(max_lag + 1, np.nan)
    for lag in range(max_lag + 1):
        if lag >= nz:
            break
        rho[lag] = np.mean(f[:, :nz - lag] * f[:, lag:])
    return rho


def normalise(f):
    """Per-slice: subtract the pixel mean, divide by the pixel rms."""
    f = f - f.mean(axis=0, keepdims=True)
    s = f.std(axis=0, keepdims=True)
    s[s == 0] = 1.0
    return f / s


def analyse(label, fields, chi_all, z_all, BOX_LEN, HII_DIM, z_lo, z_hi, max_boxes):
    sel = (z_all >= z_lo) & (z_all <= z_hi)
    chi = np.asarray(chi_all)[sel]
    out = {}
    for name in ("density", "velocity_z"):
        f = np.asarray(fields[name], dtype=np.float32)[..., sel]
        f, chi_u, dchi = uniform_along_chi(f, chi)
        npix = f.shape[0] * f.shape[1]
        f2 = normalise(f.reshape(npix, -1))
        max_lag = min(int(max_boxes * BOX_LEN / dchi), f2.shape[1] - 2)
        lags = np.arange(max_lag + 1) * dchi
        rho = los_autocorr(f2, max_lag)
        # inner vs outer ring of pixels (off-axis angle proxy)
        ix, iy = np.meshgrid(np.arange(HII_DIM), np.arange(HII_DIM), indexing="ij")
        r = np.hypot(ix - (HII_DIM - 1) / 2, iy - (HII_DIM - 1) / 2).ravel()
        inner = r <= np.percentile(r, 25)
        outer = r >= np.percentile(r, 75)
        rho_in = los_autocorr(f2, max_lag, inner)
        rho_out = los_autocorr(f2, max_lag, outer)
        out[name] = dict(lags=lags, rho=rho, rho_inner=rho_in, rho_outer=rho_out, dchi=dchi)
        # summary: peak near L vs the dips at 0.5L and 1.5L
        def peak(rr, x):
            k = int(round(x * BOX_LEN / dchi))
            w = slice(max(k - 2, 0), min(k + 3, len(rr)))
            return float(np.nanmax(rr[w]))
        def at(rr, x):
            return float(rr[int(round(x * BOX_LEN / dchi))])
        print(f"[{label}] {name:10s}  rho(L)={peak(rho,1):+.3f}  rho(0.5L)={at(rho,0.5):+.3f}  "
              f"rho(1.5L)={at(rho,1.5):+.3f} | inner rho(L)={peak(rho_in,1):+.3f}  "
              f"outer rho(L)={peak(rho_out,1):+.3f}")
    return out


def main(a):
    node_z = build_node_redshifts(a.z_min, a.z_max, n_nodes=a.n_nodes)
    inputs = build_inputs(a.random_seed, a.HII_DIM, a.BOX_LEN, node_z,
                          HII_EFF_FACTOR=a.HII_EFF_FACTOR, N_THREADS=a.n_threads)
    quantities = ("density", "velocity_z")
    res = {}
    for label in ("rectilinear", "angular"):
        if label == "rectilinear":
            lc, fields = run_rectilinear(inputs, a.z_min, a.z_max, a.cache_dir,
                                         quantities=quantities)
        else:
            lc, fields = run_angular(inputs, a.z_min, a.z_min, a.z_max, a.cache_dir,
                                     quantities=quantities)
        z_all = np.asarray(lc.lightcone_redshifts)
        chi_all = np.asarray(lc.lightcone_distances)
        res[label] = analyse(label, fields, chi_all, z_all, a.BOX_LEN, a.HII_DIM,
                             a.z_min, a.z_max, a.max_boxes)

    os.makedirs(a.out_dir, exist_ok=True)
    flat = {}
    for lab, d in res.items():
        for name, v in d.items():
            for k, arr in v.items():
                flat[f"{lab}__{name}__{k}"] = np.asarray(arr)
    np.savez(os.path.join(a.out_dir, "native_periodicity_check.npz"), **flat)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, name in enumerate(("density", "velocity_z")):
        for i, lab in enumerate(("rectilinear", "angular")):
            ax = axs[i, j]
            v = res[lab][name]
            x = v["lags"] / a.BOX_LEN
            ax.plot(x, v["rho"], color="#1f77b4", lw=2, label="all pixels")
            ax.plot(x, v["rho_inner"], color="#2ca02c", lw=1.3, label="inner 25% (near axis)")
            ax.plot(x, v["rho_outer"], color="#d62728", lw=1.3, label="outer 25% (off axis)")
            for m in range(1, int(a.max_boxes) + 1):
                ax.axvline(m, color="0.6", ls=":", lw=1)
            ax.set_title(f"{lab}: {name}")
            ax.set_ylabel(r"$\rho(\Delta\chi)$")
            if i == 1:
                ax.set_xlabel(r"$\Delta\chi / L_{\rm box}$")
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "native_periodicity_check.png"), dpi=160)
    print("wrote", os.path.join(a.out_dir, "native_periodicity_check.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--HII_DIM", type=int, default=128)
    p.add_argument("--BOX_LEN", type=float, default=800.0)
    p.add_argument("--z_min", type=float, default=5.0)
    p.add_argument("--z_max", type=float, default=12.0)
    p.add_argument("--random_seed", type=int, default=37)
    p.add_argument("--HII_EFF_FACTOR", type=float, default=30.0)
    p.add_argument("--cache_dir", default="data/cache_v4_periodicity")
    p.add_argument("--out_dir", default="data/products")
    p.add_argument("--n_threads", type=int, default=8)
    p.add_argument("--n_nodes", type=int, default=30)
    p.add_argument("--max_boxes", type=float, default=3.0)
    main(p.parse_args())
