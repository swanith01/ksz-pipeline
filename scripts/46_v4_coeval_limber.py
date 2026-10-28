#!/usr/bin/env python
"""
Script 46: coeval-direct (Limber) kSZ D_ell from py21cmfast v4 coeval boxes --
the reference the native v4 lightcones (script 17 --source native_v4 /
native_v4_ang) should be compared with in script 45 (--v4-direct).

Why: a v4 native lightcone is a different simulation from the v3 coeval boxes
behind our usual coeval-direct curve (v4 has its own astrophysics defaults and
code paths), so comparing it with the v3 Limber curve mixes two universes. This
script uses the SAME InputParameters the v4 lightcone was built from
(native_lightcone.stitch_lightcone_native_v4: build_inputs(seed, HII_DIM,
BOX_LEN, build_node_redshifts(z_min, z_max, 30), HII_EFF_FACTOR, N_THREADS)),
reads the coeval boxes at the node redshifts from the v4 cache (the lightcone run
already computed them; regenerate=False reuses them), and runs them through the
SAME Limber code as v3: coeval.momentum.qperp_power + coeval.limber.compute_cell.

Units (v4 coeval, per the ksz-lae-xcorr note and script 39): density is raw
delta; velocity_x/y/z are dx/dt in Mpc/s, so v_pec = velocity/(1+z). This is
CHECKED per box below: the linear peculiar velocity predicted from the box's own
density (continuity, Gaussian R = 15 Mpc smoothing) is regressed against the
converted v_z and v_x. The fit slope should be ~1 at every z with corr ~ +1;
anything else stops the run before a wrong reference is written.

Needs the py21cmfast 4.1 env (p21c_v41). Writes an npz with ell_direct,
Dl_direct, box_len, the redshifts and x_HI used, and the unit-check table.

Usage (match the lightcone run: script 17 uses configs/fiducial.yaml ->
seed 37, HII_EFF_FACTOR 30, N_THREADS 32, z 4-20, cache data/cache_v4):
    python scripts/46_v4_coeval_limber.py --box-len 400 --hii-dim 256 \
        --out data/products/v4_coeval_direct_box400.npz
"""
import argparse
import os

import numpy as np
import yaml

from ksz_pipeline.ksz.lightcone_v4 import build_inputs, build_node_redshifts
from ksz_pipeline.coeval.momentum import qperp_power
from ksz_pipeline.coeval.limber import compute_cell
from ksz_pipeline.utils.constants import MPC_CM

MPC_KM = MPC_CM / 1e5


def linear_velocity_slope(delta, v_kms, axis, L, z, R=15.0):
    """Regress v (km/s) on the linear prediction from delta along one axis."""
    from astropy.cosmology import Planck18 as c
    N = delta.shape[0]
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    K = np.meshgrid(k1, k1, k1, indexing="ij")
    K2 = K[0] ** 2 + K[1] ** 2 + K[2] ** 2
    K2s = K2.copy(); K2s[0, 0, 0] = 1.0
    W = np.exp(-0.5 * K2 * R ** 2)
    aHf = c.H(z).value / (1 + z) * c.Om(z) ** 0.55
    d = delta.astype(np.float64); d -= d.mean()
    vk = 1j * aHf * np.fft.fftn(d) * K[axis] / K2s; vk[0, 0, 0] = 0
    vp = np.fft.ifftn(vk * W).real
    v = v_kms.astype(np.float64); v -= v.mean()
    vr = np.fft.ifftn(np.fft.fftn(v) * W).real
    slope = float(np.sum(vr * vp) / np.sum(vp ** 2))
    corr = float(np.sum(vr * vp) / np.sqrt(np.sum(vr ** 2) * np.sum(vp ** 2)))
    return slope, corr


def main(a):
    import py21cmfast as p21c
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    sim = cfg["21cmfast"]
    seed = int(a.random_seed if a.random_seed is not None else sim["random_seed"])
    n_threads = int(a.n_threads if a.n_threads is not None else sim["N_THREADS"])
    hii_eff = sim.get("HII_EFF_FACTOR")
    cache_dir = a.cache_dir or (cfg["data"]["cache_dir"].rstrip("/") + "_v4")
    node_z = build_node_redshifts(sim["z_min"], sim["z_max"], n_nodes=a.n_nodes)
    inputs = build_inputs(seed, a.hii_dim, a.box_len, node_z,
                          HII_EFF_FACTOR=hii_eff, N_THREADS=n_threads)
    print(f"py21cmfast {p21c.__version__}; L={a.box_len} Mpc, N={a.hii_dim}, seed={seed}, "
          f"HII_EFF_FACTOR={hii_eff}, N_THREADS={n_threads}, cache={cache_dir}")
    print(f"node redshifts ({len(node_z)}): {np.round(np.sort(node_z), 2)}")
    cache = p21c.OutputCache(direc=cache_dir)

    results, checks = {}, []
    for z in sorted(node_z, reverse=True):
        coevals = p21c.run_coeval(inputs=inputs, out_redshifts=(float(z),), cache=cache,
                                  regenerate=False, progressbar=False)
        c = coevals[0] if isinstance(coevals, (list, tuple)) else coevals
        delta = np.asarray(c.density, dtype=np.float32)
        xH = np.asarray(c.neutral_fraction, dtype=np.float32)
        conv = MPC_CM / (1.0 + z)                                   # dx/dt [Mpc/s] -> v_pec [cm/s]
        vx = np.asarray(c.velocity_x, dtype=np.float32) * conv
        vy = np.asarray(c.velocity_y, dtype=np.float32) * conv
        vz = np.asarray(c.velocity_z, dtype=np.float32) * conv
        sz, cz = linear_velocity_slope(delta, vz / 1e5, 2, a.box_len, z)
        sx, cx = linear_velocity_slope(delta, vx / 1e5, 0, a.box_len, z)
        checks.append((z, float(delta.mean()), float(delta.min()), float(xH.mean()), sz, cz, sx, cx,
                       float(np.sqrt(np.mean((vz.astype(np.float64) / 1e5) ** 2)))))
        print(f"  z={z:6.2f}  <delta>={delta.mean():+.4f} min={delta.min():+.3f}  <x_HI>={xH.mean():.4f}  "
              f"v_pec rms(z)={checks[-1][-1]:6.1f} km/s  linear-fit slope z/x = {sz:.3f}/{sx:.3f} "
              f"(corr {cz:+.2f}/{cx:+.2f})", flush=True)
        k, P, Pstd = qperp_power(delta, xH, vx, vy, vz, a.box_len)
        results[float(z)] = dict(k=k, Pqperp=P, Pstd=Pstd, xH_mean=float(xH.mean()))
        del c, coevals, delta, xH, vx, vy, vz

    chk = np.array(checks)
    patchy = (chk[:, 3] > 1e-4) & (chk[:, 3] < 1 - 1e-4) & (chk[:, 0] <= 12)
    sl = chk[patchy][:, [4, 6]] if patchy.any() else chk[:, [4, 6]]
    cr = chk[patchy][:, [5, 7]] if patchy.any() else chk[:, [5, 7]]
    print(f"[units] velocity fit slope over the patchy boxes: median {np.median(sl):.3f}, "
          f"range {sl.min():.3f}-{sl.max():.3f}; corr min {cr.min():+.2f}")
    if not (0.8 < np.median(sl) < 1.25 and cr.min() > 0.8):
        raise SystemExit("[units] v4 coeval velocity does NOT match the linear prediction after "
                         "/(1+z) -- not writing a reference. Paste this log.")
    if not (np.abs(chk[:, 1]).max() < 0.01 and chk[:, 2].min() >= -1.0001):
        raise SystemExit("[units] v4 coeval density is not raw delta (mean ~0, min >= -1) -- stopping.")

    ells, Dl, sig, C, sigC, (ZS, tau), (_, xe) = compute_cell(results)
    D3000 = float(np.interp(3000, ells, Dl))
    zh = np.array(sorted(results)); xh = np.array([results[z]["xH_mean"] for z in zh])
    zmid = float(np.interp(0.5, xh, zh)) if xh.min() < 0.5 < xh.max() else float("nan")
    print(f"v4 coeval-direct D_3000 = {D3000:.4g} uK^2; v4 reionization midpoint z(x_HI=0.5) = {zmid:.2f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, ell_direct=ells, Dl_direct=Dl, sigma_direct=sig, D3000=D3000,
             box_len=float(a.box_len), hii_dim=int(a.hii_dim), random_seed=seed,
             z=zh, xH_mean=xh, z_mid=zmid, tau_z=ZS, tau=tau,
             unit_checks=chk, unit_check_cols=np.array(
                 ["z", "delta_mean", "delta_min", "xH_mean", "slope_vz", "corr_vz",
                  "slope_vx", "corr_vx", "vz_rms_kms"]))
    print("saved", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fiducial.yaml")
    p.add_argument("--box-len", type=float, required=True)
    p.add_argument("--hii-dim", type=int, required=True)
    p.add_argument("--random-seed", type=int, default=None, help="default: config 21cmfast.random_seed")
    p.add_argument("--n-threads", type=int, default=None,
                   help="default: config N_THREADS (must match the lightcone run for cache reuse)")
    p.add_argument("--n-nodes", type=int, default=30)
    p.add_argument("--cache-dir", default=None, help="default: <config cache_dir>_v4 (as script 17)")
    p.add_argument("--out", required=True)
    main(p.parse_args())
