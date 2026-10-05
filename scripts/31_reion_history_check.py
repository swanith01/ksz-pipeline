#!/usr/bin/env python
"""
Script 31: does our reionization history match Nikolic et al. (2023) Fig. 1?

Their stated history (Fig. 1 caption): midpoint z_r = 6.1, end z_end = 4.9,
duration dz = z(xHI=0.75) - z(xHI=0.25) = 0.76, tau_e = 0.042.

The ensemble runs (script 17) do not save the neutral-fraction history, so
this runs the same boxes again via ONE py21cmfast.run_coeval(redshift=[...])
call over all z_snapshots of the chosen config and reports the same four
numbers. With write=True/direc=cache_dir and the same seed as an ensemble
member, boxes already in the cache are re-used, so for a no-PHOTON_CONS
config this is fast.

PHOTON_CONS note: passing the whole redshift list in one call is the only
way py21cmfast can calibrate the photon-conservation correction against the
full history (the pipeline's own per-snapshot run_coeval_fields() calls are
single-redshift calls, which the 21cmFAST docs say cannot be calibrated
meaningfully; a standalone z=6.1 PHOTON_CONS call gave xH=0.06 here).
Use configs/nikolic_mesinger_nophotoncons.yaml for the no-correction history
and configs/nikolic_mesinger.yaml for the PHOTON_CONS one.

tau_e here is a simple integral (Planck18 background, singly-ionized He
tracking H, x_e = 1 + f_He below z=4, nothing above the first snapshot),
good to a few x 1e-3 -- enough to compare against 0.042, not a precision
number.

Needs py21cmfast (use the p21c_v3 env, inside qsub). Usage:
    python scripts/31_reion_history_check.py --config configs/nikolic_mesinger_nophotoncons.yaml --seed 101
"""
import argparse
import os

import numpy as np
import yaml

TARGET = dict(z_r=6.1, z_end=4.9, dz=0.76, tau=0.042)


def z_cross(z, xH, level):
    """Redshift where xH first drops below `level` going to lower z (linear interp)."""
    order = np.argsort(z)[::-1]               # high z -> low z
    z, xH = np.asarray(z)[order], np.asarray(xH)[order]
    for i in range(1, len(z)):
        if xH[i - 1] >= level > xH[i]:
            f = (xH[i - 1] - level) / (xH[i - 1] - xH[i])
            return float(z[i - 1] + f * (z[i] - z[i - 1]))
    return float('nan')


def tau_e(z, xH, Y=0.245):
    from astropy.cosmology import Planck18 as cosmo
    import astropy.units as u
    from astropy import constants as const
    order = np.argsort(z)
    z, xH = np.asarray(z)[order], np.asarray(xH)[order]
    zz = np.linspace(0.0, 25.0, 25001)
    f_he = Y / (4.0 * (1.0 - Y))
    xe = np.where(zz < z.min(), 1.0, np.interp(zz, z, 1.0 - xH, right=0.0)) * (1.0 + f_he)
    rho_c0 = cosmo.critical_density0.to(u.g / u.cm**3)
    nH0 = (rho_c0 * cosmo.Ob0 * (1.0 - Y) / const.m_p).to(1 / u.cm**3)
    H = cosmo.H(zz).to(1 / u.s)
    integrand = (const.c.to(u.cm / u.s) * const.sigma_T.to(u.cm**2) * nH0 * xe
                 * (1.0 + zz) ** 2 / H).to(u.dimensionless_unscaled).value
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(integrand, zz))


def summarize(z, xH):
    out = dict(z_r=z_cross(z, xH, 0.5),
               z75=z_cross(z, xH, 0.75), z25=z_cross(z, xH, 0.25),
               z_xH05=z_cross(z, xH, 0.05), z_xH01=z_cross(z, xH, 0.01),
               tau=tau_e(z, xH))
    out['dz'] = out['z75'] - out['z25']
    return out


def main(config_path, seed, tag):
    import py21cmfast as p21c
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sim = cfg['21cmfast']
    zs = sorted([float(z) for z in cfg['coeval_ksz']['z_snapshots']], reverse=True)
    cache_dir = cfg['data']['cache_dir']
    out_dir = cfg['data']['output_dir'].rstrip('/')
    flag_options = sim.get('flag_options')
    print(f"config={config_path} seed={seed} BOX_LEN={sim['BOX_LEN']} "
          f"HII_DIM={sim['HII_DIM_coeval']} flag_options={flag_options}")

    kw = dict(redshift=zs,
              user_params={"HII_DIM": int(sim['HII_DIM_coeval']), "BOX_LEN": float(sim['BOX_LEN']),
                           "N_THREADS": int(sim.get('N_THREADS', os.environ.get('OMP_NUM_THREADS', 1)))},
              random_seed=seed, write=True, direc=cache_dir)
    if sim.get('astro_params') is not None:
        kw['astro_params'] = sim['astro_params']
    if flag_options is not None:
        kw['flag_options'] = flag_options
    res = p21c.run_coeval(**kw)
    if not isinstance(res, (list, tuple)):
        res = [res]
    z = np.array([float(c.redshift) for c in res])
    xH = np.array([float(np.mean(c.xH_box)) for c in res])
    o = np.argsort(z)[::-1]
    z, xH = z[o], xH[o]

    print("\n   z      <xH>")
    for zi, xi in zip(z, xH):
        print(f"  {zi:5.2f}  {xi:7.4f}")

    s = summarize(z, xH)
    print("\n                         ours    Nikolic+23 Fig.1")
    print(f"  midpoint z(xH=0.5)    {s['z_r']:6.2f}   {TARGET['z_r']:6.2f}")
    print(f"  dz = z(.75)-z(.25)    {s['dz']:6.2f}   {TARGET['dz']:6.2f}")
    print(f"  z(xH=0.05)            {s['z_xH05']:6.2f}   z_end = {TARGET['z_end']:.2f} (their xH threshold not stated)")
    print(f"  z(xH=0.01)            {s['z_xH01']:6.2f}")
    print(f"  tau_e (approx)        {s['tau']:6.4f}   {TARGET['tau']:6.4f}")

    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/reion_history_seed{seed}{tag}.npz"
    np.savez(path, z=z, xH_mean=xH, config=config_path, seed=seed, **s)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nikolic_mesinger_nophotoncons.yaml")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--tag", default="", help="suffix for the output .npz name, e.g. _photoncons")
    a = ap.parse_args()
    main(a.config, a.seed, a.tag)
