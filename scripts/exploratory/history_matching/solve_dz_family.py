#!/usr/bin/env python
"""
solve_dz_family.py -- history-matched 21cmFAST parameter sets at FIXED z_re,
varying duration (delta_z), holding R_BUBBLE_MAX at its 21cmFAST default
(15 Mpc). This is the complementary axis to solve_history.py (which fixes
the whole history and varies R_BUBBLE_MAX): here the astrophysics is what
moves, morphology is held fixed, and we ask what a family of durations at
one fixed midpoint looks like.

Targets are generated from a smooth tanh-in-y transition,

    x_e(z) = 0.5*fH*(1 + tanh((y_re - y)/delta_y)),   y = (1+z)^1.5,
    delta_y = 1.5*(1+z_re)^0.5*delta_z,   fH = 1.0

This is the SAME functional form Gorce+2026 Eq.(2) uses for the HeIII
transition (there with z_re,HeIII=5.0, delta_z=0.5 fixed) -- reused here for
the MAIN hydrogen curve, with fH=1.0 rather than her Eq.(1)+Eq.(3)'s
fH~1.08, because 21cmFAST's xH_box in this pipeline is pure hydrogen (no
helium flags on -- see solve_history.py's own FLAGS printout), so a target
that asymptotes above 1.0 could never be matched. The tanh form is also
smooth everywhere with an EXACT z_re by construction (xe(z_re)=0.5 always,
whatever delta_z is) and no derivative kink, unlike her Eq.(1)'s piecewise
power-law -- much better-conditioned for a smooth (zeta, Tvir) fit.

Not a closure test in solve_history.py's sense (no known "right answer" to
recover, since these targets are new, not measured from a real run).
Instead, each fit's own z_re -- the 50% crossing of the MATCHED curve, not
the target -- is reported and should land close to --z-re-ref for every
delta_z: that consistency, not a residual number alone, is the check that
the family is actually fixed-z_re.

CAVEAT: for wide delta_z, the target may not have fully saturated by the
bottom of the z-grid (z=5) -- x_e(z=5) is printed per target so this is
visible rather than hidden.

Usage (p21c_v3 env, on a compute node):
  python solve_dz_family.py --z-re-ref 8.0 --delta-z-list 0.5 1.0 1.5 2.5 4.0
"""
import argparse
import json
import time

import numpy as np
from scipy.optimize import least_squares

import py21cmfast as p21c


def tanh_xe(z, z_re, delta_z, fH=1.0):
    y = (1.0 + z) ** 1.5
    y_re = (1.0 + z_re) ** 1.5
    delta_y = 1.5 * (1.0 + z_re) ** 0.5 * delta_z
    return 0.5 * fH * (1.0 + np.tanh((y_re - y) / delta_y))


def crossing_z(z_arr, xe_arr, level=0.5):
    """Redshift where xe_arr (any monotonic-in-z shape) crosses `level`."""
    order = np.argsort(xe_arr)
    return float(np.interp(level, xe_arr[order], np.asarray(z_arr)[order]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z-re-ref", type=float, default=8.0,
                    help="fixed midpoint redshift for every target in the family")
    ap.add_argument("--delta-z-list", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.5, 4.0])
    ap.add_argument("--box-len", type=float, default=400.0)
    ap.add_argument("--hii-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--direc", default="/user1/swanith/hist_match/cache")
    ap.add_argument("--max-nfev", type=int, default=40)
    ap.add_argument("--out", default="solutions_dz_family.json")
    a = ap.parse_args()

    zs = np.arange(20.0, 4.99, -0.5)          # same grid as solve_history.py

    up = p21c.UserParams(HII_DIM=a.hii_dim, BOX_LEN=a.box_len, N_THREADS=a.threads)
    ic = p21c.initial_conditions(user_params=up, random_seed=a.seed, direc=a.direc)
    pf = {z: p21c.perturb_field(redshift=z, init_boxes=ic, direc=a.direc) for z in zs}
    fl = p21c.FlagOptions()

    def astro(zeta, tvir):
        return p21c.AstroParams(HII_EFF_FACTOR=float(zeta), ION_Tvir_MIN=float(tvir))

    def history(ap_):
        xe = []
        for z in zs:
            ib = p21c.ionize_box(redshift=z, init_boxes=ic, perturbed_field=pf[z],
                                 astro_params=ap_, flag_options=fl, write=False)
            xe.append(1.0 - float(np.mean(ib.xH_box)))
        return np.array(xe)

    d0 = p21c.AstroParams()
    tv0 = float(d0.ION_Tvir_MIN)

    print(f"z_re_ref={a.z_re_ref}, delta_z list={a.delta_z_list}, "
          f"box={a.box_len:g}/{a.hii_dim}^3, R_BUBBLE_MAX=default (15 Mpc)")

    results = []
    u0 = np.array([np.log(30.0), tv0])         # cold start, near the fiducial
    for dz in a.delta_z_list:
        target = tanh_xe(zs, a.z_re_ref, dz)
        print(f"  target dz={dz:4.1f}: xe(z=20)={target[0]:.2e}, "
              f"xe(z=5)={target[-1]:.4f}"
              f"{'  <-- not fully saturated at grid bottom' if target[-1] < 0.99 else ''}")

        t0 = time.time()
        fun = lambda u: history(astro(np.exp(u[0]), u[1])) - target
        sol = least_squares(fun, u0,
                            bounds=([np.log(3.0), 4.0], [np.log(1000.0), 5.8]),
                            x_scale=[0.3, 0.1], diff_step=0.01,
                            xtol=1e-4, ftol=1e-6, max_nfev=a.max_nfev)
        zeta_fit, tvir_fit = float(np.exp(sol.x[0])), float(sol.x[1])
        matched = history(astro(zeta_fit, tvir_fit))
        z_re_achieved = crossing_z(zs, matched, 0.5)
        z25 = crossing_z(zs, matched, 0.25)
        z75 = crossing_z(zs, matched, 0.75)

        res = dict(delta_z_target=float(dz), z_re_ref=float(a.z_re_ref),
                  zeta=zeta_fit, log10_Tvir=tvir_fit,
                  max_abs_dx=float(np.abs(sol.fun).max()), nfev=int(sol.nfev),
                  z_re_achieved=z_re_achieved, width_z25_z75=float(z25 - z75),
                  minutes=round((time.time() - t0) / 60, 2))
        print(f"    -> zeta={zeta_fit:.3f}  log10Tvir={tvir_fit:.4f}  "
              f"max|dx|={res['max_abs_dx']:.4f}  z_re_achieved={z_re_achieved:.3f} "
              f"(target {a.z_re_ref})  width25-75={res['width_z25_z75']:.3f}  "
              f"[{res['minutes']:.2f} min]", flush=True)
        results.append(res)
        json.dump(results, open(a.out, "w"), indent=1)
        u0 = np.array([np.log(zeta_fit), tvir_fit])   # warm start the next dz


if __name__ == "__main__":
    main()
