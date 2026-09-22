import argparse, time, json
import numpy as np
from scipy.optimize import least_squares
import py21cmfast as p21c

ap = argparse.ArgumentParser()
ap.add_argument("--box-len", type=float, default=400.0)
ap.add_argument("--hii-dim", type=int, default=64)
ap.add_argument("--target-zeta", type=float, default=30.0)
ap.add_argument("--rmax-list", type=float, nargs="*", default=[10.0, 20.0, 40.0, 80.0])
ap.add_argument("--seed", type=int, default=37)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--direc", default="/user1/swanith/hist_match/cache")
ap.add_argument("--max-nfev", type=int, default=15)
ap.add_argument("--out", default="solutions_400_64.json")
a = ap.parse_args()

zs = np.arange(20.0, 4.99, -0.5)
d0 = p21c.AstroParams()
print("DEFAULTS:", d0, flush=True)
TV0 = float(d0.ION_Tvir_MIN)

up = p21c.UserParams(HII_DIM=a.hii_dim, BOX_LEN=a.box_len, N_THREADS=a.threads)
ic = p21c.initial_conditions(user_params=up, random_seed=a.seed, direc=a.direc)
pf = {z: p21c.perturb_field(redshift=z, init_boxes=ic, direc=a.direc) for z in zs}
fl = p21c.FlagOptions()
print("FLAGS:", fl, flush=True)

def astro(zeta, tvir, rmax):
    kw = dict(HII_EFF_FACTOR=float(zeta), ION_Tvir_MIN=float(tvir))
    if rmax is not None:
        kw["R_BUBBLE_MAX"] = float(rmax)
    return p21c.AstroParams(**kw)

def history(ap_):
    xe = []
    for z in zs:
        ib = p21c.ionize_box(redshift=z, init_boxes=ic, perturbed_field=pf[z],
                             astro_params=ap_, flag_options=fl, write=False)
        xe.append(1.0 - float(np.mean(ib.xH_box)))
    return np.array(xe)

target = history(astro(a.target_zeta, TV0, None))
np.save(a.out.replace(".json", "_target.npy"), target)
print(f"target: zeta={a.target_zeta}, Tvir={TV0:.4f}, default R_BUBBLE_MAX", flush=True)

results = []
for r in [None] + list(a.rmax_list):          # None = default Rmax = closure test
    t0 = time.time()
    fun = lambda u: history(astro(np.exp(u[0]), u[1], r)) - target
    u0 = (np.array([np.log(results[-1]['zeta']), results[-1]['log10_Tvir']])
          if (results and r is not None)
          else np.array([np.log(1.5 * a.target_zeta), TV0 + 0.15]))  # warm start
    sol = least_squares(fun, u0,
                        bounds=([np.log(3.0), 4.0], [np.log(1000.0), 5.8]),
                        x_scale=[0.3, 0.1], diff_step=0.01,
                        xtol=1e-4, ftol=1e-6, max_nfev=a.max_nfev)
    res = dict(rmax=r, zeta=float(np.exp(sol.x[0])), log10_Tvir=float(sol.x[1]),
               max_abs_dx=float(np.abs(sol.fun).max()), nfev=int(sol.nfev),
               minutes=round((time.time() - t0) / 60, 2))
    print(res, flush=True)
    results.append(res)
    json.dump(results, open(a.out, "w"), indent=1)
