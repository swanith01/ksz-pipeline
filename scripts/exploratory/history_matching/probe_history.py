import argparse, time
import numpy as np
import py21cmfast as p21c
import astropy.units as u, astropy.constants as c
from astropy.cosmology import Planck18 as cos

ap = argparse.ArgumentParser()
ap.add_argument("--box-len", type=float, default=400.0)
ap.add_argument("--hii-dim", type=int, default=64)
ap.add_argument("--zeta", type=float, default=30.0)
ap.add_argument("--tvir", type=float, default=None)   # log10 K -> ION_Tvir_MIN
ap.add_argument("--rmax", type=float, default=None)   # Mpc -> R_BUBBLE_MAX
ap.add_argument("--seed", type=int, default=37)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--direc", default="/user1/swanith/hist_match/cache")
a = ap.parse_args()

zs = np.arange(20.0, 4.99, -0.5)          # descending

def astro(zeta, tvir=None, rmax=None):
    kw = dict(HII_EFF_FACTOR=zeta)
    if tvir is not None: kw["ION_Tvir_MIN"] = tvir
    if rmax is not None: kw["R_BUBBLE_MAX"] = rmax
    return p21c.AstroParams(**kw)

t0 = time.time()
up = p21c.UserParams(HII_DIM=a.hii_dim, BOX_LEN=a.box_len, N_THREADS=a.threads)
ic = p21c.initial_conditions(user_params=up, random_seed=a.seed, direc=a.direc)
pf = {z: p21c.perturb_field(redshift=z, init_boxes=ic, direc=a.direc) for z in zs}
t_setup = time.time() - t0

fl = p21c.FlagOptions()                    # defaults; reconcile with fields.py

def history(ap_):
    xe = []
    for z in zs:
        ib = p21c.ionize_box(redshift=z, init_boxes=ic, perturbed_field=pf[z],
                             astro_params=ap_, flag_options=fl, write=False)
        xe.append(1.0 - float(np.mean(ib.xH_box)))
    return np.array(xe)

def zx(xe, level):
    return float(np.interp(level, np.maximum.accumulate(xe), zs))

Y = 0.245
fHe = Y / (4 * (1 - Y))
nH0 = ((1 - Y) * cos.Ob0 * cos.critical_density0 / c.m_p).to(u.cm**-3)
def tau(xe):
    zz = np.linspace(0, 20, 4001)
    x = np.interp(zz, zs[::-1], xe[::-1], left=1.0)
    k = (c.c * c.sigma_T * nH0 * (1 + zz)**2 / cos.H(zz)).to(u.dimensionless_unscaled).value
    return float(np.trapz(k * x * (1 + fHe), zz))   # approx: ignores He III

print(f"setup (ICs + perturb, {a.box_len:.0f} Mpc / {a.hii_dim}^3): {t_setup:.1f} s")
for label, zeta in [("base", a.zeta), ("zeta x1.2", 1.2 * a.zeta)]:
    t1 = time.time()
    xe = history(astro(zeta, a.tvir, a.rmax))
    print(f"{label:10s} zeta={zeta:6.2f}  z25={zx(xe,.25):.2f} z50={zx(xe,.5):.2f} "
          f"z75={zx(xe,.75):.2f} z99={zx(xe,.99):.2f}  tau~{tau(xe):.4f}  "
          f"[{time.time()-t1:.1f} s per history]")
