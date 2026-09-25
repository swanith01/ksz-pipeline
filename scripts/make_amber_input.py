"""
scripts/make_amber_input.py

Write an AMBER input.txt (+ linpowspec.txt + amber_run.json sidecar) for
one reionization history, with cosmology PINNED to astropy Planck18 --
the cosmology limber.py/compute_cell hardcodes. AMBER's example inputs
use Om=0.3, h=0.7, s8=0.8; feeding those fields into compute_cell would
silently mix cosmologies.

Checks run every time (fail loudly, don't warn):
  1. chi(z) from AMBER's own H(z) formula (cosmo.f90) vs astropy Planck18,
     z in [4, 20]: must agree to < 0.1%.
  2. CMB shell midpoints (where fields are dumped) must be distinct at
     2 decimals -- AMBER's filename tag is 'z=%05.2f', so collisions
     would silently overwrite fields_*.dat files.
  3. Implied n_e0 (nH + nHe, He singly ionized) vs the repo's ne0_cgs()
     value, printed for comparison (tolerance 0.5%).

Usage:
  python scripts/make_amber_input.py --out runs/amber_q01 \\
      --L 256 --N 128 --zmid 8.0 --zdel 4.0 --zasy 3.0

Then on the cluster:
  cd runs/amber_q01 && mkdir -p output/{cosmo,reion,grf,lpt,esf,mesh,cmb}
  /path/to/amber.x < input/input.txt > output/log.txt
"""
import argparse
import json
import os

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18, FlatLambdaCDM
from scipy.integrate import quad

C_KMS = 299792.458

# Each preset returns (p, target) where p is AMBER's own parameter dict
# and target is the astropy Cosmology instance everything (check_distances,
# ne0, and compute_cell downstream) must be compared/matched against --
# keeping them paired here is what stops a cosmology override drifting
# out of sync with its own ne0, the exact bug class flagged in
# compute_cell's cosmology= docstring.

def planck18_for_amber():
    """Planck18 -> AMBER's (om, ol, ob, or) with AMBER's H(z) form.
    Massive neutrino (0.06 eV) is non-relativistic at z < 20 -> matter;
    photons + massless neutrinos -> radiation; flat by construction."""
    P18 = Planck18
    nu_rel = P18.Ogamma0 * 0.22710731766 * P18.Neff * (2.0 / 3.0)  # 2 massless
    orad = P18.Ogamma0 + nu_rel
    ol = P18.Ode0
    om = 1.0 - ol - orad
    p = dict(om=om, ol=ol, ob=P18.Ob0, orad=orad, h=P18.h,
             s8=P18.meta['sigma8'], ns=P18.meta['n'], Tcmb=P18.Tcmb0.value,
             mnu=0.06)
    return p, P18


def chen2022_for_amber():
    """Chen, Trac, Mukherjee & Cen 2023 (arXiv:2203.04337) fiducial
    cosmology: [Om,Ob,s8,ns,h] = [0.3, 0.045, 0.8, 0.96, 0.7] -- the same
    AMBER paper this pipeline's fiducial (z_mid,dz,Az)=(8,4,3) matches.
    No massive-neutrino / Tcmb0 / Neff statement in the paper; assumed
    standard (Tcmb0=2.725, Neff=3.046, massless) since nothing suggests
    otherwise -- flag this assumption if a precision match ever matters
    beyond kSZ D_ell comparisons."""
    target = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.045, Tcmb0=2.725,
                           Neff=3.046)
    nu_rel = target.Ogamma0 * 0.22710731766 * target.Neff
    orad = target.Ogamma0 + nu_rel
    ol = target.Ode0
    om = 1.0 - ol - orad
    p = dict(om=om, ol=ol, ob=target.Ob0, orad=orad, h=target.h,
             s8=0.8, ns=0.96, Tcmb=target.Tcmb0.value,
             mnu=0.0)  # assumed massless, see docstring
    return p, target


COSMOLOGY_PRESETS = {'planck18': planck18_for_amber,
                     'chen2022': chen2022_for_amber}


def chi_amber(z, p):
    """Comoving distance [Mpc] using AMBER's cosmo.f90 H(z)."""
    E = lambda zz: np.sqrt(p['orad'] * (1 + zz)**4 + p['om'] * (1 + zz)**3
                           + p['ol'])
    return C_KMS / (100.0 * p['h']) * quad(lambda zz: 1.0 / E(zz), 0, z)[0]


def check_distances(p, target, tol=1e-3):
    zs = np.arange(4.0, 20.01, 1.0)
    rel = np.array([chi_amber(z, p) / target.comoving_distance(z).value - 1
                    for z in zs])
    worst = np.abs(rel).max()
    print(f"chi(z) AMBER-vs-target cosmology, z=4..20: max |rel diff| = "
         f"{worst:.2e}")
    if worst > tol:
        raise SystemExit(f"cosmology mismatch {worst:.2e} > {tol:.0e}")


def shell_midpoints(zmin, zmax, zdel, spacing):
    """Replicates cmbreion.f90 cmb_ray's bin midpoints exactly."""
    if spacing == 'lin':
        nz = int(np.ceil((zmax - zmin) / zdel))
        return zmin + zdel * (np.arange(nz) + 0.5)
    dlg = np.log10(1 + zdel / (1 + zmin))
    nz = int(np.ceil(np.log10((1 + zmax) / (1 + zmin)) / dlg))
    return (1 + zmin) * 10**(dlg * (np.arange(nz) + 0.5)) - 1


def check_ne0(p, target, XH, YHe):
    """Returns the computed ne0 [cm^-3] -- the caller passes THIS to
    compute_cell(ne0=...), not the repo's Planck18-derived constant,
    whenever target isn't Planck18. Printed regardless so it's always
    visible which value a run actually used."""
    rho_b = (target.critical_density0 * target.Ob0).to(u.g / u.cm**3).value
    mp = 1.67262192e-24
    ne0 = rho_b / mp * (XH + YHe / 4.0)
    print(f"n_e0 (H + singly-ionized He) = {ne0:.6e} cm^-3  "
         f"[target cosmology: Ob={target.Ob0:.5f}, h={target.h:.4f}]")
    if target is Planck18:
        NE0_REPO = 2.064357e-07  # ne0_cgs() value, validation_table §4
        rel = ne0 / NE0_REPO - 1
        print(f"  vs repo ne0_cgs() = {NE0_REPO:.6e} ; rel diff {rel:+.2e}")
        if abs(rel) > 5e-3:
            raise SystemExit("XH/YHe/Ob inconsistent with repo ne0_cgs()")
    return ne0


def eh_nowiggle_T(k_h, p):
    """Eisenstein & Hu 1998 (ApJ 496, 605) no-wiggle transfer function,
    eqs. 26, 28-31. k_h in h/Mpc. Used only for the k > 150 h/Mpc tail."""
    h = p['h']
    omh2 = p['om'] * h**2
    obh2 = p['ob'] * h**2
    fb = p['ob'] / p['om']
    theta = p['Tcmb'] / 2.7
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10 * obh2**0.75)   # Mpc
    a_g = 1 - 0.328 * np.log(431 * omh2) * fb + 0.38 * np.log(22.3 * omh2) * fb**2
    k = k_h * h                                                     # 1/Mpc
    gam = p['om'] * h * (a_g + (1 - a_g) / (1 + (0.43 * k * s)**4))
    q = k_h * theta**2 / gam
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q**2)


def write_linpowspec(path, p):
    """z=0 linear Delta^2(k), k in h/Mpc, via CAMB, using THIS preset's
    own (Ob, Om, ns) -- AMBER renormalizes the amplitude to s8 itself
    (cosmology.f90), but the shape (Ob/Om ratio, ns) is not free and
    must match p, not be hardcoded to one preset."""
    import camb
    pars = camb.set_params(H0=100 * p['h'],
                           ombh2=p['ob'] * p['h']**2,
                           omch2=(p['om'] - p['ob']) * p['h']**2,
                           mnu=p['mnu'], ns=p['ns'], As=2.1e-9, WantTransfer=True,
                           kmax=200.0)
    pars.set_matter_power(redshifts=[0.0], kmax=200.0)
    res = camb.get_results(pars)
    # CAMB is only trustworthy inside its transfer k-range: below ~1e-4
    # h/Mpc it returns garbage (1e-38 at k=1e-5, found while testing this
    # script -- sigma8 did not catch it). Take [1e-4, 150] from CAMB and
    # extrapolate to AMBER's example range [1e-5, 1e4] h/Mpc, which its
    # excursion-set filter needs down to M_min:
    #   low k : Delta^2 ~ k^(3+ns)       (T(k)->1, exact limit)
    #   high k: Eisenstein & Hu (1998) no-wiggle T(k), amplitude-matched
    #           to CAMB at the last CAMB point. (A power-law extrapolation
    #           was tried first and overshot AMBER's own example file by
    #           2x at k=1e3 and 4.8x at k=1e4, since CDM's Delta^2 slope
    #           keeps flattening logarithmically. CAMB itself to k=1e4 is
    #           too slow to run per sweep point.)
    #           AMBER's own example file turns over (negative slope) above
    #           k~200, unlike pure CDM, so the two tails differ there by
    #           up to 3x -- but AMBER's sigma(M) (esf.f90 S_of_R, top-hat)
    #           suppresses that range by (kR)^-4: swapping tails changes
    #           sigma(M_min) by 1.4e-6 at M_min=1e8, 4.8e-4 at 1e6 Msun/h
    #           (checked 2026-09-10). Irrelevant for this study.
    kh, _, pk = res.get_matter_power_spectrum(minkh=1e-4, maxkh=150.0,
                                              npoints=1500)
    d2 = kh**3 * pk[0] / (2 * np.pi**2)
    k_lo = np.logspace(-5, np.log10(kh[0]), 60, endpoint=False)
    d2_lo = d2[0] * (k_lo / kh[0])**(3 + p['ns'])
    k_hi = np.logspace(np.log10(kh[-1]), 4, 120)[1:]
    eh = lambda k: k**(3 + p['ns']) * eh_nowiggle_T(k, p)**2
    d2_hi = d2[-1] * eh(k_hi) / eh(kh[-1])
    kh = np.concatenate([k_lo, kh, k_hi])
    d2 = np.concatenate([d2_lo, d2, d2_hi])
    sl = np.diff(np.log(d2)) / np.diff(np.log(kh))
    if not (np.all(np.isfinite(d2)) and np.all(d2 > 0)
            and sl[:50].min() > 3.5 and np.all(np.diff(kh) > 0)):
        raise SystemExit("linpowspec failed sanity checks (low-k slope / "
                         "positivity / monotonic k)")
    print(f"linpowspec: k=[{kh[0]:.1e},{kh[-1]:.1e}] h/Mpc")
    with open(path, 'w') as f:
        f.write(f"{'k':>12s} {'D^2_lin':>12s}\n")
        for a, b in zip(kh, d2):
            f.write(f"{a:12.6E} {b:12.6E}\n")


TEMPLATE = """#-------------------------------------------------------------------------------
# SIM
{ncore}\t\t\t\t# Number of core
input\t\t\t\t# Directory for inputs
output\t\t\t\t# Directory for outputs
#-------------------------------------------------------------------------------
# COSMO
{L}\t\t\t\t# Box length [Mpc/h]
{om:.6f}\t\t\t# Omega_m
{ol:.6f}\t\t\t# Omega_l
{ob:.6f}\t\t\t# Omega_b
{orad:.6E}\t\t\t# Omega_r
{h:.4f}\t\t\t\t# h
{s8:.4f}\t\t\t\t# sigma_8
{ns:.4f}\t\t\t\t# n_s
-1\t\t\t\t# w
{Tcmb:.4f}\t\t\t\t# T_cmb
1090\t\t\t\t# recombination redshift
{XH:.4f}\t\t\t\t# X hydrogen fraction
{YHe:.4f}\t\t\t\t# Y Helium fraction
linpowspec.txt\t\t\t# Linear power spectrum file
cosmo\t\t\t\t# Directory
#-------------------------------------------------------------------------------
# REION
write\t\t\t\t# make, read, write
{zmid:.4f}\t\t\t\t# Midpoint redshift
{zdel:.4f}\t\t\t\t# Duration
{zasy:.4f}\t\t\t\t# Asymmetry
0.05\t\t\t\t# Ion frac early
0.50\t\t\t\t# Ion frac mid
0.95\t\t\t\t# Ion frac late
{Mmin:.3E}\t\t\t# Minimum halo mass [Msolar/h]
{mfp}\t\t\t\t# Mean free path [Mpc/h]
reion\t\t\t\t# Directory
#-------------------------------------------------------------------------------
# GRF
make\t\t\t\t# make, read, write
{seed}\t\t\t\t# integer seed for GRF
grf\t\t\t\t# Directory
#-------------------------------------------------------------------------------
# LPT
make\t\t\t\t# make, read, write
2\t\t\t\t# Order: 1, 2
tsc\t\t\t\t# Assign: tsc, cic, ngp
lpt\t\t\t\t# Directory
#-------------------------------------------------------------------------------
# ESF
make\t\t\t\t# make, read, write
sharpk\t\t\t\t# Filter: sharpk, tophat
tsc\t\t\t\t# Assign: tsc, cic, ngp
esf\t\t\t\t# Directory
#-------------------------------------------------------------------------------
# Mesh
make\t\t\t\t# make, read, write
{N}\t\t\t\t# Number of cells/particles per side length
mesh\t\t\t\t# Directory
#-------------------------------------------------------------------------------
# CMB
write\t\t\t\t# CMB: make, write
{mapmake}\t\t\t\t# Map: make, write
{czmin}\t\t\t\t# z min
{czmax}\t\t\t\t# z max
{czdel}\t\t\t\t# z del
{cspacing}\t\t\t\t# z spacing: lin, log
100\t\t\t\t# l min
10000\t\t\t\t# l max
{nside}\t\t\t\t# Nside
cmb\t\t\t\t# Directory
#-------------------------------------------------------------------------------
# 21cm
no                              # make, write
5.5                             # z min
15.5\t\t\t\t# z max
1.0                             # z del
lin                             # z spacing: lin, log
21cm\t\t\t\t# Directory
#------------------------------------------------------------------------------
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--L', type=float, required=True, help='Mpc/h')
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--zmid', type=float, required=True)
    ap.add_argument('--zdel', type=float, required=True)
    ap.add_argument('--zasy', type=float, required=True)
    ap.add_argument('--Mmin', type=float, default=1e8)
    ap.add_argument('--mfp', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--ncore', type=int, default=16)
    ap.add_argument('--czmin', type=float, default=4.5)
    ap.add_argument('--czmax', type=float, default=18.5)
    ap.add_argument('--czdel', type=float, default=0.5)
    ap.add_argument('--cspacing', choices=['lin', 'log'], default='lin')
    ap.add_argument('--mapmake', default='no', help="'no' or 'write'")
    ap.add_argument('--nside', type=int, default=0)
    ap.add_argument('--XH', type=float, default=0.76)
    ap.add_argument('--YHe', type=float, default=0.24)
    ap.add_argument('--no-camb', action='store_true')
    ap.add_argument('--cosmology', choices=sorted(COSMOLOGY_PRESETS),
                    default='planck18',
                    help="'planck18' (default, matches compute_cell's "
                         "hardcoded default) or 'chen2022' (Chen et al. "
                         "2023 AMBER-kSZ paper's fiducial cosmology). "
                         "Whatever is NOT planck18 must be passed to "
                         "compute_cell(cosmology=..., ne0=...) explicitly "
                         "-- see amber_run.json's 'ne0_cgs' and "
                         "'cosmology_preset' for the exact matched values.")
    a = ap.parse_args()

    p, target = COSMOLOGY_PRESETS[a.cosmology]()
    check_distances(p, target)
    ne0 = check_ne0(p, target, a.XH, a.YHe)

    zm = shell_midpoints(a.czmin, a.czmax, a.czdel, a.cspacing)
    tags = [f"{z:05.2f}" for z in zm]
    if len(set(tags)) != len(tags):
        raise SystemExit("CMB shell midpoints collide at 2 decimals -> "
                         "fields_*.dat would overwrite; widen czdel")
    if a.mapmake == 'write':
        if a.nside <= 0 or (a.nside & (a.nside - 1)) != 0:
            raise SystemExit(
                f"--mapmake write needs --nside as a power of two "
                f"(128, 256, 512, ...); got {a.nside}. A reasonable first "
                f"value for N={a.N} is 128 or 256 -- higher costs more "
                f"compute/output for map detail this comparison doesn't "
                f"need yet.")
        print(f"map-making ON: Nside={a.nside} -> "
              f"{12*a.nside**2:,} pixels per shell")
    print(f"{len(zm)} snapshots at z = {np.round(zm, 3).tolist()}")

    os.makedirs(os.path.join(a.out, 'input'), exist_ok=True)
    txt = TEMPLATE.format(**p, **{k: v for k, v in vars(a).items()
                                  if k not in ('out', 'no_camb')})
    with open(os.path.join(a.out, 'input', 'input.txt'), 'w') as f:
        f.write(txt)
    if not a.no_camb:
        write_linpowspec(os.path.join(a.out, 'input', 'linpowspec.txt'), p)
    meta = dict(vars(a), cosmology_preset=a.cosmology, amber_params=p,
                ne0_cgs=ne0, snapshot_z=zm.tolist())
    with open(os.path.join(a.out, 'amber_run.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {a.out}/input/input.txt, amber_run.json"
          + ("" if a.no_camb else ", input/linpowspec.txt"))
    if a.cosmology != 'planck18':
        print(f"NOTE: cosmology={a.cosmology} -- downstream compute_cell "
             f"calls need cosmology=<matching astropy instance>, "
             f"ne0={ne0:.6e} passed explicitly, or they will silently "
             f"fall back to Planck18.")


if __name__ == '__main__':
    main()
