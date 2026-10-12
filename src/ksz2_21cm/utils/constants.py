"""
Physical constants and mean electron density.

VENDORED from ksz-pipeline (github.com/swanith01/ksz-pipeline), commit
7305482def70cd94ab1f7b553db13070fd619b76 (2026-10-06), file
src/ksz_pipeline/utils/constants.py, verbatim below this header.

RECONCILIATION NOTE: this repo's own src/ksz2_21cm/utils/cosmology.py
already uses H0=67.66, Om0=0.30964144154550644 -- py21cmfast 3.3.1's own
CosmoParams() defaults. Those numbers are (to the digits that matter)
astropy's Planck18 preset, which is exactly what this vendored file uses
for all post-processing (tau, C_ell, comoving distances). So this does
NOT introduce a second, competing cosmology into the repo -- it's the
same one cosmology.py already settled on, just pulled from
astropy.cosmology.Planck18 directly rather than hardcoded. Don't hand-
edit the numbers in cosmology.py independently of this file going
forward, or that could drift.

Do not hand-edit this file to "merge" it with anything already in
ksz2-21cm -- if ksz2-21cm needs a constant this file doesn't have, add it
alongside, don't fork the ones that already exist here.

---

All CGS unless noted. Uses astropy for constants and Planck18 cosmology
throughout — matches the convention in kSZBoxes.py (Cell 2).

NOTE: the lightcone simulation itself uses 21cmFAST's internal cosmology
(slightly different from Planck18). This mismatch is small and documented
in MIGRATION.md. For post-processing (tau, C_ell) we use Planck18 everywhere.
"""

from astropy.cosmology import Planck18 as cosmo
from astropy import constants as const

# Thomson cross-section [cm^2]
SIGMA_T = const.sigma_T.cgs.value

# 1 Mpc in cm
MPC_CM = 3.0856775814913673e24

# CMB temperature today [K]
T_CMB_K = 2.7255

# Speed of light [cm/s]
C_CGS = const.c.cgs.value

# Hubble constant today [km/s/Mpc], Planck18
H0_KM_S_MPC = cosmo.H0.value


def ne0_cgs(Y_He=0.24, include_He=True):
    """
    Mean comoving electron number density today [cm^-3].
    Assumes fully ionized H and singly ionized He (He II).

    Parameters
    ----------
    Y_He       : helium mass fraction (default 0.24)
    include_He : if True, adds He II electrons (default True)

    Notes
    -----
    The lightcone script uses a fixed ne0 = 2.06e-7 cm^-3 (hydrogen only).
    The coeval script calls this function with include_He=True, giving a
    slightly different normalisation — primary source of the baseline offset
    between the two pipelines (Semester-6 notes sec 1.1).
    """
    m_p    = const.m_p.cgs.value
    rho_c0 = cosmo.critical_density0.cgs.value
    X_H    = 1.0 - Y_He
    n_H0   = X_H * (cosmo.Ob0 * rho_c0) / m_p
    if include_He:
        y = Y_He / (4.0 * X_H)
        electrons_per_H = 1.0 + y
    else:
        electrons_per_H = 1.0
    return n_H0 * electrons_per_H


# Hydrogen-only ne0 matching the lightcone script exactly
NE0_HYDROGEN_ONLY = 2.06e-7   # cm^-3
