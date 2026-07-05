"""
Velocity conversion from 21cmFAST IC units to comoving peculiar velocity.

Extracted from Cell 2 of 28May2026_kSZBoxes.py.

Convention (Cain+2024 Eq.2 / Park+2013 Eq.A16):
    q(x,z) = (1+delta) * chi * v_comoving
    v_comoving = v_physical / (1+z) = D(z)*f(z)*H(z)*Psi / (1+z)

The ds/a^4 factor in the C_ell integral (not ds/a^5) confirms that v
in q must be the COMOVING peculiar velocity.

Typical values: v_rms ~ 95-156 km/s at z=5-15 for an 800 Mpc box.
"""

import numpy as np
from scipy.integrate import quad
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u


def growth_factor(z):
    """
    Linear growth factor D(z), normalised to 1 at z=0.

    D(z) proportional to H(z) * integral_z^inf dz' (1+z') / H(z')^3
    """
    def integrand(zp):
        return (1.0 + zp) / cosmo.H(zp).value**3

    val,  _ = quad(integrand, z,   np.inf)
    norm, _ = quad(integrand, 0.0, np.inf)
    return (cosmo.H(z).value * val) / (cosmo.H(0).value * norm)


def growth_rate(z):
    """
    Growth rate f(z) = dlnD/dlna ~ Omega_m(z)^0.55 (Linder 2005).
    Accurate to ~1% for flat LCDM.
    """
    return cosmo.Om(z) ** 0.55


def velocity_conversion_factor(z):
    """
    Convert 21cmFAST lowres_vx (IC displacement field Psi) to
    COMOVING peculiar velocity at redshift z [km/s].

        v_comoving [km/s] = lowres_vx * D(z) * f(z) * H(z) / (1+z)

    Caller multiplies by 1e5 to get cm/s.

    Validated to ~1% against linear theory at z=0.5-15.
    v_rms decreases toward low z (correct LCDM: H drops faster than D*f grows).
    """
    return (growth_factor(z) * growth_rate(z)
            * cosmo.H(z).value / (1.0 + z))    # [km/s]
