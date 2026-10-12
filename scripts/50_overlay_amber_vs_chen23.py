"""Overlay AMBER's native D_coeval/Limber kSZ power against the digitized
Chen+23 Fig. 10 fiducial curve."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_pipeline.amber import io
from ksz_pipeline.utils.constants import T_CMB_K

RUN = "runs/sweep_2gpc_1536/zmid8_dz4_az3_mh8_mfp3"
CHEN_CSV = "data/chen23_fiducial.csv"
OUT = "data/plots/amber_vs_chen23_fiducial_overlay.png"

la, Ca = io.read_amber_cl(os.path.join(RUN, "output", "cmb", "cl_ksz.txt"))

# cl_ksz.txt stores C_ell as dimensionless (dT/T_cmb)^2 -- convert to D_ell [muK^2]
T_cmb_uK = T_CMB_K * 1e6
Dl_amber = la * (la + 1) / (2 * np.pi) * Ca * T_cmb_uK**2

chen = np.loadtxt(CHEN_CSV, delimiter=",", skiprows=1)
l_chen, Dl_chen = chen[:, 0], chen[:, 1]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(la, Dl_amber, color="C0", lw=2, label="AMBER native (D_coeval/Limber), this run")
ax.plot(l_chen, Dl_chen, "o-", color="k", ms=4, label="Chen+23 fiducial (digitized, Fig. 10)")
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$D_\ell\ [\mu K^2]$")
ax.set_xlim(0, 8500)
ax.legend()
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
print(f"AMBER D_ell range: {Dl_amber.min():.3g} - {Dl_amber.max():.3g} muK^2 "
      f"over l={la.min():.0f}-{la.max():.0f}")
