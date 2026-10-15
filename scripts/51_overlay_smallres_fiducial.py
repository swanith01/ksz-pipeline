"""Three-way overlay at sweep_chen2023's fiducial point (smaller box/grid
than the production sweep_2gpc_1536 run): AMBER native, our independently
recomputed D_coeval/Limber (already saved in amber_gate.npz), and the
digitized Chen+23 Fig. 10 fiducial curve.

Resolution-limited check -- disagreement with the paper curve here may be a
box/grid-size effect, not a method error. What it cleanly tests regardless
of resolution is whether our method agrees with AMBER's own native calc.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_pipeline.amber import io
from ksz_pipeline.utils.constants import T_CMB_K

RUN = "runs/sweep_chen2023/zmid8_dz4_az3_mh8_mfp3"
CHEN_CSV = "data/chen23_fiducial.csv"
OUT = "data/plots/amber_vs_ours_vs_chen23_smallres_fiducial.png"

la, Ca = io.read_amber_cl(os.path.join(RUN, "output", "cmb", "cl_ksz.txt"))
T_cmb_uK = T_CMB_K * 1e6
Dl_amber = la * (la + 1) / (2 * np.pi) * Ca * T_cmb_uK**2

gate = np.load(os.path.join(RUN, "amber_gate.npz"))
ells_ours, Dl_ours = gate["ells"], gate["D_compute_cell"]

chen = np.loadtxt(CHEN_CSV, delimiter=",", skiprows=1)
l_chen, Dl_chen = chen[:, 0], chen[:, 1]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(la, Dl_amber, color="C0", lw=2, label="AMBER native, sweep_chen2023 res")
ax.plot(ells_ours, Dl_ours, "s--", color="C1", ms=5, lw=1.5, label="Our D_coeval/Limber, same run")
ax.plot(l_chen, Dl_chen, "o-", color="k", ms=4, label="Chen+23 fiducial (digitized)")
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$D_\ell\ [\mu K^2]$")
ax.set_xlim(0, 8500)
ax.legend()
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
print(f"AMBER native: {Dl_amber.min():.3g} - {Dl_amber.max():.3g} muK^2")
print(f"Our method:   {Dl_ours.min():.3g} - {Dl_ours.max():.3g} muK^2")
