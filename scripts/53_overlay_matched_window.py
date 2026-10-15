"""Matched-window overlay: our D_coeval/Limber (results_qperp_from_amber ->
compute_cell, restricted to the patchy window z=7.25-14.25) against AMBER's
OWN window-restricted calculation (D_amber_window, same npz, same window,
same ell grid) -- this is the actual GATE 2b comparison. The Chen+23
digitized fiducial is shown for reference too, but it represents the FULL
(un-windowed) kSZ signal, so it isn't expected to match this windowed pair
as closely -- that comparison belongs to the fullres (un-windowed) overlay
instead, not this one.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "runs/sweep_2gpc_1536/zmid8_dz4_az3_mh8_mfp3"
CHEN_CSV = "data/chen23_fiducial.csv"
OUT = "data/plots/amber_window_vs_ours_vs_chen23_fiducial.png"

gate = np.load(os.path.join(RUN, "amber_gate.npz"))
ells = gate["ells"]
Dl_ours = gate["D_compute_cell"]
Dl_amber_window = gate["D_amber_window"]

chen = np.loadtxt(CHEN_CSV, delimiter=",", skiprows=1)
l_chen, Dl_chen = chen[:, 0], chen[:, 1]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ells, Dl_amber_window, color="C0", lw=2, label="AMBER native, SAME window (z=7.25-14.25)")
ax.plot(ells, Dl_ours, "s--", color="C1", ms=5, lw=1.5, label="Our D_coeval/Limber, same window")
ax.plot(l_chen, Dl_chen, "o-", color="k", ms=4, label="Chen+23 fiducial (digitized, full/un-windowed)")
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$D_\ell\ [\mu K^2]$")
ax.set_xlim(0, 8500)
ax.legend()
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
print(f"AMBER window: {Dl_amber_window.min():.4g} - {Dl_amber_window.max():.4g} muK^2")
print(f"Our method:   {Dl_ours.min():.4g} - {Dl_ours.max():.4g} muK^2")
print(f"max|ratio-1| across full ell range: {np.max(np.abs(Dl_ours/Dl_amber_window - 1)):.4%}")
