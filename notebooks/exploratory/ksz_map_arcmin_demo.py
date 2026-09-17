# %% [markdown]
# # kSZ map in arcmin, using the angular.py utility
#
# UPDATED: now uses the real 512x512, windowed [z_lo,z_hi], chi_eff-
# consistent fiducial map (ksz_map_windowed_fiducial.npy, saved by
# script 14 after the 2026-08 patch) -- NOT the older 128^3, unwindowed
# ksz_map_lightcone.npy from earlier in the session.

# %%
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ksz_pipeline.utils.angular import plot_map_arcmin, mpc_to_arcmin, ell_min_for_box

PRODUCTS_DIR = "../../data/products"
PLOTS_DIR = "../../data/plots"

chi_eff = 8504.0  # from closure_test.npz
box_len_mpc = 800.0

ksz_map = np.load(f"{PRODUCTS_DIR}/ksz_map_windowed_fiducial.npy")
print(f"Map shape: {ksz_map.shape}")

extent_arcmin = mpc_to_arcmin(box_len_mpc, chi_eff)
print(f"{box_len_mpc} Mpc box at chi_eff={chi_eff} Mpc -> {extent_arcmin:.1f} arcmin "
      f"({extent_arcmin/60:.2f} deg) on a side")

fig, ax = plt.subplots(figsize=(7, 6))
im = plot_map_arcmin(ax, ksz_map, box_len_mpc, chi_eff)
plt.colorbar(im, label=r'$\Delta T/T|_{\rm kSZ}$')
ax.set_title(f"kSZ map, fiducial 512x512, windowed z=[4.50,18.00]\n"
             f"chi_eff={chi_eff:.0f} Mpc")
plt.tight_layout()
fig.savefig(f"{PLOTS_DIR}/ksz_map_arcmin_demo.png", dpi=140, bbox_inches='tight')
print(f"Saved -> {PLOTS_DIR}/ksz_map_arcmin_demo.png")
