# %% [markdown]
# # kSZ map in arcmin, using the angular.py utility
#
# CAVEAT ON WHICH MAP THIS IS: data/products/ksz_map_lightcone.npy predates
# this session's closure-test/matched-window/chi_eff work -- it's the
# original "z=4.0-20.0" full LOS range, NOT the windowed [z_lo,z_hi]
# map any recent script actually validated. Neither script 14 nor 17
# currently saves the raw 2D map array to disk (only Dl curves) -- if a
# genuinely up-to-date, windowed, chi_eff-consistent map is needed for a
# real figure, that requires adding a np.save() to one of those scripts
# and rerunning. This demo exists to show the arcmin conversion works
# correctly, not to present final results.

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

ksz_map = np.load(f"{PRODUCTS_DIR}/ksz_map_lightcone.npy")
print(f"Map shape: {ksz_map.shape}")

extent_arcmin = mpc_to_arcmin(box_len_mpc, chi_eff)
print(f"{box_len_mpc} Mpc box at chi_eff={chi_eff} Mpc -> {extent_arcmin:.1f} arcmin "
      f"({extent_arcmin/60:.2f} deg) on a side")

fig, ax = plt.subplots(figsize=(7, 6))
im = plot_map_arcmin(ax, ksz_map, box_len_mpc, chi_eff)
plt.colorbar(im, label=r'$\Delta T/T|_{\rm kSZ}$')
ax.set_title(f"kSZ map (OLDER full-range map, see caveat above)\n"
             f"chi_eff={chi_eff:.0f} Mpc")
plt.tight_layout()
fig.savefig(f"{PLOTS_DIR}/ksz_map_arcmin_demo.png", dpi=140, bbox_inches='tight')
print(f"Saved -> {PLOTS_DIR}/ksz_map_arcmin_demo.png")
