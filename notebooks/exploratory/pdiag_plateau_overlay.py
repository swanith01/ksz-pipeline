# %% [markdown]
# # P_diag full D_ell curves, overlaid across the n_groups=[15,40] plateau
#
# The scalar D_3000 plateau (1.3% spread) already confirmed n=26 isn't a
# fragile, coincidental choice -- this shows whether the FULL spectrum
# shape agrees across that same range, not just one point on it.

# %%
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRODUCTS_DIR = "../../data/products"
PLOTS_DIR = "../../data/plots"

d = np.load(f"{PRODUCTS_DIR}/pdiag_grouping_convergence.npz")
n_groups_all = [int(n) for n in d['n_groups']]
d3000_direct = float(d['d3000_direct'])

plateau_range = [n for n in n_groups_all if 15 <= n <= 40]
print(f"n_groups in plateau range: {plateau_range}")

fig, ax = plt.subplots(figsize=(8, 6))
cmap = plt.cm.viridis
for i, n in enumerate(plateau_range):
    ell = d[f'ell_n{n}']
    Dl_diag = d[f'Dl_diag_n{n}']
    color = cmap(i / max(len(plateau_range) - 1, 1))
    ax.plot(ell, Dl_diag, color=color, lw=1.8, label=f'n={n}')

ax.axhline(d3000_direct, color='k', ls='--', lw=1.2, alpha=0.6)
ax.axvline(3000, color='gray', ls=':', lw=0.8, alpha=0.6)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$] (P_diag)')
ax.set_title('P_diag full spectrum, overlaid across the n_groups=[15,40] plateau')
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plot_path = f"{PLOTS_DIR}/pdiag_plateau_overlay.png"
fig.savefig(plot_path, dpi=140, bbox_inches='tight')
print(f"Saved -> {plot_path}")
