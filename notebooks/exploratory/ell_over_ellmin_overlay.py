# %% [markdown]
# # P_diag/direct vs ell/ell_min: sample-variance vs periodicity, ell-space
#
# Same logic as the Delta-chi/L_box periodicity overlay, applied to ell
# instead of radial separation. ell_min = each box's own smallest
# resolvable multipole (set by its fundamental Fourier mode). Plotting
# P_diag/direct against ell/ell_min puts both box sizes on a common
# "how many multiples of my own floor am I" axis.
#
# READ: if both boxes' excess (ratio well above 1) turns on and decays
# at roughly the SAME ell/ell_min value, that supports pure finite-box
# sample variance -- a generic property of ANY box, correctly scaled.
# If they DON'T line up even on this normalized axis, periodicity (or
# something else) is still contaminating low ell, not just sample
# variance.
#
# NO NEW COMPUTE -- reuses the same two npz files as the mode-counting
# check and the Delta-chi/L_box overlay.

# %%
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ksz_pipeline.utils.angular import ell_min_for_box

PRODUCTS_DIR = "../../data/products"
PLOTS_DIR = "../../data/plots"


def loglog_interp(xq, xp, fp):
    xp, fp = np.asarray(xp), np.asarray(fp)
    m = (xp > 0) & (fp > 0)
    lx, lf = np.log(xp[m]), np.log(fp[m])
    lq = np.log(np.clip(xq, xp[m].min(), xp[m].max()))
    return np.exp(np.interp(lq, lx, lf))


fid = np.load(f"{PRODUCTS_DIR}/coherence_decomposition_fiducial.npz")
b400 = np.load(f"{PRODUCTS_DIR}/coherence_decomposition_box400.npz")

L_fid = 800.0  # fiducial predates the box_len save key
L_400 = float(b400['box_len'])
chi_eff = float(fid['chi_eff'])

ell_min_fid = ell_min_for_box(chi_eff, L_fid)
ell_min_400 = ell_min_for_box(chi_eff, L_400)
print(f"ell_min: fiducial={ell_min_fid:.1f}  box400={ell_min_400:.1f}  "
      f"ratio={ell_min_400/ell_min_fid:.2f}\n")

fig, ax = plt.subplots(figsize=(8, 6))

for d, ell_min, color, label in [(fid, ell_min_fid, 'tab:blue', f'{L_fid:.0f} Mpc'),
                                    (b400, ell_min_400, 'tab:orange', f'{L_400:.0f} Mpc')]:
    ell_diag = d['ell_dec']
    Dl_diag = d['Dl_diag']
    Dl_direct_on_diag = loglog_interp(ell_diag, d['ell_direct'], d['Dl_direct'])
    ratio = Dl_diag / Dl_direct_on_diag
    x = ell_diag / ell_min
    ax.plot(x, ratio, 'o-', color=color, ms=4, label=label)
    print(f"{label}: ratio at x=ell/ell_min ~1-2 -> "
          f"{ratio[np.argmin(np.abs(x-1.5))]:.2f}")

ax.axhline(1.0, color='k', lw=1, ls='--', alpha=0.7, label='P_diag = direct')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\ell / \ell_{\rm min}$ (own box)')
ax.set_ylabel('P_diag / direct')
ax.set_title('Low-ell excess, normalized by each box\'s own sample-variance floor')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(f"{PLOTS_DIR}/ell_over_ellmin_overlay.png", dpi=140, bbox_inches='tight')
print(f"\nSaved -> {PLOTS_DIR}/ell_over_ellmin_overlay.png")
