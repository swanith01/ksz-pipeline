# %% [markdown]
# # Is P_diag's low-ell rise just each box's own sample-variance floor?
#
# Zero-padding turned out to be the wrong lever here: each snapshot's
# transverse (x,y) plane already comes from a genuinely periodic 21cmFAST
# box, so there's no fake FFT seam to fix -- zero-padding a field that's
# already correctly periodic doesn't add real large-scale information,
# it just interpolates onto a denser k-grid.
#
# The more likely explanation: near a box's fundamental mode (the
# largest scale it can represent at all), there simply aren't many
# independent Fourier modes to average over. This is an UNAVOIDABLE
# property of any finite box, nothing to do with the FFT or a coding
# bug. It makes a specific, checkable prediction: EACH box size has its
# OWN "floor" (ell_min = 2*pi*chi_eff/L_box), and low-ell noise/excess
# should track that floor, not appear at some fixed universal ell.
#
# NO NEW COMPUTE -- both coherence_decomposition npz files already exist
# locally. Pure post-processing.

# %%
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ksz_pipeline.utils.angular import ell_min_for_box

PRODUCTS_DIR = "../../data/products"
PLOTS_DIR = "../../data/plots"

fid = np.load(f"{PRODUCTS_DIR}/coherence_decomposition_fiducial.npz")
b400 = np.load(f"{PRODUCTS_DIR}/coherence_decomposition_box400.npz")

# fiducial predates the box_len save key -- known value, hardcoded
L_fid = 800.0
L_400 = float(b400['box_len'])
chi_eff = float(fid['chi_eff'])  # same value reused for both runs, by design

ell_min_fid = ell_min_for_box(chi_eff, L_fid)
ell_min_400 = ell_min_for_box(chi_eff, L_400)
print(f"chi_eff = {chi_eff:.1f} Mpc (same for both runs)")
print(f"fiducial (L={L_fid:.0f} Mpc): ell_min = {ell_min_fid:.1f}")
print(f"box400   (L={L_400:.0f} Mpc): ell_min = {ell_min_400:.1f}")
print(f"ratio: {ell_min_400/ell_min_fid:.2f} (expect ~2.0, since L halved)\n")

fig, ax = plt.subplots(figsize=(9, 6.5))

ax.plot(fid['ell_dec'], fid['Dl_diag'], color='tab:blue', lw=2, ls='--',
        label=f'P_diag, {L_fid:.0f} Mpc')
ax.plot(fid['ell_direct'], fid['Dl_direct'], color='tab:blue', lw=2, alpha=0.5,
        label=f'direct, {L_fid:.0f} Mpc')
ax.plot(b400['ell_dec'], b400['Dl_diag'], color='tab:orange', lw=2, ls='--',
        label=f'P_diag, {L_400:.0f} Mpc')
ax.plot(b400['ell_direct'], b400['Dl_direct'], color='tab:orange', lw=2, alpha=0.5,
        label=f'direct, {L_400:.0f} Mpc')

ax.axvline(ell_min_fid, color='tab:blue', lw=1.2, ls=':', alpha=0.8)
ax.axvline(ell_min_400, color='tab:orange', lw=1.2, ls=':', alpha=0.8)
ax.text(ell_min_fid, 1, f'  ell_min\n  ({L_fid:.0f} Mpc)',
        color='tab:blue', fontsize=8, va='bottom')
ax.text(ell_min_400, 1, f'  ell_min\n  ({L_400:.0f} Mpc)',
        color='tab:orange', fontsize=8, va='bottom')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\ell$'); ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')
ax.set_title("Does each box's low-ell excess start at ITS OWN sample-variance floor?")
ax.legend(fontsize=9, loc='lower left')
plt.tight_layout()
fig.savefig(f"{PLOTS_DIR}/low_ell_mode_counting_check.png", dpi=140, bbox_inches='tight')
print(f"Saved -> {PLOTS_DIR}/low_ell_mode_counting_check.png")

print("\nRead this as: if P_diag's low-ell rise/excess in EACH curve begins near "
      "THAT box's own dotted ell_min line (not at a fixed ell shared by both), "
      "that supports 'unavoidable sample variance from a finite box', not a bug "
      "or leftover periodicity. If both curves' excess starts at the SAME ell "
      "regardless of box size, that argues against pure mode-counting and points "
      "back toward something else (periodicity or q_parallel) extending to low ell too.")
