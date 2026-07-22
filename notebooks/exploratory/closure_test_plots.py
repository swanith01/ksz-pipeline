# %% [markdown]
# # Closure-test plots: matched-window direct vs. stitched
#
# Uses `closure_test.npz` (script 14) instead of the old separately-windowed
# `ksz_Dl_coeval.npz`/`ksz_Dl_stitched.npz` -- both curves here are on the
# SAME unified patchy z-window, which is the whole point of the closure test.
#
# Reuses Shaw et al. 2012 (homogeneous/late-time kSZ template) and
# Chaubal et al. 2026 (SPT-3G/ACT DR6 total-kSZ observational points) exactly
# as tabulated in Georgiev_comparision.py -- no new literature numbers here.
#
# NOTE ON TIMING: this only needs the WINDOWED Dl_direct/Dl_stitched arrays,
# which were already correctly computed in job 1684054 (the D_ell windowing
# fix landed before the chi_end_reionization label bug was even introduced).
# Job 1684234 is a re-run that only fixes the chi_end_reionization/midpoint
# LABELING -- it does not change these curves. Safe to plot now without
# waiting for 1684234 to clear the queue.
#
# No new science logic here, per the same "notebooks/exploratory/" convention
# as Georgiev_comparision.py -- only loads already-computed products and plots.

# %%
import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')  # headless-safe for login-node runs
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import NullFormatter

plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.linewidth': 1.3})

PRODUCTS_DIR = "../../data/products"
GEORGIEV_PICKLE = "../../data/external/georgiev/obs_vs_params_xe_ksz_data.pickle"
PLOTS_DIR = "../../data/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def loglog_interp(xq, xp, fp):
    xp, fp = np.asarray(xp), np.asarray(fp)
    m = (xp > 0) & (fp > 0)
    lx, lf = np.log(xp[m]), np.log(fp[m])
    lq = np.log(np.clip(xq, xp[m].min(), xp[m].max()))
    return np.exp(np.interp(lq, lx, lf))


def patchy_upper_limit(mean, sigma, hksz):
    return mean + 2 * sigma - hksz


# %%
# ---- load closure-test results (matched-window direct + stitched) ----
closure = np.load(f"{PRODUCTS_DIR}/closure_test.npz", allow_pickle=True)
ell_direct, Dl_direct   = closure['ell_direct'],   closure['Dl_direct']
sigma_Dl_direct         = closure['sigma_Dl_direct']
ell_stitched, Dl_stitched = closure['ell_stitched'], closure['Dl_stitched']
Dl_stitched_err        = closure['Dl_stitched_err']
z_lo, z_hi              = float(closure['z_lo']), float(closure['z_hi'])
chi_eff                 = float(closure['chi_eff'])

print(f"Closure-test matched window: z=[{z_lo:.2f}, {z_hi:.2f}]  (chi_eff={chi_eff:.1f} Mpc)")
print("direct   D_3000 =", float(np.interp(3000, ell_direct, Dl_direct)), "uK^2  (matched window)")
print("stitched D_3000 =", float(np.interp(3000, ell_stitched, Dl_stitched)), "uK^2  (matched window, chi_eff)")

# reionization history, for the xe(z) panel -- trimmed to the SAME window
# used for the kSZ curves above, so the two panels visually correspond
reion = np.load(f"{PRODUCTS_DIR}/coeval_reion.npz")

# %%
# ---- Georgiev et al. published parameter-study data (for zre-sweep context) ----
with open(GEORGIEV_PICKLE, 'rb') as f:
    geo = pickle.load(f)
zlin, lrange = geo['zlin'], geo['lrange']
paramnames, results, ntest = geo['paramnames'], geo['results'], geo['ntest']
idx_zre = paramnames.index('zre')

# %% [markdown]
# ## Plot 1: patchy kSZ (matched window) vs. observational 2-sigma upper limits

# %%
# Shaw et al. 2012 -- kept only for the effective-zrei printout context;
# full late-time addition happens in Plot 2.
ZREI_SHAW_FIDUCIAL = 10.0

# Chaubal et al. 2026, Table 5 + intro (same values as Georgiev_comparision.py)
spt_agora_mean, spt_agora_sigma = 3.96, 0.82
act_dr6_mean,   act_dr6_sigma   = 2.0,  0.9
spt_freeCIBSZ_mean  = np.array([2.1, 2.4, 2.72, 3.36, 2.3])
spt_freeCIBSZ_sigma = np.array([1.6, 1.0, 0.70, 0.90, 1.0])
hkSZ_high = 1.84
hkSZ_low  = 0.85

agora_UL         = patchy_upper_limit(spt_agora_mean, spt_agora_sigma, hkSZ_high)
agora_lowhksz_UL = patchy_upper_limit(spt_agora_mean, spt_agora_sigma, hkSZ_low)
freeCIBSZ_UL     = patchy_upper_limit(spt_freeCIBSZ_mean[1], spt_freeCIBSZ_sigma[1], hkSZ_high)
act_UL           = patchy_upper_limit(act_dr6_mean, act_dr6_sigma, hkSZ_low)

fig, (ax_ksz, ax_xe) = plt.subplots(1, 2, figsize=(14, 6))

cmap = mpl.colormaps['PuRd']
norm = colors.Normalize(vmin=geo['params_test'][:, idx_zre].min(),
                         vmax=geo['params_test'][:, idx_zre].max())

for u in range(ntest):
    theta = results[idx_zre]['theta'][u]
    ax_ksz.plot(lrange, results[idx_zre]['ksz'][u], color=cmap(0.35 + 0.55 * norm(theta[idx_zre])),
                lw=1.5, ls=':', label=f"Georgiev+24: zre={theta[idx_zre]:.3g}")

ax_ksz.plot(ell_direct, Dl_direct, color='k', lw=2.2, ls='-', label='ours: direct (matched window)')
ax_ksz.fill_between(ell_direct, Dl_direct - sigma_Dl_direct, Dl_direct + sigma_Dl_direct,
                     color='k', alpha=0.15)
ax_ksz.plot(ell_stitched, Dl_stitched, color='tab:blue', lw=2.0, ls='-', label='ours: stitched (matched window, chi_eff)')
ax_ksz.fill_between(ell_stitched, Dl_stitched - Dl_stitched_err, Dl_stitched + Dl_stitched_err,
                     color='tab:blue', alpha=0.2)

pts = [('SPT-3G Agora', agora_UL, 'orange'),
       ('SPT-3G Agora w/ low hkSZ', agora_lowhksz_UL, 'green'),
       ('SPT-3G free CIB+SZ', freeCIBSZ_UL, 'crimson'),
       ('ACT DR6', act_UL, 'tab:red')]
for label, val, col in pts:
    ax_ksz.annotate('', xy=(3000, val), xytext=(3000, val * 1.25),
                     arrowprops=dict(arrowstyle='-|>', color=col, lw=2))
    ax_ksz.plot([], [], color=col, marker='v', linestyle='none', label=f'{label} (2σ UL)')

ax_ksz.set_xscale('log'); ax_ksz.set_yscale('log')
ax_ksz.set_xlabel(r"Angular multipole $\ell$")
ax_ksz.set_ylabel(r"Patchy $\mathcal{D}_\ell^\mathrm{kSZ}$ [$\mu$K$^2$]")
ax_ksz.set_title(f"matched window z=[{z_lo:.2f}, {z_hi:.2f}]", fontsize=10)
ax_ksz.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, borderaxespad=0)

# ---- right: reionization histories, same window marked ----
for u in range(ntest):
    theta = results[idx_zre]['theta'][u]
    ax_xe.plot(zlin, results[idx_zre]['xe'][u], color=cmap(0.35 + 0.55 * norm(theta[idx_zre])),
               lw=1.5, ls=':', label=f"Georgiev+24: zre={theta[idx_zre]:.3g}")
ax_xe.plot(reion['z'], reion['xe'], color='tab:blue', lw=2.2, ls='-', label='ours (coeval, patchy range)')
ax_xe.axvspan(z_lo, z_hi, color='gray', alpha=0.15, label='closure-test matched window')

ax_xe.set_xlim(0, 20)
ax_xe.set_xlabel(r"Redshift $z$")
ax_xe.set_ylabel(r"$x_e(z)$")
ax_xe.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, borderaxespad=0)

fig.tight_layout()
fig.savefig(f"{PLOTS_DIR}/closure_patchy_ksz_vs_limits.pdf", bbox_inches='tight')
print(f"Saved -> {PLOTS_DIR}/closure_patchy_ksz_vs_limits.pdf")

# %% [markdown]
# ## Plot 2: total kSZ (patchy + Shaw homogeneous template) vs. Chaubal+26 points
#
# NOTE: georgiev_recon (our Eq.10 reconstruction) is NOT included here --
# it was never re-run on the closure test's unified window, so plotting it
# alongside these matched-window curves would silently reintroduce the
# window mismatch this whole exercise exists to remove.

# %%
shaw_ell = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
shaw_CSF = np.array([1.43, 2.00, 2.19, 2.27, 2.32, 2.36, 2.40, 2.44, 2.48, 2.52])
shaw_zrei_exp_CSF = np.array([0.63, 0.66, 0.64, 0.60, 0.55, 0.52, 0.48, 0.45, 0.42, 0.40])

# Effective zrei = z_lo, the closure-test's OWN matched window boundary --
# more defensible than the old ad hoc reion['z'].min(), since both methods
# now agree this is where the patchy regime starts.
zrei_ours = z_lo
print(f"Effective zrei (closure-test z_lo): {zrei_ours:.2f}")
print(f"(Shaw calibrated the zrei exponent over 8-12 -- {zrei_ours:.1f} is an extrapolation, treat as approximate)")

rescale_CSF = (zrei_ours / ZREI_SHAW_FIDUCIAL) ** shaw_zrei_exp_CSF
shaw_CSF_corrected = shaw_CSF * rescale_CSF

spt_ell = np.array([2000, 3000, 4000, 5000, 6000])
spt_freeCIB_G15SZ_mean  = np.array([1.7, 1.75, 1.67, 1.6, 1.6])
spt_freeCIB_G15SZ_sigma = np.array([1.3, 0.86, 0.96, 1.1, 1.3])

late_interp = lambda ell: loglog_interp(ell, shaw_ell, shaw_CSF_corrected)
ell_plot = np.geomspace(1000, 10000, 200)
total_direct   = loglog_interp(ell_plot, ell_direct, Dl_direct) + late_interp(ell_plot)
total_stitched = loglog_interp(ell_plot, ell_stitched, Dl_stitched) + late_interp(ell_plot)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(ell_plot, total_direct, color='k', lw=2.2, label='ours: direct (matched window) + late-time (CSF)')
ax.plot(ell_plot, total_stitched, color='tab:blue', lw=2.0, ls='--', label='ours: stitched (matched window) + late-time (CSF)')
ax.plot(shaw_ell, shaw_CSF_corrected, color='gray', lw=1.5, ls=':', label='late-time only (Shaw CSF, zrei-corrected)')

ax.errorbar(spt_ell, spt_freeCIB_G15SZ_mean, yerr=spt_freeCIB_G15SZ_sigma,
            fmt='x', color='seagreen', markersize=8, capsize=3, label='SPT-3G free CIB (Chaubal+26)')
ax.errorbar(spt_ell + 60, spt_freeCIBSZ_mean, yerr=spt_freeCIBSZ_sigma,
            fmt='o', color='darkgreen', markersize=6, capsize=3, mfc='none', label='SPT-3G free CIB+SZ (Chaubal+26)')
ax.errorbar([3000], [act_dr6_mean], yerr=[[act_dr6_sigma], [act_dr6_sigma]],
            fmt='^', color='tab:red', markersize=9, capsize=3, label='ACT DR6 (Louis/Beringue+25)')

ax.set_xscale('log')
xticks = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
ax.set_xticks(xticks)
ax.set_xticklabels([str(x) for x in xticks], rotation=60, ha='right')
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"Angular multipole $\ell$")
ax.set_ylabel(r"Total $\mathcal{D}_\ell^\mathrm{kSZ}$ [$\mu$K$^2$]")
ax.set_title(f"matched window z=[{z_lo:.2f}, {z_hi:.2f}], zrei_eff={zrei_ours:.2f}", fontsize=10)
ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
fig.tight_layout()
fig.savefig(f"{PLOTS_DIR}/closure_total_ksz_vs_observations.pdf", bbox_inches='tight')
print(f"Saved -> {PLOTS_DIR}/closure_total_ksz_vs_observations.pdf")
