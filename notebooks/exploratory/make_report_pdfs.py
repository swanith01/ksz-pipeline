# %% [markdown]
# # Regenerate key plots as PDF for the LaTeX report
#
# Pure local replotting from .npz files already on disk -- NO cluster
# needed. Run this from notebooks/exploratory/.
#
# If any file is missing, this prints the exact scp command to fetch it
# and skips that one plot rather than crashing.

# %%
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRODUCTS_DIR = "../../data/products"
PLOTS_DIR = "../../data/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def _loglog_interp(xq, xp, fp):
    xp, fp = np.asarray(xp), np.asarray(fp)
    m = (xp > 0) & (fp > 0)
    lx, lf = np.log(xp[m]), np.log(fp[m])
    lq = np.log(np.clip(xq, xp[m].min(), xp[m].max()))
    return np.exp(np.interp(lq, lx, lf))


def need(path, scp_hint):
    if not os.path.exists(path):
        print(f"SKIPPED (missing): {path}\n  -> scp swarm:~/ksz-pipeline/{scp_hint} "
              f"{os.path.dirname(path)}/\n")
        return False
    return True


# %% [markdown]
# ## 1. Box periodicity: Delta-chi/L_box overlay

# %%
p_fid = f"{PRODUCTS_DIR}/coherence_decomposition_fiducial.npz"
p_400 = f"{PRODUCTS_DIR}/coherence_decomposition_box400.npz"
if need(p_fid, "data/products/coherence_decomposition_fiducial.npz") and \
   need(p_400, "data/products/coherence_decomposition_box400.npz"):
    fid = np.load(p_fid)
    b400 = np.load(p_400)
    L_fid, L_400 = 800.0, float(b400['box_len'])

    fig, ax = plt.subplots(figsize=(8, 6))
    for d, L, color, label in [(fid, L_fid, 'tab:blue', 'fiducial'),
                                 (b400, L_400, 'tab:orange', '400 Mpc')]:
        x = d['dchi_centers'] / L
        y = d['cross_mean']
        yerr = d['cross_std'] / np.sqrt(np.maximum(d['n_pairs'], 1))
        mask = d['n_pairs'] > 0
        ax.errorbar(x[mask], y[mask], yerr=yerr[mask], fmt='o-', color=color,
                     capsize=3, label=f'{label} ({L:.0f} Mpc)')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    for n in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        ax.axvline(n, color='gray', lw=0.5, ls=':', alpha=0.5)
    ax.set_xlabel(r'$|\Delta\chi| / L_{\rm box}$')
    ax.set_ylabel('mean pairwise cross-power')
    ax.set_title('P_off vs Delta-chi/L_box: box repetition test')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/dchi_over_Lbox_overlay.pdf", bbox_inches='tight')
    print(f"Saved -> {PLOTS_DIR}/dchi_over_Lbox_overlay.pdf")

# %% [markdown]
# ## 2. Low-ell sample-variance floor: ell/ell_min overlay

# %%
if os.path.exists(p_fid) and os.path.exists(p_400):
    from ksz_pipeline.utils.angular import ell_min_for_box
    chi_eff = float(fid['chi_eff'])
    ell_min_fid = ell_min_for_box(chi_eff, L_fid)
    ell_min_400 = ell_min_for_box(chi_eff, L_400)

    fig, ax = plt.subplots(figsize=(8, 6))
    for d, ell_min, color, label in [(fid, ell_min_fid, 'tab:blue', f'{L_fid:.0f} Mpc'),
                                        (b400, ell_min_400, 'tab:orange', f'{L_400:.0f} Mpc')]:
        ell_diag = d['ell_dec']
        Dl_diag = d['Dl_diag']
        Dl_direct_i = _loglog_interp(ell_diag, d['ell_direct'], d['Dl_direct'])
        ratio = Dl_diag / Dl_direct_i
        x = ell_diag / ell_min
        ax.plot(x, ratio, 'o-', color=color, ms=4, label=label)
    ax.axhline(1.0, color='k', lw=1, ls='--', alpha=0.7, label='P_diag = direct')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\ell / \ell_{\rm min}$ (own box)')
    ax.set_ylabel('P_diag / direct')
    ax.set_title('Low-ell excess, normalized by each box\'s own sample-variance floor')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/ell_over_ellmin_overlay.pdf", bbox_inches='tight')
    print(f"Saved -> {PLOTS_DIR}/ell_over_ellmin_overlay.pdf")

# %% [markdown]
# ## 3. P_diag radial-grouping plateau (scalar D_3000 vs n_groups)

# %%
p_grp = f"{PRODUCTS_DIR}/pdiag_grouping_convergence.npz"
if need(p_grp, "data/products/pdiag_grouping_convergence.npz"):
    d = np.load(p_grp)
    n_groups_list = [int(n) for n in d['n_groups']]
    d3000_direct = float(d['d3000_direct'])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(n_groups_list, d['d3000_diag'], 'o-', color='tab:blue', label='P_diag D_3000')
    ax.axhline(d3000_direct, color='k', ls='--', lw=1.5, label=f'direct D_3000={d3000_direct:.3g}')
    ax.axvspan(15, 40, color='gray', alpha=0.15, label='range around n=26')
    ax.axvline(26, color='tab:orange', lw=1, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel('number of radial groups')
    ax.set_ylabel(r'$D_{3000}$ [$\mu$K$^2$]')
    ax.set_title('P_diag vs radial grouping: is n=26 a robust choice?')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/pdiag_grouping_convergence.pdf", bbox_inches='tight')
    print(f"Saved -> {PLOTS_DIR}/pdiag_grouping_convergence.pdf")

    plateau_range = [n for n in n_groups_list if 15 <= n <= 40 and f'ell_n{n}' in d]
    if plateau_range:
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.cm.viridis
        for i, n in enumerate(plateau_range):
            color = cmap(i / max(len(plateau_range) - 1, 1))
            ax.plot(d[f'ell_n{n}'], d[f'Dl_diag_n{n}'], color=color, lw=1.8, label=f'n={n}')
        ax.axhline(d3000_direct, color='k', ls='--', lw=1.2, alpha=0.6)
        ax.axvline(3000, color='gray', ls=':', lw=0.8, alpha=0.6)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$'); ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$] (P_diag)')
        ax.set_title('P_diag full spectrum, overlaid across the n_groups=[15,40] plateau')
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        fig.savefig(f"{PLOTS_DIR}/pdiag_plateau_overlay.pdf", bbox_inches='tight')
        print(f"Saved -> {PLOTS_DIR}/pdiag_plateau_overlay.pdf")

# %% [markdown]
# ## 4. Coherence decomposition: P_diag vs direct (fiducial)

# %%
if os.path.exists(p_fid):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fid['ell_direct'], fid['Dl_direct'], color='k', lw=2, label='coeval-direct')
    ax.plot(fid['ell_dec'], fid['Dl_diag'], color='tab:blue', lw=2, ls='--', label='stitched P_diag (grouped)')
    ax.plot(fid['ell_dec'], fid['Dl_total'], color='tab:red', lw=1.5, label='stitched P_total')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$'); ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')
    ax.set_title('P_diag vs direct, fiducial resolution')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/coherence_decomposition_fiducial.pdf", bbox_inches='tight')
    print(f"Saved -> {PLOTS_DIR}/coherence_decomposition_fiducial.pdf")

# %% [markdown]
# ## 5. q_perp cross-z vs stitched P_off

# %%
p_qperp = f"{PRODUCTS_DIR}/qperp_cross_z.npz"
if need(p_qperp, "data/products/qperp_cross_z.npz") and os.path.exists(p_fid):
    qc = np.load(p_qperp)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(qc['ell_qperp'], qc['Dl_qperp_cross'], 'o-', color='tab:green', ms=4,
             label='q_perp cross-z (periodicity-free)')
    ax.plot(fid['ell_dec'], fid['Dl_off'], 's-', color='tab:blue', ms=4,
             label='stitched P_off (periodicity + Limber-failure)')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\ell$'); ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')
    ax.set_title('q_perp cross-z (periodicity-free) vs stitched P_off')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/qperp_cross_z.pdf", bbox_inches='tight')
    print(f"Saved -> {PLOTS_DIR}/qperp_cross_z.pdf")

print("\nDone. Note: closure_patchy_ksz_vs_limits.pdf and "
      "closure_total_ksz_vs_observations.pdf are ALREADY pdf -- no action needed.")
