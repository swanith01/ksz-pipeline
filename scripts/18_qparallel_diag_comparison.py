#!/usr/bin/env python
"""
Script 18: does uncancelled q_parallel explain P_diag's low-ell divergence?

MOTIVATION: script 17's fiducial run shows stitched P_diag (grouped, per-
snapshot auto-power only) diverging sharply from coeval-direct below
ell~600-800, despite good agreement from the peak through the tail.

HYPOTHESIS: coeval-direct algebraically projects q_parallel OUT of the
momentum field in 3D k-space, per snapshot, before the Limber sum ever
runs. Stitched never does this -- it relies ENTIRELY on q_parallel
cancelling through summation along the line of sight (the oscillating
phase averaging out over a long path). That cancellation is inherently
a CROSS-slice effect (see the coherence_decomposition theory note) --
it can only happen through how different slices combine. P_diag treats
each snapshot's slice as standing alone, with NO mechanism for that
cancellation to occur. So P_diag should carry q_parallel's full,
UNCANCELLED contribution -- and q_parallel's own power is known to be
strongest at low ell, fading at high ell (see the reference comparison
figure discussed earlier this session). That predicts exactly the
observed shape.

TEST: compare P_diag against (direct + q_parallel), not against direct
alone. If the combined curve tracks P_diag much more closely at low ell
than direct alone does, that's quantitative support for the hypothesis.

METHOD: momentum.py's qparallel_power() was written earlier this session
but never wired into coeval_sweep.py's cached pipeline (which only calls
qperp_power). This script computes it fresh, at the SAME fiducial
BOX_LEN/HII_DIM/z_window as script 14/17, reusing cached coeval boxes
(cache-hit expected, no new py21cmfast simulation).

DELIBERATE REUSE, NOT DUPLICATION: limber.py's compute_cell() is fed
q_parallel's P(k)/Pstd under the SAME dict keys ('Pqperp'/'Pstd') it
normally expects for q_perp -- compute_cell's Limber-integration math
doesn't care what the power spectrum physically represents, so this
gets q_parallel's D_ell through the EXACT same trusted, tested pipeline
(same tau, same window, same ne0, same error propagation) rather than
re-deriving the Limber sum a second time. Not a bug, not a mislabeling
of physical quantities in the output -- just internal reuse of a
generic function.

Usage
-----
    python scripts/18_qparallel_diag_comparison.py --config configs/fiducial.yaml
"""
import argparse
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ksz_pipeline.coeval.fields import run_coeval_fields
from ksz_pipeline.coeval.momentum import qparallel_power
from ksz_pipeline.coeval.limber import compute_cell, _interp_loglog
from ksz_pipeline.utils.constants import ne0_cgs


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sim_cfg = cfg['21cmfast']
    BOX_LEN = sim_cfg['BOX_LEN']
    HII_DIM = sim_cfg['HII_DIM_coeval']
    cache_dir = cfg['data']['cache_dir']
    out_dir  = cfg['data']['output_dir'].rstrip('/')
    plot_dir = cfg['data']['plot_dir'].rstrip('/')

    print(f"q_parallel low-ell test -- BOX_LEN={BOX_LEN} Mpc, HII_DIM={HII_DIM}\n")

    # ---- load the matched window/z-set and direct curve from script 14,
    # and P_diag from script 17 -- reused, not recomputed ----
    closure_path = f"{out_dir}/closure_test.npz"
    coherence_path = f"{out_dir}/coherence_decomposition_fiducial.npz"
    for p in (closure_path, coherence_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found -- run the corresponding script first.")
    closure = np.load(closure_path)
    coherence = np.load(coherence_path)

    ZS_win = [float(z) for z in closure['z_window']]
    ell_direct, Dl_direct = closure['ell_direct'], closure['Dl_direct']
    ell_diag, Dl_diag = coherence['ell_dec'], coherence['Dl_diag']
    print(f"Loaded {len(ZS_win)} windowed z's from closure_test.npz, "
          f"P_diag from coherence_decomposition_fiducial.npz\n")

    # ================================================================
    # compute q_parallel power per snapshot -- fresh, cache-hit expected
    # ================================================================
    print("Computing P_qparallel per snapshot (cache-hit expected on boxes)...")
    results_qpar = {}
    for z in ZS_win:
        delta, xH, vx, vy, vz = run_coeval_fields(
            z, HII_DIM, BOX_LEN, cache_dir,
            N_THREADS=sim_cfg['N_THREADS'], random_seed=sim_cfg['random_seed'])
        k_qpar, P_qpar, P_std = qparallel_power(delta, xH, vx, vy, vz, BOX_LEN)
        # Reused key names 'Pqperp'/'Pstd' -- see module docstring for why
        # this is deliberate reuse of compute_cell, not a mislabeling.
        results_qpar[z] = dict(k=k_qpar, Pqperp=P_qpar, Pstd=P_std,
                                xH_mean=float(xH.mean()))
        print(f"  z={z:.2f}  P_qparallel computed", flush=True)

    ell_qpar, Dl_qpar, sigma_qpar, *_ = compute_cell(results_qpar, ne0=ne0_cgs())
    print(f"\nq_parallel-only D_3000 = {float(np.interp(3000, ell_qpar, Dl_qpar)):.4g} uK^2")

    # ================================================================
    # combined direct + q_parallel, on a common ell grid
    # ================================================================
    lo = max(ell_direct.min(), ell_qpar.min())
    hi = min(ell_direct.max(), ell_qpar.max())
    ell_common = np.logspace(np.log10(lo), np.log10(hi), 60)
    Dl_direct_i = _interp_loglog(ell_common, ell_direct, Dl_direct)
    Dl_qpar_i   = _interp_loglog(ell_common, ell_qpar, Dl_qpar)
    Dl_combined = Dl_direct_i + Dl_qpar_i

    Dl_diag_i = _interp_loglog(ell_common, ell_diag, Dl_diag)

    # ================================================================
    # THE TEST: is P_diag closer to (direct+qpar) than to direct alone?
    # ================================================================
    resid_direct   = np.abs(Dl_diag_i - Dl_direct_i) / Dl_direct_i
    resid_combined = np.abs(Dl_diag_i - Dl_combined) / Dl_combined

    print(f"\n{'ell':>8} {'P_diag':>10} {'direct':>10} {'direct+qpar':>12} "
          f"{'|resid| direct':>15} {'|resid| combined':>17}")
    for i in range(0, len(ell_common), 5):
        print(f"{ell_common[i]:>8.0f} {Dl_diag_i[i]:>10.3g} {Dl_direct_i[i]:>10.3g} "
              f"{Dl_combined[i]:>12.3g} {resid_direct[i]:>15.1%} {resid_combined[i]:>17.1%}")

    low_ell_mask = ell_common < 800
    mean_resid_direct_low   = np.mean(resid_direct[low_ell_mask])
    mean_resid_combined_low = np.mean(resid_combined[low_ell_mask])
    print(f"\nMean |residual| at ell<800:")
    print(f"  P_diag vs direct alone:      {mean_resid_direct_low:.1%}")
    print(f"  P_diag vs direct+q_parallel: {mean_resid_combined_low:.1%}")
    if mean_resid_combined_low < mean_resid_direct_low * 0.7:
        print(">>> Substantial improvement -- supports the q_parallel-cancellation "
              "hypothesis for the low-ell divergence.")
    elif mean_resid_combined_low < mean_resid_direct_low:
        print(">>> Modest improvement -- q_parallel may be A contributor, likely not the whole story.")
    else:
        print(">>> No improvement -- q_parallel does not explain the low-ell divergence; "
              "look elsewhere.")

    # ================================================================
    # plot + save
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ell_common, Dl_diag_i, color='tab:blue', lw=2, ls='--', label='stitched P_diag')
    ax.plot(ell_common, Dl_direct_i, color='k', lw=2, label='direct (q_perp only)')
    ax.plot(ell_common, Dl_combined, color='tab:red', lw=2, ls=':', label='direct + q_parallel')
    ax.plot(ell_common, Dl_qpar_i, color='gray', lw=1.2, ls='-.', label='q_parallel only')
    ax.axvline(800, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$'); ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')
    ax.set_title('Does q_parallel explain P_diag\'s low-ell excess?')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plot_path = f"{plot_dir}/qparallel_diag_comparison.png"
    fig.savefig(plot_path, dpi=130, bbox_inches='tight')
    print(f"\nSaved -> {plot_path}")

    np.savez(f"{out_dir}/qparallel_diag_comparison.npz",
              ell_common=ell_common, Dl_diag=Dl_diag_i, Dl_direct=Dl_direct_i,
              Dl_combined=Dl_combined, Dl_qparallel=Dl_qpar_i,
              resid_direct=resid_direct, resid_combined=resid_combined)
    print(f"Saved -> {out_dir}/qparallel_diag_comparison.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    args = parser.parse_args()
    main(args.config)
