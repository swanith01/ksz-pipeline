"""
Coeval-box convergence sweeps (box-size and resolution) for both the
direct/Cain-style P_qperp measurement and the Georgiev Eq.10
reconstruction.

One sweep function handles both convergence types: box-size and
resolution differ only in which (BOX_LEN, HII_DIM) pairs are passed in,
not in the sweep logic itself. Callers build that list; see
scripts/04_convergence_coeval.py.

Reuses, unchanged: coeval.fields.run_coeval_fields, coeval.momentum.
qperp_power, coeval.pee_pvv_pev.measure_pee_pvv_pev, coeval.
georgiev_convolution.qperp_from_pee_pvv_pev, coeval.limber.compute_cell.

Adapted from this project's older ksz_boxsize_convergence.py /
ksz_resolution_convergence.py (different, pre-migration repo).
Deliberately NOT reusing their own compute_Dell -- it predates this
session's tau0 (missing z=0..z_min optical depth), helium (HeII/HeIII),
and xH_mean-based patchy filter (they hardcoded z>=5.0 instead) fixes.
Uses coeval.limber.compute_cell instead, which already has all three.
"""

import os
import pickle

import numpy as np

# run_coeval_fields is imported lazily, inside run_one_config, not here --
# it pulls in py21cmfast, and everything else in this module (caching,
# Georgiev reconstruction, compute_cell wiring) has no reason to require
# that just to be imported/tested.
from ..coeval.momentum import qperp_power
from ..coeval.pee_pvv_pev import measure_pee_pvv_pev
from ..coeval.georgiev_convolution import qperp_from_pee_pvv_pev
from ..coeval.limber import compute_cell


def _loglog_interp(xq, xp, fp):
    xp, fp = np.asarray(xp), np.asarray(fp)
    m = (xp > 0) & (fp > 0)
    lx, lf = np.log(xp[m]), np.log(fp[m])
    lq = np.log(np.clip(xq, xp[m].min(), xp[m].max()))
    return np.exp(np.interp(lq, lx, lf))


def _georgiev_reconstruct(entry):
    """Build the Eq.10 reconstruction for one cached per-z entry."""
    Pee_f = lambda k: _loglog_interp(k, entry['k_pee'], entry['Pee'])
    Pvv_f = lambda k: _loglog_interp(k, entry['k_pee'], entry['Pvv'])
    Pev_f = lambda k: np.interp(k, entry['k_pee'], entry['Pev'])
    # Bound k' to the measured range (with padding), not
    # georgiev_convolution's generic default -- see that module's
    # docstring on why an unbounded kprime_max silently breaks with
    # interpolator inputs.
    conv_kwargs = dict(kprime_min=float(np.min(entry['k_pee'])) * 0.5,
                        kprime_max=float(np.max(entry['k_pee'])) * 2.0)
    return qperp_from_pee_pvv_pev(entry['k'], Pee_f, Pvv_f, Pev_f, **conv_kwargs)


def run_one_config(BOX_LEN, HII_DIM, z_snapshots, cache_dir, tag, force=False,
                    N_THREADS=None, random_seed=None):
    """
    Compute direct (Cain) and Georgiev-reconstructed D_ell for one
    (BOX_LEN, HII_DIM) configuration. Caches per-redshift Pee/Pvv/Pev/
    Pqperp results under cache_dir/qperp_{tag}.pkl, checkpointed after
    every redshift (same pattern as 02_make_ksz_coeval_boxes.py), so an
    interrupted sweep resumes instead of restarting.

    Parameters
    ----------
    BOX_LEN, HII_DIM : float, int
    z_snapshots       : sequence of float
    cache_dir         : str, py21cmfast cache AND this function's own
                        pickle cache both live under here
    tag               : str, unique label for this configuration (used
                        in the cache filename -- e.g. "box400_N64" or
                        "res128"); must be unique across a sweep or
                        configurations will silently share a cache file
    force             : bool, recompute even if a matching pickle cache
                        exists
    N_THREADS         : int, optional -- see coeval/fields.py; pass
                        explicitly (e.g. config's 21cmfast.N_THREADS)
                        for anything beyond quick interactive testing --
                        confirmed 14Jul2026 that omitting this silently
                        runs single-threaded regardless of cores requested

    Returns
    -------
    dict: ells_direct, Dl_direct, D3000_direct,
          ells_georgiev, Dl_georgiev, D3000_georgiev,
          ratios (dict z -> direct/reconstructed ratio array, keys are
              exactly z_snapshots, not the full cache),
          results (the FULL cache dict, which may contain more
              redshifts than z_snapshots if this cache_path/tag has
              been shared across calls with different z_snapshots --
              e.g. dz_sweep.py deliberately does this. Use for
              diagnostics like plotting P(k) at any cached z),
          results_subset (exactly the z_snapshots entries, i.e. what
              was actually used to compute Dl_direct/Dl_georgiev above
              -- use this, not results, if you need to know precisely
              what went into the D_ell calculation)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"qperp_{tag}.pkl")

    if not force and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            results = pickle.load(f)
    else:
        results = {}

    missing = [z for z in z_snapshots if z not in results]
    if missing:
        from ..coeval.fields import run_coeval_fields
    for z in missing:
        delta, xH, vx, vy, vz = run_coeval_fields(
            z, HII_DIM, BOX_LEN, cache_dir, N_THREADS=N_THREADS,
            random_seed=random_seed)
        k_q, P_q, P_std = qperp_power(delta, xH, vx, vy, vz, BOX_LEN)
        k_pee, Pee, Pvv, Pev = measure_pee_pvv_pev(delta, xH, vx, vy, vz, BOX_LEN)
        results[z] = dict(k=k_q, Pqperp=P_q, Pstd=P_std,
                           xH_mean=float(xH.mean()),
                           k_pee=k_pee, Pee=Pee, Pvv=Pvv, Pev=Pev)
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)

    # Filter to exactly the requested z_snapshots for THIS call's D_ell.
    # `results` (loaded from cache_path) may legitimately contain MORE
    # redshifts than z_snapshots asks for -- e.g. dz_sweep.py shares one
    # cache file across dz_x1/x2/x4 on purpose, so after the finest call
    # populates it, `results` holds the full fine grid regardless of
    # which subset a later call requested. Using `results` directly
    # here (the original bug) silently ignored z_snapshots entirely and
    # made every dz variant identical -- confirmed via quicktest
    # 8Jul2026, where dz_x1/x2/x4 came back bit-identical, which is what
    # exposed this.
    results_subset = {z: results[z] for z in z_snapshots}

    # -- direct (Cain), via the shared, already-fixed Limber code --
    ells_d, Dl_d, sigma_d, C_d, sigma_C_d, (ZS_d, tau_d), (_, xe_d) = compute_cell(results_subset)
    D3000_direct = float(np.interp(3000, ells_d, Dl_d)) if len(ells_d) else float('nan')

    # -- Georgiev Eq.10 reconstruction, same compute_cell for the Limber sum --
    results_g, ratios = {}, {}
    for z, entry in results_subset.items():
        if 'k_pee' not in entry:
            continue
        P_recon = _georgiev_reconstruct(entry)
        ratios[z] = entry['Pqperp'] / P_recon
        results_g[z] = dict(k=entry['k'], Pqperp=P_recon,
                             Pstd=np.full_like(P_recon, 1e-300),
                             xH_mean=entry['xH_mean'])

    if results_g:
        ells_g, Dl_g, sigma_g, C_g, sigma_C_g, (ZS_g, tau_g), (_, xe_g) = compute_cell(results_g)
        D3000_georgiev = float(np.interp(3000, ells_g, Dl_g)) if len(ells_g) else float('nan')
    else:
        ells_g, Dl_g, D3000_georgiev = np.array([]), np.array([]), float('nan')

    return dict(ells_direct=ells_d, Dl_direct=Dl_d, D3000_direct=D3000_direct,
                ells_georgiev=ells_g, Dl_georgiev=Dl_g, D3000_georgiev=D3000_georgiev,
                ratios=ratios, results=results, results_subset=results_subset)


def run_sweep(param_list, z_snapshots, cache_dir, force=False, N_THREADS=None,
              random_seed=None):
    """
    Run run_one_config for each configuration in param_list.

    Parameters
    ----------
    param_list   : list of (BOX_LEN, HII_DIM, tag) tuples
    z_snapshots  : sequence of float, same redshift grid for every
                   configuration in the sweep
    cache_dir    : str
    force        : bool
    N_THREADS    : int, optional -- see run_one_config

    Returns
    -------
    dict keyed by tag -> run_one_config's return dict
    """
    out = {}
    for BOX_LEN, HII_DIM, tag in param_list:
        dx = BOX_LEN / HII_DIM
        print(f"=== {tag}: BOX_LEN={BOX_LEN} Mpc, HII_DIM={HII_DIM} "
              f"(dx={dx:.3f} Mpc) ===", flush=True)
        out[tag] = run_one_config(BOX_LEN, HII_DIM, z_snapshots, cache_dir,
                                   tag, force=force, N_THREADS=N_THREADS,
                                   random_seed=random_seed)
        print(f"  D_3000 direct={out[tag]['D3000_direct']:.4g} uK^2  "
              f"georgiev={out[tag]['D3000_georgiev']:.4g} uK^2", flush=True)
    return out
