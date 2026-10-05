#!/usr/bin/env python
"""
Script 17: coherence decomposition, FIDUCIAL resolution (with optional
box-size override for the periodicity-vs-physics test).

Scales script 16's quicktest logic up to full fiducial resolution
(800 Mpc / 512^3, all 29 z_snapshots), per Girish's 2026-08-14 note.

BOX-SIZE OVERRIDE (added 2026-08-17, after the fiducial run showed a
P_off spike at Delta-chi ~830 Mpc, suspiciously close to BOX_LEN=800):
--box-len/--hii-dim let you rerun at a DIFFERENT box size while keeping
cell size (dx) IDENTICAL to fiducial (so resolution isn't also changing
at the same time -- box size is isolated as the only varied axis). If
the Delta-chi bump moves proportionally with BOX_LEN, that's strong
evidence of a periodicity/box-reuse artifact, not real physics; if it
stays fixed in physical Mpc regardless of box size, that argues for
something real. chi_eff/z_lo/z_hi are reused from closure_test.npz
regardless of box size (these are LOS/redshift-only quantities,
independent of the transverse box size) -- but the coeval-direct
REFERENCE curve is recomputed fresh at the overridden box/resolution
(reusing coeval_sweep.run_one_config, same as script 16's quicktest),
since P_qperp genuinely depends on box size/resolution and
closure_test.npz's own Dl_direct is fiducial-specific.

THREE THINGS ADDED beyond script 16's quicktest, at fiducial defaults:
1. SNAPSHOT-LEVEL GROUPING (group_slices_by_snapshot) before the P_diag/
   P_off split -- "diagonal" now means what it means for coeval-direct
   (one unit per snapshot's full box depth), not per thin LOS pixel.
   Script 16's quicktest skipped this and got a 56.6% P_diag/direct
   mismatch that was very likely partly this granularity mismatch, not
   (only) missing physics -- see the accompanying theory note for why
   P_diag was never guaranteed to equal direct even with correct
   grouping.
2. PERIODICITY CONTROL (random_shift_slices) -- independent random
   cyclic shift per GROUPED slice. Preserves each slice's own power
   exactly (pure phase rotation), destroys any FIXED cross-slice
   alignment -- both genuine LOS q_parallel-cancellation correlation
   AND any periodicity/box-reuse artifact. A drop in P_off under
   shifting means the coherent excess IS alignment-dependent; it does
   NOT by itself distinguish real physics from a stitching artifact --
   both are alignment-dependent. ATON is the real discriminator if this
   control alone doesn't settle it.
3. MATCHED WINDOW + chi_eff, reused from closure_test.npz (script 14),
   same convention as scripts 05/10/13 -- avoids reintroducing the
   window-mismatch confound the closure test exists to remove. Also
   reuses closure_test.npz's OWN Dl_direct directly rather than
   recomputing coeval-direct fresh, so the comparison is against the
   exact trusted number.

MEMORY: peak ~20-50 GB (corrected estimate -- an earlier ~9.9 TB figure
was a units error, GB vs TB). Comfortably within the 125-515 GB node
memory seen on this cluster.

CACHING: uses the MAIN fiducial cache_dir directly (no subdirectory),
deliberately, so the already-built coeval boxes and stitched lightcone
are reused rather than resimulated. If N_THREADS is not actually
reaching py21cmfast under the hood (open, unconfirmed question from
earlier this session), this could still run single-threaded regardless
of config -- request generous walltime.

ADDED 23 Sep 2026 -- --source {coeval,native}: the SAME decomposition
and Delta-chi periodicity test, run on a lightcone from py21cmfast's own
run_lightcone() instead of coeval boxes stitched by hand. Only the ONE
call that builds `stitched` differs -- everything downstream (tau,
visibility, patchy mask, per-slice theta, the compute_ksz_map sanity
check, snapshot grouping, the unshifted/shifted decomposition, the
Delta-chi binning, the plot, the save) is untouched, because
native_lightcone.stitch_lightcone_native() matches
stitch_lightcone_from_coeval()'s output contract exactly (see that
module's docstring -- confirmed working end-to-end on script 20's
grouping-convergence test, 23 Sep 2026). Default is still 'coeval' --
existing behaviour and output files are byte-for-byte unchanged unless
--source native is passed explicitly.

ONE THING DELIBERATELY NOT PORTED: the non-fiducial-box-size branch's
"fresh coeval-direct reference" (coeval_sweep.run_one_config) has no
native equivalent -- there is no "native_sweep.run_one_config" and
building one is out of scope for what this test actually asks (does
native P_off show the same box-size-dependent signature the coeval path
showed, not does native match a resampled coeval-direct at arbitrary box
sizes). For --source native, the 'direct' reference is ALWAYS
closure_test.npz's fiducial-coeval Dl_direct regardless of box_len/
hii_dim overrides -- a rough shape anchor only, never a validated match,
same caveat already used in script 20 for the analogous case. Printed
and stated explicitly, not silently assumed.

VELOCITY CAVEAT (native path only): amplitude not independently
validated -- see native_lightcone.py's docstring. An overall wrong scale
factor rescales D_ell by that factor squared but cannot manufacture or
hide the Delta-chi SHAPE (where a periodicity bump sits, whether it
moves with box size) -- the thing this script's periodicity test
actually asks about.

ADDED (1Oct2026): astro_params/flag_options support, read from the
loaded config's 21cmfast.astro_params / 21cmfast.flag_options (absent
in fiducial.yaml -> both None -> every code path below behaves exactly
as before this change). For the Nikolic/Mesinger/Gorce (2023)
replication (configs/nikolic_mesinger.yaml), which uses a DIFFERENT
astrophysics model (21cmFAST's mass-dependent SFR/escape-fraction
model) at a DIFFERENT box/resolution than fiducial.yaml.

Two things this touches, both deliberate -- read before trusting a run:

1. is_fiducial_config's REUSE-closure_test.npz branch only checks
   whether BOX_LEN/HII_DIM match the LOADED CONFIG's own values (i.e.
   whether --box-len/--hii-dim were passed) -- it has no way to know
   the loaded config itself might use entirely different astrophysics
   than fiducial.yaml. Blindly trusting it here would compare a
   Nikolic-astro-params STITCHED curve against a HII_EFF_FACTOR-astro
   DIRECT reference from the original fiducial run -- same bug class as
   the ne0/N_THREADS defaults this pipeline already got bitten by.
   Fixed: a separate `astro_overridden` flag forces the FRESH-direct-
   reference branch whenever astro_params/flag_options are set in the
   config, regardless of box/dim override status. The "FIDUCIAL config"
   vs "BOX-SIZE OVERRIDE" print below still goes by the original
   box/dim-only check (unchanged, just means "no CLI override" -- not
   literally fiducial.yaml); the added astro_overridden print makes the
   actual physics being used unambiguous regardless of that wording.

2. run_one_config() (coeval_sweep.py) is called with astro_params/
   flag_options added to its kwargs whenever astro_overridden is True.
   NOT YET VERIFIED that run_one_config accepts these -- coeval_sweep.py
   wasn't available when this was written. Fails loudly (TypeError with
   an explicit message) rather than silently falling back to the wrong
   astrophysics model for the direct reference if it doesn't.

3. tag_suffix gets an extra "_nikolic" whenever astro_overridden, on
   top of the existing box-size suffix logic -- WITHOUT this, running
   configs/nikolic_mesinger.yaml with no CLI override would write to
   the exact same coherence_decomposition_fiducial.{png,npz} filenames
   as the original 800Mpc/512^3 fiducial run, silently clobbering it.

Usage
-----
    python scripts/17_coherence_decomposition_fiducial.py --config configs/fiducial.yaml
    python scripts/17_coherence_decomposition_fiducial.py --config configs/fiducial.yaml --source native

ADDED (1Oct2026, same day as astro_params): --random-seed and --skip-direct,
for building an ENSEMBLE of independent realizations (varying cosmic initial
conditions) rather than trusting a single draw -- motivated directly by
Nikolic et al. (2023)'s own Appendix A, which shows a 500 Mpc box's D_3000
has ~15% realization-to-realization scatter (f=1.27+/-0.19 calibration
factor, N=20 seeds) purely from missing large-scale power/cosmic variance at
that box size. A single run at one fixed seed can't be read against their
curve with any confidence -- an unlucky draw looks like a discrepancy that
isn't one, a lucky draw looks like agreement that isn't meaningful either.

--random-seed overrides ONLY the STITCHED lightcone's cosmic seed (passed to
stitch_lightcone_from_coeval / stitch_lightcone_native), NOT the direct
reference's seed, which always uses the config's own random_seed regardless
-- see --skip-direct below for why.

--skip-direct skips the coeval-direct reference computation entirely. The
direct reference depends only on BOX_LEN/HII_DIM/astro_params -- NOT on
wrap_cycle_seed or this run's own --random-seed override -- so one ensemble
of N realizations needs exactly ONE direct-reference computation shared
across all N, not N redundant ones. Also sidesteps needing coeval_sweep.py's
astro_params passthrough verified before an ensemble's STITCHED runs can
start -- run once without --skip-direct (after that passthrough exists) to
get the shared reference, then with it for the rest of the batch. With
--skip-direct, d3000_direct is saved as NaN, the direct curve is omitted
from the plot, and the |P_diag-direct|/direct commentary is skipped (it
would otherwise silently read as "reasonably matches", which is wrong --
NaN > 0.3 is False in Python, not an error, so this is guarded explicitly
rather than left to fail silently).
    python scripts/17_coherence_decomposition_fiducial.py --config configs/nikolic_mesinger.yaml
"""
import argparse
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ksz_pipeline.ksz.stitch_from_coeval import stitch_lightcone_from_coeval, build_los_z_grid
from ksz_pipeline.ksz.native_lightcone import stitch_lightcone_native
from ksz_pipeline.ksz.optical_depth import (compute_tau, compute_visibility,
                                             analytic_tau_below, compute_patchy_mask)
from ksz_pipeline.ksz.lightcone_integral import compute_ksz_map, ksz_map_to_Dl
from ksz_pipeline.ksz.coherence_decomposition import (compute_ksz_map_per_slice,
                                                       decompose_p_total_diag_off,
                                                       group_slices_by_snapshot,
                                                       random_shift_slices,
                                                       cross_power_by_dchi)
from ksz_pipeline.utils.constants import ne0_cgs, MPC_CM



def _plot_nikolic_band(ax):
    """
    Overlay the digitized Nikolic/Mesinger/Gorce (2023) Figure 1 kSZ band
    (data/external/nikolic_ksz_{lower,power,upper}.csv) on an ell-vs-D_ell
    axis -- added at the user's request once the Nikolic replication
    ensemble was running (1 Oct 2026). ILLUSTRATIVE ONLY: their curve is
    from a 1.5 Gpc/1050^3 box, ours from 500 Mpc/256^3 -- very different
    box size/resolution/sample variance, NOT a like-for-like overlay,
    labeled as such in the legend. Never raises -- missing CSVs just skip
    the overlay with a printed note, so this can't break a run that
    otherwise has nothing to do with these files.
    """
    nikolic_dir = 'data/external'
    try:
        lo  = np.loadtxt(f'{nikolic_dir}/nikolic_ksz_lower.csv', delimiter=',', skiprows=1)
        hi  = np.loadtxt(f'{nikolic_dir}/nikolic_ksz_upper.csv', delimiter=',', skiprows=1)
        mid = np.loadtxt(f'{nikolic_dir}/nikolic_ksz_power.csv', delimiter=',', skiprows=1)
    except OSError as e:
        print(f"NOTE: could not load digitized Nikolic kSZ band for overplotting "
              f"({e}) -- expected data/external/nikolic_ksz_{{lower,power,upper}}.csv. "
              f"Skipping overlay.")
        return
    ell_lo, Dl_lo   = lo[:, 0], lo[:, 1]
    ell_hi, Dl_hi   = hi[:, 0], hi[:, 1]
    ell_mid, Dl_mid = mid[:, 0], mid[:, 1]
    ell_band = np.linspace(max(ell_lo.min(), ell_hi.min()), min(ell_lo.max(), ell_hi.max()), 200)
    Dl_lo_i = np.interp(ell_band, ell_lo, Dl_lo)
    Dl_hi_i = np.interp(ell_band, ell_hi, Dl_hi)
    ax.fill_between(ell_band, Dl_lo_i, Dl_hi_i, color='tab:orange', alpha=0.25,
                     label='Nikolic+23 Fig.1 (1.5 Gpc/1050$^3$, illustrative --\n'
                           'different box size/resolution)')
    ax.plot(ell_mid, Dl_mid, color='tab:orange', lw=1.2)


def main(config_path, seed_for_shift, box_len_override, hii_dim_override, source,
         wrap_cycle_seed=None, random_seed_override=None, skip_direct=False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sim_cfg = cfg['21cmfast']
    BOX_LEN = box_len_override if box_len_override is not None else sim_cfg['BOX_LEN']
    HII_DIM = hii_dim_override if hii_dim_override is not None else sim_cfg['HII_DIM_coeval']
    is_fiducial_config = (BOX_LEN == sim_cfg['BOX_LEN'] and HII_DIM == sim_cfg['HII_DIM_coeval'])
    dx_fiducial = sim_cfg['BOX_LEN'] / sim_cfg['HII_DIM_coeval']
    dx_this_run = BOX_LEN / HII_DIM
    if abs(dx_this_run - dx_fiducial) > 1e-6:
        print(f"WARNING: this run's dx={dx_this_run:.4f} Mpc differs from "
              f"fiducial's dx={dx_fiducial:.4f} Mpc -- resolution is NOT held "
              f"fixed, so box size is not isolated as the only varied axis. "
              f"If testing the periodicity hypothesis, pick HII_DIM so dx matches.\n")
    z_min, z_max = sim_cfg['z_min'], sim_cfg['z_max']
    z_snapshots = sorted(cfg['coeval_ksz']['z_snapshots'])
    cache_dir = cfg['data']['cache_dir']   # SAME as fiducial -- deliberate, for cache-hit
    out_dir  = cfg['data']['output_dir'].rstrip('/')
    plot_dir = cfg['data']['plot_dir'].rstrip('/')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # astro_params/flag_options: absent in fiducial.yaml -> both None ->
    # astro_overridden False -> every branch below behaves exactly as
    # before this was added. See module docstring point 1-3.
    astro_params = sim_cfg.get('astro_params')
    flag_options = sim_cfg.get('flag_options')
    astro_overridden = astro_params is not None or flag_options is not None
    # Only the STITCHED lightcone's cosmic seed is overridable -- the direct
    # reference always uses the config's own random_seed (see --skip-direct
    # in the module docstring for why: it's a shared, per-(BOX_LEN,HII_DIM,
    # astro_params) anchor, not something that needs to vary per ensemble draw).
    stitched_random_seed = (random_seed_override if random_seed_override is not None
                             else sim_cfg['random_seed'])

    tag_suffix = "" if is_fiducial_config else f"_box{int(BOX_LEN)}"
    if source == 'native':
        tag_suffix += "_native"
    if wrap_cycle_seed is not None and source == 'coeval':
        tag_suffix += f"_wrapcycle{wrap_cycle_seed}"
    elif random_seed_override is not None:
        # UNFIXED-stitching control runs (no --wrap-cycle-seed) of an ensemble:
        # without this tag every seed would write the same filename and
        # overwrite the last. Wrap-cycle runs already encode the seed above,
        # and runs without --random-seed are untouched (byte-identical names).
        tag_suffix += f"_seed{random_seed_override}"
    if astro_overridden:
        tag_suffix += "_nikolic"
    print(f"Coherence decomposition -- BOX_LEN={BOX_LEN} Mpc, "
          f"HII_DIM={HII_DIM} (dx={dx_this_run:.4f} Mpc), {len(z_snapshots)} z_snapshots, "
          f"source={source}. "
          f"{'FIDUCIAL config.' if is_fiducial_config else 'BOX-SIZE OVERRIDE -- periodicity test run.'}")
    if astro_overridden:
        print(f"ASTRO PARAMS OVERRIDDEN (not this pipeline's default HII_EFF_FACTOR model): "
              f"astro_params={astro_params}, flag_options={flag_options}. The 'direct' "
              f"reference below is recomputed FRESH at these params -- closure_test.npz "
              f"is NOT reused regardless of box/dim override status. Output files get an "
              f"extra '_nikolic' tag ({tag_suffix}) so they don't collide with the "
              f"existing fiducial outputs.")
        if source == 'native':
            print("WARNING: astro_params/flag_options were set but --source native uses "
                  "stitch_lightcone_native(), which this change does NOT touch -- these "
                  "params are IGNORED for the stitched lightcone itself on the native path. "
                  "Only the (already-recomputed) direct reference would reflect them. Use "
                  "--source coeval for the Nikolic/Mesinger/Gorce replication.\n")
    if random_seed_override is not None:
        print(f"--random-seed={random_seed_override}: overriding the STITCHED lightcone's "
              f"cosmic seed (config default was {sim_cfg['random_seed']}). The direct "
              f"reference, if computed this run, still uses the config's own "
              f"random_seed={sim_cfg['random_seed']} regardless -- see module docstring.")
    if skip_direct:
        print(f"--skip-direct: skipping the coeval-direct reference entirely this run -- "
              f"ensemble/batch mode. Run once without this flag to get the reference shared "
              f"across the whole batch (depends only on BOX_LEN/HII_DIM/astro_params, not on "
              f"wrap_cycle_seed or --random-seed).")
    if source == 'native':
        print("CAVEAT: velocity amplitude not independently validated for the native "
              "path -- see native_lightcone.py's docstring. Fine for the Delta-chi "
              "SHAPE test below; do not read absolute D_3000/cross-power values as "
              "validated. The 'direct' reference plotted is ALWAYS closure_test.npz's "
              "fiducial-coeval value regardless of this run's own box_len/hii_dim -- "
              "a rough anchor only, see module docstring.")
        if wrap_cycle_seed is not None:
            print(f"WARNING: --wrap-cycle-seed={wrap_cycle_seed} was passed but "
                  f"stitch_lightcone_native has no such parameter -- IGNORED. This "
                  f"run uses no rotation at all, same as native's existing default.\n")

    # ---- load the matched window + chi_eff from the closure test (script 14) --
    # these are LOS/redshift-only quantities, reused regardless of box size. ----
    closure_path = f"{out_dir}/closure_test.npz"
    if not os.path.exists(closure_path):
        raise FileNotFoundError(
            f"{closure_path} not found -- run scripts/14_closure_test.py first.")
    closure = np.load(closure_path)
    chi_eff = float(closure['chi_eff'])
    z_lo, z_hi = float(closure['z_lo']), float(closure['z_hi'])

    if skip_direct:
        ell_direct, Dl_direct = np.array([]), np.array([])
    elif (is_fiducial_config and not astro_overridden) or source == 'native':
        # Reuse script 14's own trusted direct curve directly. For native at a
        # box-size override this is ALSO the fallback (see module docstring --
        # no native equivalent of coeval_sweep.run_one_config exists or is
        # needed for what this test asks). NOT taken when astro_overridden,
        # even if box/dim match the loaded config's own values -- see module
        # docstring point 1: closure_test.npz's astrophysics model is fixed
        # to the ORIGINAL fiducial run's, never valid as a reference once the
        # loaded config's own astro params differ from that.
        ell_direct, Dl_direct = closure['ell_direct'], closure['Dl_direct']
    else:
        # Box size differs, and/or astro params are overridden, coeval source
        # -- coeval-direct's own P_qperp depends on both, so recompute a
        # FRESH direct reference at THIS run's own (BOX_LEN, HII_DIM,
        # astro_params), same as script 16's quicktest does for box size.
        print(f"Computing a FRESH coeval-direct reference at BOX_LEN={BOX_LEN}, "
              f"HII_DIM={HII_DIM} (closure_test.npz's own Dl_direct is fiducial-"
              f"specific, not reusable here)...")
        from ksz_pipeline.convergence.coeval_sweep import run_one_config as run_coeval_one_config
        run_one_config_kwargs = dict(
            tag=f"coherence_direct{tag_suffix}",
            N_THREADS=sim_cfg['N_THREADS'],
            random_seed=sim_cfg['random_seed'],
        )
        if astro_overridden:
            # MUST match the stitched calculation's astro_params/flag_options
            # below, or D_direct and D_stitched silently describe two
            # different astrophysics models. See module docstring point 2 --
            # not yet verified run_one_config accepts these; fail loudly
            # rather than silently proceeding with mismatched physics.
            run_one_config_kwargs['astro_params'] = astro_params
            run_one_config_kwargs['flag_options'] = flag_options
        try:
            direct = run_coeval_one_config(BOX_LEN, HII_DIM, z_snapshots, cache_dir,
                                            **run_one_config_kwargs)
        except TypeError as e:
            if astro_overridden:
                raise TypeError(
                    "run_one_config() raised a TypeError, and astro_params/flag_options "
                    "were being passed to it -- most likely it does not accept those "
                    "kwargs yet. It needs the same astro_params/flag_options passthrough "
                    "that coeval/fields.py and stitch_from_coeval.py already got, or this "
                    "'direct' reference will silently use the WRONG astrophysics model "
                    f"(this pipeline's default HII_EFF_FACTOR, not Nikolic et al.'s). "
                    f"Original error: {e}"
                ) from e
            raise
        ell_direct, Dl_direct = direct['ells_direct'], direct['Dl_direct']

    if skip_direct:
        d3000_direct = float('nan')
        print(f"Matched window z=[{z_lo:.2f},{z_hi:.2f}], chi_eff={chi_eff:.1f} Mpc "
              f"-- direct reference skipped (--skip-direct)\n")
    else:
        d3000_direct = float(np.interp(3000, ell_direct, Dl_direct))
        print(f"Matched window z=[{z_lo:.2f},{z_hi:.2f}], chi_eff={chi_eff:.1f} Mpc "
              f"-- direct D_3000={d3000_direct:.4g} uK^2\n")

    # ================================================================
    # stitched fields -- ONE call differs by source, everything else the
    # same regardless of which loader built it (matching output contract)
    # ================================================================
    if source == 'coeval':
        wc_note = f" [wrap_cycle_seed={wrap_cycle_seed}]" if wrap_cycle_seed is not None else ""
        print(f"Building stitched lightcone (fiducial scale){wc_note}...")
        cell_size = BOX_LEN / HII_DIM
        z_arr = build_los_z_grid(z_min, z_max, cell_size)
        stitched = stitch_lightcone_from_coeval(
            z_snapshots=z_snapshots, z_arr=z_arr, HII_DIM=HII_DIM, BOX_LEN=BOX_LEN,
            cache_dir=cache_dir, angle_deg=0.0,
            N_THREADS=sim_cfg['N_THREADS'], random_seed=stitched_random_seed,
            wrap_cycle_seed=wrap_cycle_seed,
            astro_params=astro_params, flag_options=flag_options)
    else:  # source == 'native'
        print("Building NATIVE lightcone via py21cmfast's own run_lightcone()...")
        stitched = stitch_lightcone_native(
            z_min=z_min, z_max=z_max, HII_DIM=HII_DIM, BOX_LEN=BOX_LEN,
            cache_dir=cache_dir, angle_deg=0.0,
            N_THREADS=sim_cfg['N_THREADS'], random_seed=stitched_random_seed)

    density_1plus = 1.0 + stitched['density']
    x_HII_field   = 1.0 - stitched['xH_box']
    v_los_Mpc_s   = stitched['velocity_z'] / MPC_CM
    x_e_interp    = 1.0 - stitched['xH_box'].mean(axis=(0, 1))
    pos_axis      = stitched['pos_axis']
    z_arr         = stitched['z_arr']

    tau0 = analytic_tau_below(z_arr.min())
    z_mid, ds, dtau, tau = compute_tau(x_e_interp, z_arr, pos_axis, tau0=tau0)
    tau_at_lc, visibility, visibility_3D = compute_visibility(tau, z_arr, z_mid)
    patchy_mask, patchy_mask_3D = compute_patchy_mask(x_e_interp)
    print(f"  n_lc_pix (full) = {len(z_arr)}")

    # ---- truncate to the MATCHED window, same as script 14 ----
    i0 = np.searchsorted(z_arr, z_lo)
    i1 = np.searchsorted(z_arr, z_hi)
    density_1plus_w  = density_1plus[:, :, i0:i1]
    x_HII_field_w    = x_HII_field[:, :, i0:i1]
    v_los_Mpc_s_w    = v_los_Mpc_s[:, :, i0:i1]
    z_arr_w          = z_arr[i0:i1]
    ds_w             = ds[i0:i1 - 1]
    visibility_3D_w  = visibility_3D[:, :, i0:i1]
    patchy_mask_3D_w = patchy_mask_3D[:, :, i0:i1]
    print(f"  n_lc_pix (matched window) = {len(z_arr_w)}\n")

    # ================================================================
    # per-slice decomposition (thin LOS pixels), windowed
    # ================================================================
    print("Computing per-slice theta (this is the ~20-50 GB peak step)...")
    theta_slices, chi_mid_mpc = compute_ksz_map_per_slice(
        density_1plus_w, x_HII_field_w, v_los_Mpc_s_w, z_arr_w, ds_w,
        visibility_3D_w, ne0=ne0_cgs(), patchy_mask_3D=patchy_mask_3D_w)

    # SANITY CHECK: per-slice sum reproduces the trusted compute_ksz_map,
    # on the SAME windowed inputs script 14 itself used. This checks OUR OWN
    # per-slice math is self-consistent -- meaningful and free for either
    # source, since it doesn't depend on which loader produced the fields.
    ksz_map_reference = compute_ksz_map(
        density_1plus_w, x_HII_field_w, v_los_Mpc_s_w, z_arr_w, ds_w,
        visibility_3D_w, ne0=ne0_cgs(), patchy_mask_3D=patchy_mask_3D_w)
    max_abs_diff = np.max(np.abs(ksz_map_reference - theta_slices.sum(axis=-1)))
    print(f"SANITY CHECK (per-slice sum vs compute_ksz_map): "
          f"max|diff| = {max_abs_diff:.3e} (should be ~0)")
    if max_abs_diff > 1e-8 * np.max(np.abs(ksz_map_reference)):
        print("  *** WARNING: larger than floating-point noise -- stop and debug. ***\n")
    else:
        print("  OK.\n")

    del density_1plus_w, x_HII_field_w, v_los_Mpc_s_w, visibility_3D_w, patchy_mask_3D_w
    del density_1plus, x_HII_field, v_los_Mpc_s, visibility_3D, patchy_mask_3D  # free memory

    # ================================================================
    # SNAPSHOT-LEVEL GROUPING -- "diagonal" now matches coeval-direct's
    # own definition (see theory note)
    # ================================================================
    print("Grouping thin LOS pixels into per-snapshot slices...")
    theta_grouped, chi_grouped = group_slices_by_snapshot(theta_slices, chi_mid_mpc, z_snapshots)
    print(f"  {theta_slices.shape[-1]} thin LOS pixels -> {theta_grouped.shape[-1]} snapshot groups\n")
    del theta_slices  # free the large thin-pixel array, no longer needed

    # ================================================================
    # UNSHIFTED decomposition -- the actual stage-1 test
    # ================================================================
    ell_dec, Dl_total, Dl_diag, Dl_off = decompose_p_total_diag_off(
        theta_grouped, BOX_LEN, chi_eff)
    d3000_total = float(np.interp(3000, ell_dec, Dl_total))
    d3000_diag  = float(np.interp(3000, ell_dec, Dl_diag))
    d3000_off   = float(np.interp(3000, ell_dec, Dl_off))

    print(f"{'':26s} {'D_3000 [uK^2]':>15s}")
    print(f"{'coeval-direct':26s} {d3000_direct:>15.4g}")
    print(f"{'stitched P_total':26s} {d3000_total:>15.4g}")
    print(f"{'stitched P_diag (grouped)':26s} {d3000_diag:>15.4g}")
    print(f"{'stitched P_off':26s} {d3000_off:>15.4g}")

    if skip_direct:
        print(f"\n(direct reference skipped -- no |P_diag - direct|/direct comparison this run)")
    else:
        frac_diff = abs(d3000_diag - d3000_direct) / d3000_direct
        print(f"\n|P_diag - direct| / direct = {frac_diff:.1%}")
        if source == 'native':
            print(">>> --source native: this comparison uses closure_test.npz's "
                  "coeval-derived direct value as a rough anchor only (see module "
                  "docstring) -- a large fraction here is EXPECTED given the "
                  "unvalidated velocity amplitude, not necessarily a problem. "
                  "Proceed to the periodicity control and Delta-chi test below, "
                  "which do not depend on this absolute comparison.")
        elif frac_diff > 0.3:
            print(">>> Still a substantial mismatch even with correct snapshot-level "
                  "grouping. Per the theory note, P_diag was never GUARANTEED to equal "
                  "direct exactly -- but a mismatch this large still likely means an "
                  "unreconciled convention issue, not (only) the q_parallel-cancellation "
                  "physics. Investigate before treating P_off below as informative.")
        else:
            print(">>> P_diag now reasonably matches coeval-direct at proper granularity. "
                  "Proceed to the periodicity control and Delta-chi test below.")

    # ================================================================
    # PERIODICITY CONTROL -- random shift, preserves per-slice power,
    # destroys fixed cross-slice alignment
    # ================================================================
    print(f"\nRunning periodicity control (random shift, seed={seed_for_shift})...")
    theta_shifted = random_shift_slices(theta_grouped, seed=seed_for_shift)
    ell_shift, Dl_total_shift, Dl_diag_shift, Dl_off_shift = decompose_p_total_diag_off(
        theta_shifted, BOX_LEN, chi_eff)

    d3000_diag_shift = float(np.interp(3000, ell_shift, Dl_diag_shift))
    d3000_off_shift  = float(np.interp(3000, ell_shift, Dl_off_shift))
    print(f"  P_diag  unshifted vs shifted: {d3000_diag:.4g} vs {d3000_diag_shift:.4g} "
          f"(should closely agree -- shift preserves per-slice power exactly; "
          f"a mismatch here is a bug in random_shift_slices or the decomposition)")
    print(f"  P_off   unshifted vs shifted: {d3000_off:.4g} vs {d3000_off_shift:.4g} "
          f"(a large drop under shifting means the coherent excess IS "
          f"alignment-dependent -- consistent with either real LOS physics or a "
          f"stitching artifact, does NOT by itself distinguish the two)")

    # ================================================================
    # Delta-chi binned cross-power, unshifted vs shifted
    # ================================================================
    print("\nComputing Delta-chi binned cross-power (unshifted and shifted)...")
    dchi_c, cross_mean, cross_std, n_pairs = cross_power_by_dchi(theta_grouped, chi_grouped, BOX_LEN)
    dchi_c_s, cross_mean_s, cross_std_s, n_pairs_s = cross_power_by_dchi(theta_shifted, chi_grouped, BOX_LEN)

    print(f"{'dchi [Mpc]':>12} {'unshifted':>16} {'shifted':>16} {'n_pairs':>10}")
    for i in range(len(dchi_c)):
        if n_pairs[i] > 0:
            print(f"{dchi_c[i]:>12.1f} {cross_mean[i]:>16.4e} {cross_mean_s[i]:>16.4e} {n_pairs[i]:>10d}")

    # ================================================================
    # plot + save
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    direct_label = 'coeval-direct (script 14)' if source == 'coeval' else \
                   'coeval-direct (script 14, ROUGH ANCHOR ONLY -- see caveat)'
    if astro_overridden:
        direct_label = 'coeval-direct (fresh, Nikolic astro params)'
    if not skip_direct:
        ax1.plot(ell_direct, Dl_direct, 'k-', lw=2, label=direct_label)
    src_word = "stitched" if source == "coeval" else "native"
    wc_tag = f", wrap_cycle_seed={wrap_cycle_seed}" if (wrap_cycle_seed is not None and source == 'coeval') else ""
    ax1.plot(ell_dec, Dl_diag, color='tab:blue', lw=2, ls='--',
             label=f'{src_word} P_diag (grouped, unshifted{wc_tag})')
    ax1.plot(ell_dec, Dl_total, color='tab:red', lw=1.5,
             label=f'{src_word} P_total (unshifted{wc_tag})')
    if astro_overridden:
        # Dropped at the user's request for the illustrative Nikolic-
        # comparison plot (1 Oct 2026) -- decluttered in favor of the
        # digitized Nikolic band below. Dl_total_shift is still COMPUTED
        # above (ax2 and the periodicity print block both still use it) --
        # only this one plot call is skipped, and only for astro_overridden
        # runs; the default/fiducial periodicity-test plot is unchanged.
        pass
    else:
        ax1.plot(ell_shift, Dl_total_shift, color='tab:green', lw=1.5, ls=':',
                  label=f'{src_word} P_total (shifted control{wc_tag})')
    if astro_overridden and source == 'coeval':
        _plot_nikolic_band(ax1)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel(r'$\ell$'); ax1.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')
    if astro_overridden and source == 'coeval':
        ax1_title = 'P_diag vs direct, Nikolic/Mesinger/Gorce replication'
    elif source == 'coeval':
        ax1_title = 'P_diag vs direct, fiducial resolution'
    else:
        ax1_title = f'P_diag vs direct, {source} source'
    ax1.set_title(ax1_title)
    ax1.legend(fontsize=8)

    valid = n_pairs > 0
    ax2.errorbar(dchi_c[valid], cross_mean[valid],
                 yerr=cross_std[valid] / np.sqrt(np.maximum(n_pairs[valid], 1)),
                 fmt='o-', color='tab:purple', capsize=3, label='unshifted')
    ax2.errorbar(dchi_c_s[valid], cross_mean_s[valid],
                 yerr=cross_std_s[valid] / np.sqrt(np.maximum(n_pairs_s[valid], 1)),
                 fmt='s--', color='gray', capsize=3, label='shifted control')
    ax2.axhline(0, color='k', lw=0.8, ls='--')
    ax2.set_xlabel(r'$|\Delta\chi|$ [Mpc]')
    ax2.set_ylabel('mean pairwise cross-power')
    ax2.set_title('P_off vs radial separation: real vs shifted' if source == 'coeval'
                  else f'P_off vs radial separation: real vs shifted ({source})')
    if wc_tag:
        ax2.set_title(ax2.get_title() + wc_tag)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plot_path = f"{plot_dir}/coherence_decomposition{tag_suffix or '_fiducial'}.png"
    fig.savefig(plot_path, dpi=130, bbox_inches='tight')
    print(f"\nSaved -> {plot_path}")

    np.savez(f"{out_dir}/coherence_decomposition{tag_suffix or '_fiducial'}.npz",
              box_len=BOX_LEN, hii_dim=HII_DIM, source=source,
              wrap_cycle_seed=wrap_cycle_seed if wrap_cycle_seed is not None else -1,
              astro_overridden=astro_overridden,
              ell_direct=ell_direct, Dl_direct=Dl_direct, d3000_direct=d3000_direct,
              ell_dec=ell_dec, Dl_total=Dl_total, Dl_diag=Dl_diag, Dl_off=Dl_off,
              ell_shift=ell_shift, Dl_total_shift=Dl_total_shift,
              Dl_diag_shift=Dl_diag_shift, Dl_off_shift=Dl_off_shift,
              dchi_centers=dchi_c, cross_mean=cross_mean, cross_std=cross_std, n_pairs=n_pairs,
              cross_mean_shifted=cross_mean_s, cross_std_shifted=cross_std_s,
              chi_eff=chi_eff, z_lo=z_lo, z_hi=z_hi, seed_for_shift=seed_for_shift)
    print(f"Saved -> {out_dir}/coherence_decomposition{tag_suffix or '_fiducial'}.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    parser.add_argument("--shift-seed", type=int, default=42)
    parser.add_argument("--box-len", type=float, default=None,
                         help="Override BOX_LEN for the periodicity test -- "
                              "pick --hii-dim so dx matches fiducial's own "
                              "(e.g. --box-len 400 --hii-dim 256, since "
                              "fiducial is 800/512, same dx=1.5625 Mpc)")
    parser.add_argument("--hii-dim", type=int, default=None,
                         help="Override HII_DIM -- see --box-len")
    parser.add_argument("--source", choices=["coeval", "native"], default="coeval",
                         help="lightcone data source; default preserves existing "
                              "behaviour exactly")
    parser.add_argument("--wrap-cycle-seed", type=int, default=None,
                         help="Use per-wrap-cycle rotation (stitch_from_coeval.py's "
                              "wrap_cycle_seed) instead of the no-rotation default. "
                              "--source coeval only -- ignored with a warning for "
                              "--source native. Default None preserves existing "
                              "behaviour exactly (byte-identical output).")
    parser.add_argument("--random-seed", type=int, default=None,
                         help="Override the config's 21cmfast.random_seed for the "
                              "STITCHED lightcone only (an ensemble over cosmic initial "
                              "conditions -- see module docstring). The direct reference, "
                              "when computed, always uses the config's own random_seed "
                              "regardless. Default None preserves existing behaviour "
                              "exactly.")
    parser.add_argument("--skip-direct", action="store_true",
                         help="Skip the coeval-direct reference entirely (see module "
                              "docstring) -- for ensemble runs where one shared reference "
                              "(from a single run without this flag) covers the whole "
                              "batch. Default False preserves existing behaviour exactly.")
    args = parser.parse_args()
    main(args.config, args.shift_seed, args.box_len, args.hii_dim, args.source,
         args.wrap_cycle_seed, args.random_seed, args.skip_direct)
