# ksz-pipeline

Kinetic Sunyaev–Zel'dovich (kSZ) power spectrum pipeline built on 21cmFAST.
Computes the kSZ angular power spectrum from reionization simulations via
two independent methods, for cross-validation.

This pipeline is the methodological foundation of the kSZ–LAE
cross-correlation paper.

---

## Start here: what's trusted right now

**Use these three.** The first two are validated against Reichardt et al.
(2021)'s D_pkSZ = 1.1 (+1.0/−0.7) μK² data point at full resolution:

| Method | Script | Status |
|---|---|---|
| Coeval boxes (direct + Georgiev) | `scripts/02_make_ksz_coeval_boxes.py` | ✅ Trusted |
| Stitched lightcone | `scripts/03_stitched_lightcone_crosscheck.py` | ✅ Trusted |
| Coherence decomposition (wrap-cycle-fixed stitching, P_diag/P_off/P_total) | `scripts/17_coherence_decomposition_fiducial.py` | ✅ Trusted — periodicity artifact (Δχ ≈ box-length multiples) confirmed suppressed, see `plots/30-sep/`. This also resolved the previously-open ~2× stitched-vs-coeval-direct disagreement (see below). Also the vehicle for the Nikolić+23 replication below. |

**Don't use these right now:**

| Method | Script | Status |
|---|---|---|
| Native lightcone | `scripts/01_make_ksz_lightcone_maps.py` | ❌ Deprecated — unresolved D_ℓ excess |
| Angular lightcone (py21cmfast v4) | `scripts/07_v4_angular_vs_rectilinear.py` | ⏸️ Set aside — see Open Items |

If you're picking this repo up fresh, run `02` and `03` and nothing else
until you've read the "Open Items" section below.

---

## Setup

```bash
git clone https://github.com/swanith01/ksz-pipeline.git
cd ksz-pipeline
conda env create -f environment.yml   # or use an existing env with py21cmfast 3.x
conda activate ksz-pipeline           # name may vary -- see below
pip install -e . --no-deps
```

`--no-deps` is deliberate: this avoids pip pulling different numpy/scipy/
astropy versions than whatever your py21cmfast install was built against.

**On the TIFR cluster (swarm / pride):** conda doesn't auto-activate in
fresh shells — run `source ~/miniconda3/etc/profile.d/conda.sh` first.
The working env for every script except 07 is `p21c_v3` (py21cmfast 3.3.1).
Script 07 (angular, v4) needs a separate v4 env (`p21c_v41` / `PF21c_v41`,
py21cmfast 4.1.0) — install `ksz_pipeline` there separately too, editable
installs are per-environment.

---

## Running it

**Quick, small-scale sanity check** (minutes, not hours — do this first
on anything new):
```bash
python scripts/02_make_ksz_coeval_boxes.py --config configs/quicktest.yaml
python scripts/03_stitched_lightcone_crosscheck.py --config configs/quicktest.yaml
```

**Real (fiducial) run** — 800 Mpc, HII_DIM_coeval=512, real compute, use
the cluster:
```bash
qsub jobs/run_fiducial_coeval.pbs
qsub jobs/run_fiducial_stitched.pbs
```
Check `jobs/*.pbs` for the resource/queue conventions before writing new
ones — mirror them rather than guessing at PBS syntax.

**Never run either of the above directly on a login node** — always via
`qsub` (batch) or `qsub -I` (interactive). This includes quick standalone
diagnostic scripts, not just the main pipeline scripts — anything that
calls into py21cmfast does real compute.

Results land in `data/products/*.npz`. Plot them with
`notebooks/exploratory/three_way_comparison.ipynb`.

---

## Repository structure

```
ksz-pipeline/
  configs/
    fiducial.yaml            # 800 Mpc, HII_DIM_coeval=512, z=4-20
    quicktest.yaml            # 100 Mpc, HII_DIM=32 -- fast iteration
    nikolic_mesinger.yaml      # 500 Mpc, HII_DIM_coeval=256 -- Nikolic+23
                                 # replication target, see below
  data/
    cache/               # py21cmfast's own box cache (gitignored)
    products/             # final .npz/.npy results
    external/              # digitized comparison data (e.g. Nikolic+23
                             # Fig.1 kSZ curves, Georgiev reference data)
    plots/                 # raw per-run plot output from scripts
  plots/
    <date>/                 # curated, dated subfolders of the plots worth
                             # keeping for a writeup/discussion -- not every
                             # run's output, just the confirming ones
                             # (e.g. 30-sep/ = wrap-cycle fix verification,
                             # oct1/ = Nikolic+23 ensemble comparison)
  src/ksz_pipeline/
    coeval/                # box generation, momentum/power spectra,
                            # Limber projection, Georgiev reconstruction
    ksz/                     # optical depth/visibility, map building,
                              # stitched-lightcone construction (incl.
                              # wrap-cycle rotation fix), v4 angular
    convergence/                # box-size / resolution / dz sweeps
    plotting/                     # shared matplotlib styles
    utils/                          # physical constants
  scripts/
    01_make_ksz_lightcone_maps.py       # deprecated, see above
    02_make_ksz_coeval_boxes.py           # trusted
    03_stitched_lightcone_crosscheck.py     # trusted
    04-06_convergence_*.py                    # convergence sweeps
    07_v4_angular_vs_rectilinear.py             # set aside, see above
    17_coherence_decomposition_fiducial.py        # P_diag/P_off/P_total
                                                     # decomposition, wrap-
                                                     # cycle-fixed stitching;
                                                     # also runs the
                                                     # Nikolic+23 replication
                                                     # below. Trusted.
    29_nikolic_ensemble_comparison.py               # aggregate D_3000 vs
                                                       # Nikolic+23; no
                                                       # py21cmfast dependency
  jobs/            # working PBS templates -- mirror these. The specific
                    # job files used for the wrap-cycle/Nikolic-ensemble
                    # work below are cluster-local scratch, not committed
                    # here -- clone an existing template (e.g.
                    # run_fiducial_stitched.pbs) and point it at the
                    # relevant config/script instead.
  notebooks/exploratory/    # plotting notebooks, no science logic
```

---

## Nikolić et al. (2023) replication — in progress

Attempting to replicate the kSZ power spectrum from
[Nikolić, Mesinger, Qin & Gorce 2023](https://arxiv.org/abs/2307.01265)
(MNRAS 526, 3170) at their main inference-box scale (500 Mpc / 256³ —
their illustrative 1.5 Gpc/1050³ Figure 1 box is computationally
prohibitive here, ~375 GiB vs. ~5.4 GiB), using their published Figure-1
astrophysical parameters (flexible mass-dependent SFR/escape-fraction
model; target reionization history z_r=6.1, z_end=4.9, τ_e=0.042).

- **Config:** `configs/nikolic_mesinger.yaml` — see its header comments
  for exactly what is/isn't matched to the paper.
- **Method:** `scripts/17_coherence_decomposition_fiducial.py --config
  configs/nikolic_mesinger.yaml --source coeval --skip-direct
  --random-seed <seed> --wrap-cycle-seed <seed>`, run as a 6-seed
  ensemble (101–106) to capture cosmic-variance scatter, mirroring the
  paper's own 20-seed methodology at this box size. Aggregate:
  `scripts/29_nikolic_ensemble_comparison.py` (no py21cmfast dependency —
  can run anywhere the `.npz` products are available).
- **Headline result:** ensemble D_3000 = 0.50 ± 0.08 μK² (16.7% relative
  scatter) vs. Nikolić's own calibration-implied value for this box size,
  0.84 ± 0.13 μK² (derived from their Fig.1 value and their quoted
  f=1.27±0.19 scaling factor, N=20 seeds) — a 2.7–4.1σ gap, depending on
  whose scatter you use. Our realization-to-realization scatter matches
  theirs closely (~17% vs. ~15%), which rules out a trivial setup
  mistake, but the mean amplitude gap is real and currently unexplained.
- **Ruled out so far:** `astro_params`/`flag_options` silently not
  landing in 21cmFAST (confirmed via readback off a real `coeval` object,
  not just echoed from config); cosmology mismatch as the *primary*
  cause (we use this pipeline's default Planck18, not their Planck 2020
  — tested directly, only accounts for ~17% of a related
  reionization-timing offset); realization-to-realization noise (0.2%
  scatter in `xH(z=6.1)` across 3 seeds, versus a ~16% offset to
  explain).
- **Currently testing:** the paper's ionizing-photon-conservation
  correction (`PHOTON_CONS`, Park et al. 2022 — confirmed used by the
  paper per their footnote 11), which was not in our runs until now.
  Gotcha found along the way, worth remembering: `PHOTON_CONS` needs the
  full sequential redshift history to calibrate correctly — a single
  isolated `run_coeval()` call gives a nonsense overcorrection (tested:
  `xH(z=6.1)` swung from 0.58 to 0.06 in isolation). The real test is
  wiring it through the full 29-snapshot pipeline, in progress now.

Digitized Figure-1 kSZ curve: `data/external/nikolic_ksz_{lower,power,upper}.csv`.
Ensemble outputs: `data/products/coherence_decomposition_wrapcycle10{1-6}_nikolic.npz`;
curated plots in `plots/oct1/`.

---

## Open items

Things that are genuinely unresolved right now, not history:

- **Nikolić et al. (2023) replication: ~3–4σ unexplained kSZ amplitude
  gap** at matched box size and astro-params. See the dedicated section
  above — astro_params landing and realization scatter have been ruled
  out, cosmology has been ruled out as the primary cause, and the
  photon-conservation correction is the current lead, being tested now.
- **The Georgiev reconstruction overshoots the direct measurement by
  roughly 3× at every redshift, confirmed at full resolution** (512³,
  not just small test boxes). This needs a real look at the Eq.10
  convolution normalization, not more resolution.
- **Angular lightcone (script 07)** works technically but was set aside
  per project guidance — it also runs on a different py21cmfast major
  version (4.x vs 3.x used everywhere else), which complicates comparing
  it directly to the trusted methods.
- **Native lightcone (script 01)** has a large, unexplained D_ℓ excess.
  Root cause was never found; the stitched-lightcone method exists
  specifically to sidestep it. Not planned to be revisited unless
  something changes.
- **`patchy/`, `reion_history/`, `io/` modules** referenced in earlier
  planning don't exist yet. Patchy optical-depth screening and the
  reionization-history (`HII_EFF_FACTOR`) parameter scan are not yet
  ported into this repo.
- **Box-size/resolution convergence sweeps** (`scripts/04`, `05`) are
  built and pass quicktest-scale checks, but haven't been run at
  fiducial scale yet.
- **`data/products/` is currently tracked in git.** Binary result files
  in version control is worth reconsidering as this grows.

For anything not covered here, check `git log` before assuming something
is broken — a lot of subtle unit/convention issues in the coeval and
lightcone construction code have already been found and fixed; commit
messages describe what and why.

---

## Authors

- Swanith Upadhye
- Supervisor: Prof. Girish Kulkarni (TIFR)
