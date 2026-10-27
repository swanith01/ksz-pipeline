# ksz-pipeline

Kinetic Sunyaev–Zel'dovich (kSZ) power spectrum pipeline built on 21cmFAST.
Computes the kSZ angular power spectrum from reionization simulations via
two independent methods, for cross-validation.

This pipeline is the methodological foundation of the kSZ–LAE
cross-correlation paper.

---

## Start here: what's trusted right now

**Use these two methods.** Both are validated against Reichardt et al.
(2021)'s D_pkSZ = 1.1 (+1.0/−0.7) μK² data point at full resolution:

| Method | Script | Status |
|---|---|---|
| Coeval boxes (direct + Georgiev) | `scripts/02_make_ksz_coeval_boxes.py` | ✅ Trusted |
| Stitched lightcone | `scripts/03_stitched_lightcone_crosscheck.py` | ✅ Trusted |

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
The working env for scripts 01–06 is `p21c_v3` (py21cmfast 3.3.1). Script
07 (angular, v4) needs a separate v4 env (`p21c_v41` / `PF21c_v41`,
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
`qsub` (batch) or `qsub -I` (interactive).

Results land in `data/products/*.npz`. Plot them with
`notebooks/exploratory/three_way_comparison.ipynb`.

---

## AMBER (`amber-integration` branch) — status as of 2026-10-27

AMBER (github.com/hytrac/amber) supplies simulation **fields only**; every
kSZ method (`compute_cell`, `qperp_power`, the stitched lightcone) is this
repo's own code, so changes across a reionization-history sweep are changes
in the input, never in the method. Full conventions and unit mapping:
`docs/amber_integration.md` (build, conventions, validation gate, HEALPix
lightcone status).

### Quick reference

```bash
bash external/amber_patch/build_amber.sh            # clone + patch + build -> ~/amber/src/amber.x
python scripts/make_amber_input.py --out runs/X --L 256 --N 256 \
       --zmid 8 --zdel 4 --zasy 3 --cosmology chen2022
(cd runs/X && mkdir -p output/{cosmo,reion,grf,lpt,esf,mesh,cmb} \
       && ~/amber/src/amber.x < input/input.txt > output/log.txt)   # via PBS for real sizes
python scripts/30_amber_validation_gate.py --run runs/X     # always, per new config
```

### What is trusted (and how well)

| Result | Where | Agreement |
|---|---|---|
| Our D_coeval/Limber vs AMBER's own Limber, same fields | `scripts/30` (gate), `scripts/53`, `scripts/54` | ≤0.2% over ℓ=1000–8000 (2 Gpc/h, 1536³) |
| Ours vs **Chen+23 Fig. 10 fiducial**, matched z=5–30 | `scripts/54`, `data/plots/26Oct-amber/` | 1.0% at ℓ=3000; 1–3% over ℓ=2000–8000 (8% at ℓ=1000, where the digitized curve is coarse) |
| **Stitched lightcone vs Limber**, 2 Gpc/h (LOS = 0.98 box) | `scripts/55`, `data/plots/26Oct-amber/` | total 0.97–0.98, patchy 1.01–1.04 over ℓ=1000–8000 |
| Corrected stitched curves for the 256³ Chen+23 sweep | `scripts/56`, job 58, `data/plots/27Oct-amber/` | dotted curves now sensible (see below) |

Tag **`dcoeval-limber-validated-2026-10-26`** (commit `3edacfb`) marks the
Chen+23-matched validation; it replaces the earlier `…-2026-10-14` tag.

### Findings, in order (what changed and why)

1. **Pruned raw fields are safely regenerable.** All 14
   `runs/sweep_chen2023` points had lost their `fields_*.dat`. Re-running
   `amber.x` from the preserved `input/input.txt` (seed 1, never overridden)
   reproduced `tau.txt`, `cl_tau.txt`, `cl_kappa.txt` byte-identically and
   `cl_ksz.txt` to ~1e-7 (float summation order). Jobs 34 (verify one point)
   and 35 (regenerate the rest).

2. **Chen+23's curve includes post-reionization kSZ (2026-10-26).** Their
   Sec. 5.2: spectra "integrated from z=5 to z=30". Our gate only covered the
   patchy window (z≈7.25–14.25 for the fiducial). Adding the fully ionized
   shells down to z=5 (`compute_cell(..., post_eor_zmin=...)`, `scripts/54`)
   closes the gap: the post-EoR stretch is **18.9% of D_3000**. Their
   cosmology (`--cosmology chen2022`) and fiducial history were already
   matched; the existing digitized curve (`data/chen23_fiducial.csv`) is the
   right one.

3. **The "haywire" stitched curves came from `v = mom/rho` (2026-10-26).**
   AMBER deposits mass (`rho2` = 1+δ) and momentum (`mom2` = (1+δ)v) on the
   mesh with TSC, interlacing and window deconvolution; the deconvolution
   rings, so `rho2` goes **negative in voids (down to −2.8 at z≈5,
   0.5–2% of cells below 0.05)**. The stitched path divided `mom/rho`,
   interpolated δ and v separately, then multiplied back — pairing huge v
   with a neighbour's normal density. Fix: stitch `q = x_HII · mom` directly
   (`stream_lightcone`, `mode='momentum'`), exactly as `compute_cell` builds
   P_q⊥. On the 2 Gpc/h run, stitched/Limber at ℓ=1000/2000/3000/5000/8000:

   | path | ratios |
   |---|---|
   | velocity (`mom/rho`), rotated | 2.47 / 5.41 / 10.8 / 28.8 / 59.1 |
   | momentum, rotated | 0.99 / 0.95 / 0.91 / 0.80 / 0.59 |
   | momentum, **no rotation** | 0.98 / 0.98 / 0.98 / 0.97 / 0.97 |

   Per-slice diagnostics: the velocity path was 2–12× too large in the fully
   ionized z=5–7 stretch and within ~8% above z≈8.5 (where rho stays > 0).

4. **Bilinear wrap-cycle rotation smooths small scales (2026-10-26).** The
   remaining high-ℓ deficit (0.80 / 0.59 at ℓ=5000 / 8000) disappears with no
   rotation (row 3 above) — interpolation onto a rotated grid is a low-pass
   filter. For boxes that repeat along the LOS, use **axis re-orientation**
   instead (`geometry='reorient'`, a discrete version of G. Kulkarni's
   suggestion to rotate about an axis perpendicular to the LOS): per pass through
   the box, a random box axis becomes the LOS (with that axis's momentum
   component) plus random integer periodic shifts. Exact, no interpolation.

5. **Corrected 256³ sweep (2026-10-27).** `scripts/56` (momentum +
   re-orientation + AMBER's own χ(z)), job 58, plots
   `data/plots/27Oct-amber/sweep_chen2023_*_momentum_reorient.png`. Patchy
   D_3000 of the dotted curves, e.g. Δz=12.8: 1.56 µK² (was 26 with the
   velocity path), A_z=1: 0.98 (was 30.6), A_z=2: 0.85, A_z=5: 0.74,
   A_z=8: 0.75. Caveat: for Δz=12.8 and A_z=1 reionization is unfinished at
   the LOS start z=5 (x_e=0.83, 0.98), so their stitched patchy window is
   truncated there while Limber's lowest shell reaches z=4.75.

6. **q∥ and the low-ℓ excess (2026-10-27).** Ma et al.'s kSZ auto-power
   figure (2-D maps vs Limber q⊥/q∥) suggested the stitched low-ℓ excess is
   the longitudinal momentum. `scripts/57` (job 59, 256³ fiducial, z≥5)
   stitches full q and q⊥-only (q∥ removed per snapshot in Fourier space)
   with identical geometry:

   | ℓ | 200 | 300 | 500 | 1000 | 2000 | 3000 | 5000 |
   |---|---|---|---|---|---|---|---|
   | stitched q / stitched q⊥ | 2.10 | 1.95 | 1.01 | 1.01 | 1.00 | 1.01 | 1.00 |
   | stitched q⊥ / Limber q⊥ | 2.9 | 2.0 | 0.71 | 0.87 | 1.21 | 0.88 | 1.02 |

   So q∥ is about half of the stitched power at ℓ=200–300 and nothing from
   ℓ≈500 up; q⊥ alone is still 2–3× Limber at ℓ=200–300 (box repeats and
   Limber breakdown in a 378 Mpc box are the candidates). The Limber kernel
   applied to P_q∥ (Ma et al.'s dotted curve) overpredicts the actual q∥
   leakage by ~200–400× — it locates the power, it does not predict it,
   consistent with the July 21cmFAST test (`scripts/09`, `18`).

### Which code to use

| Task | Use | Notes |
|---|---|---|
| Limber D_ℓ, any z range | `compute_cell` (+ `post_eor_zmin`), `scripts/30`, `54` | validated |
| Stitched D_ℓ, one run | `scripts/55` (`--mode momentum`; `--no-rotation` if LOS ≤ 1 box) | memory-light, runs at 1536³ |
| Stitched D_ℓ, a sweep | `scripts/56` → `scripts/41 --lightcone-npz … --suffix …` | momentum + re-orientation |
| q⊥/q∥ split | `scripts/57`; `momentum.qsplit_power_from_q`, `transverse_part` | |
| `scripts/32`, `33`, `stitch_from_amber` | **superseded for AMBER** | velocity path, bilinear rotation, Planck18 distances; kept to reproduce old plots |

### Data on swarm (`~/ksz-pipeline-amber/runs/`)

- `sweep_2gpc_1536/zmid8_dz4_az3_mh8_mfp3`: raw fields **deleted 2026-10-26**
  (1.5 TB freed; regenerating = ~48 h AMBER run). Kept (~14 GB, mostly the
  `zre` field): `qperp_checkpoint.pkl` (P_q⊥, all 28 shells),
  `output/cmb/power_z=*.txt`, `tau.txt`, `cl_*.txt`, `amber_gate.npz`,
  `post_eor_vs_chen23.npz`, and the three `stitched_streaming_*.npz`
  (D_ℓ and the 2D kSZ maps for velocity-rotated, momentum-rotated,
  momentum-no-rotation).
- `sweep_chen2023/*`: raw fields **kept** (99 GB) for further stitched tests.
- Retention rule going forward: keep raw fields only while a planned test
  needs them; derived spectra and maps are small.

### Workflow and operational notes

- Code changes arrive as patches: apply on usha (`git am`), run tests, push;
  swarm only pulls. Plots made on swarm: `scp` to usha, commit there.
- `data/plots/` is gitignored; dated folders (`26Oct-amber/`, …) are
  force-added (`git add -f`) snapshots.
- PBS: source `setvars.sh` before `set -u`; activate conda inside the job;
  `cd` to the repo root before `qsub`; use `python -u` (otherwise a killed
  job leaves an empty log); give stitched sweeps hours, not minutes.
- Light post-processing (`scripts/54`, reading checkpoints and text files)
  is fine on the login node; anything reading raw fields goes through PBS.
- **"21cm" means three different things here:** AMBER's own 21cm module
  (built into `amber.x`, switched off in every generated `input.txt`, never
  read by this repo); this repo's 21cmFAST path (scripts 01–22); and the
  separate `ksz2-21cm` repository.

### Earlier AMBER work (still valid)

- AMBER's native ray-traced HEALPix lightcone (`Map=write`) runs on swarm
  (2026-09-26, N=128/Nside=128); its map-based C_ℓ shows excess over its own
  Limber C_ℓ at low–mid ℓ (×1.2–144 over ℓ=100–384). The low-ℓ q∥/Limber
  findings above are relevant to that excess but have not been applied to it.
- 2048³ does not fit any swarm node; 1536³ (dx≈1.30 Mpc/h) is the maximum.
- Running `scripts/30` chained after `amber.x` in one job can be OOM-killed
  by page-cache pressure from the TB-scale writes; run it as its own job.
- The angular D_diag/D_off decomposition (patch `0002`) stays shelved
  (decision 2026-09-29).

---

## Repository structure

```
ksz-pipeline/
  configs/
    fiducial.yaml      # 800 Mpc, HII_DIM_coeval=512, z=4-20
    quicktest.yaml      # 100 Mpc, HII_DIM=32 -- fast iteration
  data/
    cache/               # py21cmfast's own box cache (gitignored)
    products/             # final .npz/.npy results
    plots/                 # gitignored; dated sub-folders are force-added
    chen23_fiducial.csv     # digitized Chen+23 Fig. 10 fiducial
  external/amber_patch/      # AMBER field-dump patch, HEALPix stub, build_amber.sh
  src/ksz_pipeline/
    coeval/                # box generation, momentum/power spectra
                            # (P_q_perp, P_q_parallel split), Limber
                            # projection, Georgiev reconstruction
    ksz/                     # optical depth/visibility, map building,
                              # stitched-lightcone construction, v4 angular
    amber/                     # io (readers), adapter (units/conventions),
                                # stitch_from_amber (in-memory, superseded),
                                # stream_lightcone (slab-streaming stitcher)
    convergence/                # box-size / resolution / dz sweeps
    plotting/                     # shared matplotlib styles
    utils/                          # physical constants
  scripts/
    01-22_*.py         # 21cmFAST path (01 deprecated, 02/03 trusted, 04-22 studies)
    make_amber_input.py  # AMBER input.txt + linpowspec + amber_run.json
    30-31              # AMBER validation gate + plot
    32-33              # AMBER stitched, velocity path (superseded)
    40-44              # AMBER sweep driver, plots, harvest, timing
    50-54              # overlays vs Chen+23 (54: matched z=5-30, post-EoR)
    55                 # streaming stitched lightcone, one run
    56                 # streaming stitched lightcone, sweep (momentum+reorient)
    57                 # q_perp / q_parallel test
  jobs/            # working PBS templates -- mirror these (33-35, 55-59: AMBER)
  pbs/             # 2 Gpc/h, 1536^3 AMBER run + probe
  notebooks/exploratory/    # plotting notebooks, no science logic
```

---

## Open items

Things that are genuinely unresolved right now, not history:

- **Stitched vs. coeval-direct D_3000 disagree by roughly 2×.** Both
  individually land near Reichardt's value, but they don't yet agree
  with each other — this is the current top priority. Check whether the
  ratio is flat across ℓ (normalization-type issue) or varies with ℓ
  (shape/geometric issue).
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
- **AMBER's native-map excess power (low–mid ℓ) is not yet root-caused**
  — periodicity vs. Limber breakdown are both plausible and not yet
  distinguished from a single map. The angular D_diag/D_off decomposition
  that would disambiguate this is deliberately shelved for now, not
  forgotten.
- **AMBER stitched lightcone, low ℓ (ℓ ≲ 300).** On the 2 Gpc/h run the
  corrected stitched map still rises to ~1–7 µK² at ℓ<300 where Limber is
  ≲0.2; on the 256³ fiducial, q∥ explains about half of the excess at
  ℓ=200–300 and q⊥ alone is still 2–3× Limber. Candidates: Limber breakdown
  at large angles, sharp integration edges (z=5 start, patchy mask), few
  modes / box repeats. Untried: a smooth taper at the integration limits
  (would need the 2 Gpc/h fields, now deleted, or a new large-box run).
- **AMBER stitched, residual 2–3% low** vs Limber (2 Gpc/h, ℓ=1000–8000),
  plausibly the single-χ_eff k→ℓ mapping of a rectilinear map.
- **`stitch_from_amber` and `scripts/32`/`33` still default to the velocity
  path, bilinear rotation and Planck18 distances.** Superseded by
  `stream_lightcone` / `scripts/55`–`57`; either switch their defaults or
  retire them.
- **Δz=12.8 and A_z=1 sweep points**: reionization unfinished at the
  stitched LOS start z=5; extend the LOS to the lowest snapshot (z=4.75)
  for a like-for-like patchy comparison.
- **`docs/HANDOFF_wrap_cycle.md`** describes `main`; its note that
  `amber-integration` lacks wrap-cycle code is out of date.
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
