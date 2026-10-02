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

## AMBER cross-check — progress (updated 2026-10-02)

AMBER (github.com/hytrac/amber) is used as an independent, semi-numerical
check on this repo's own kSZ methods — see `docs/amber_integration.md`
for the full technical conventions/unit-mapping writeup. Summary of
where this stands:

**Done and trusted:**

- Field-dump patch (`external/amber_patch/0001-...patch`) + adapter
  (`src/ksz_pipeline/amber/`) let AMBER's fields feed straight into this
  repo's own `compute_cell`/`qperp_power` — see the validation gate
  results table in `docs/amber_integration.md`. No admin rights needed;
  `build_amber.sh` clones, patches and builds AMBER against ifort+MKL.
- **Chen et al. (2023) Fig. 10 reproduction + a 256³ (z_mid, Δz) sweep**
  are done and committed (`scripts/40_run_amber_sweep.py` /
  `41_plot_amber_sweep.py`, plot in `data/plots/26Sep-amber/`).
- **AMBER's native ray-traced lightcone (`Map=write`, real HEALPix
  build) now actually runs**, not just builds — confirmed on swarm
  2026-09-26 at N=128/Nside=128: produces real `cl_ksz_healpix.txt`/
  `cl_kappa_healpix.txt`/`cl_tau_healpix.txt` and FITS maps, cross-checked
  against an independent `healpy.anafast` on the map itself (agreement
  to ~0.1–10%, worse only near the map's own `lmax`, as expected from
  AMBER's `Nlmax=3·Nside` convention).
- **Overlay: AMBER's native map-based C_ℓ vs. its own Limber-from-
  P_qperp C_ℓ, same realization.** Shows real excess power at low–mid ℓ
  (ratio ~1.2–144× across ℓ=100–384), converging toward the Limber value
  near the map's ℓ_max. Consistent with either known Limber breakdown at
  low ℓ or AMBER's angular periodicity (the box is tiled repeatedly
  across the sky at fixed shell radius, confirmed by reading `map_make`'s
  plain-modulo wrapping in `cmbreion.f90` — no derandomizing rotation
  between periodic copies). Not distinguishable from a single map alone.
- **2 Gpc/h, N=1536 fiducial-point run (Chen+23 fiducial: z_mid=8,
  Δz=4, A_z=3, M_h=1e8 M_sun, λ_mfp=3 Mpc/h), full 28-shell history
  (z=4.5–18.5), `Map=no`, completed successfully 2026-10-01** (~1h14m
  runtime once running). Confirms **2048³ is infeasible on this
  cluster** — no node has enough RAM — so 1536³ (dx≈1.30 Mpc/h/cell) is
  used instead; this is already reflected in the commit history.
  Produced the Limber-side products (`cl_kappa.txt`, `cl_tau.txt`,
  `cl_ksz.txt`, 28× `power_z=*.txt`, `tau.txt`) cleanly; separately
  validated with `scripts/30_amber_validation_gate.py` run as its own
  lightweight follow-up job (see caveat below).

**Explicitly shelved (a scope decision, not a blocker):**

- The angular D_diag/D_off cross-shell-coherence decomposition — i.e.
  measuring whether AMBER's angular periodicity shows up as coherent
  cross-shell power in its native lightcone — would need a second
  Fortran patch (`0002-dump-shell-ksz-maps.patch`, stacked on `0001`) to
  dump AMBER's per-shell HEALPix maps, which stock AMBER never writes.
  That patch plus the full angular-harmonics decomposition module were
  drafted and validated against synthetic HEALPix data, but **are not
  deployed**. Decision made 2026-09-29: not worth the additional
  source-patching + time commitment right now, given the advisor's
  priority to move on to the kSZ–LAE work. Revisit only if explicitly
  re-raised.

**Operational lessons from the 2 Gpc/h run (useful for any future AMBER
PBS job on this cluster):**

- `set -u` (nounset) must be turned on **after** sourcing Intel's
  `setvars.sh`, never before — that script isn't nounset-safe. (Same
  rule `build_amber.sh` already follows; a new PBS script missed it.)
- A relative `-o`/working-directory path is resolved against
  `$PBS_O_WORKDIR`, which is wherever `qsub` was run **from**, not where
  the `.pbs` script file lives — always `cd` into the repo root
  immediately before `qsub`, even when giving the script an absolute
  path.
- A PBS batch job does **not** inherit the submitting shell's conda
  activation — `python` is not on `PATH` by default inside the job.
  Explicitly `source .../conda.sh && conda activate <env>` inside the
  script, before `set -u`.
- Running `scripts/30_amber_validation_gate.py` **chained immediately
  after** `amber.x` inside the same job can get OOM-killed (`exit 137`)
  even though the gate script itself only reads small files (`tau.txt`,
  `amber_run.json`) — the preceding run's ~1.6 TB of `fields_z=*.dat`
  writes leave lingering page-cache pressure against the job's memory
  cgroup. Run the gate as its own small, separate job (or interactively)
  after the main run finishes instead.

**Storage flag:** one fiducial point at N=1536 wrote **~1.6 TB** of raw
per-shell `fields_z=*.dat` dumps. Running the remaining 15 Chen+23 sweep
points at this resolution would need **~25 TB** total if those raw dumps
are all kept — worth deciding a retention policy (e.g. keep only
`power_z=*.txt`/`cl_*.txt` long-term) before that sweep is launched.

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
    plots/
  src/ksz_pipeline/
    coeval/                # box generation, momentum/power spectra,
                            # Limber projection, Georgiev reconstruction
    ksz/                     # optical depth/visibility, map building,
                              # stitched-lightcone construction, v4 angular
    convergence/                # box-size / resolution / dz sweeps
    plotting/                     # shared matplotlib styles
    utils/                          # physical constants
  scripts/
    01_make_ksz_lightcone_maps.py       # deprecated, see above
    02_make_ksz_coeval_boxes.py           # trusted
    03_stitched_lightcone_crosscheck.py     # trusted
    04-06_convergence_*.py                    # convergence sweeps
    07_v4_angular_vs_rectilinear.py             # set aside, see above
  jobs/            # working PBS templates -- mirror these
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
  distinguished from a single map; see "AMBER cross-check" above. The
  angular D_diag/D_off decomposition that would disambiguate this is
  deliberately shelved for now, not forgotten.
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
