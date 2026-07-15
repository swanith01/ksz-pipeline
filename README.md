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
