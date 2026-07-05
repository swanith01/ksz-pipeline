# ksz-pipeline

Kinetic Sunyaev–Zel'dovich (kSZ) power spectrum pipeline built on 21cmFAST.

This repository contains the full numerical pipeline used to compute the kSZ angular power
spectrum via two complementary approaches — an evolving lightcone line-of-sight integration
and a coeval-box Limber projection — together with convergence studies, patchy optical-depth
screening tests, and reionisation-history parameter scans.

This pipeline forms the methodological foundation of the kSZ–LAE cross-correlation paper.

---

## Scientific background

The kSZ temperature fluctuation is computed from the electron momentum field during
the Epoch of Reionization (EoR). Two methods are implemented and compared:

1. **Lightcone (LOS integral):** Direct integration of the electron momentum field along
   skewed lines of sight through the evolving 21cmFAST lightcone.
2. **Coeval boxes (Limber projection):** Transverse momentum power spectrum computed
   from snapshot cubes and projected via the Limber approximation.

Additional modules cover:
- Box-size and resolution convergence of both methods.
- Patchy vs global optical-depth screening comparison.
- kSZ signal dependence on reionisation history (`HII_EFF_FACTOR` scans).

---

## Repository structure

```
ksz-pipeline/
  README.md              # this file
  LICENSE
  CITATION.cff
  environment.yml        # conda environment (recommended)
  pyproject.toml         # pip-installable package definition
  .gitignore

  data/
    README.md            # where the simulation data live (no large files committed)

  configs/
    fiducial.yaml        # fiducial 21cmFAST parameter set
    variants/            # one yaml per parameter-scan variant

  src/
    ksz_pipeline/
      ksz/               # lightcone LOS integral (kSZ maps)
      coeval/            # coeval-box Limber projection
      convergence/       # box-size and resolution convergence utilities
      patchy/            # patchy optical-depth screening
      reion_history/     # reionisation-history parameter scans
      io/                # loading 21cmFAST outputs, saving results
      plotting/          # all plot functions (no science logic here)
      utils/             # physical constants, cosmology helpers

  scripts/
    01_make_ksz_lightcone_maps.py
    02_make_ksz_coeval_boxes.py
    03_convergence_boxsize.py
    04_convergence_resolution.py
    05_patchy_screening.py
    06_reion_history_scan.py
    07_make_all_figures.py

  notebooks/
    exploratory/         # scratch notebooks only — no final results here

  paper/
    figure_scripts/      # one script per paper figure, standalone and reproducible

  jobs/
    lightcone_ksz.pbs    # PBS job script for lightcone runs
    coeval_ksz.pbs       # PBS job script for coeval-box runs
    array_scan.pbs       # PBS array job for parameter scans

  tests/
    test_constants.py
    test_ksz_lightcone.py
    test_limber.py
```

---

## Getting started

### 1. Clone the repository
```bash
git clone https://github.com/swanith01/ksz-pipeline.git
cd ksz-pipeline
```

### 2. Set up the environment
```bash
conda env create -f environment.yml
conda activate ksz-pipeline
pip install -e .
```

### 3. Point to your data
Edit `data/README.md` to record where your 21cmFAST lightcone and coeval outputs live
on your local machine or HPC. Then update `configs/fiducial.yaml` with those paths.

### 4. Run the pipeline
```bash
# Lightcone kSZ maps
python scripts/01_make_ksz_lightcone_maps.py --config configs/fiducial.yaml

# Coeval-box Limber projection
python scripts/02_make_ksz_coeval_boxes.py --config configs/fiducial.yaml

# Convergence tests
python scripts/03_convergence_boxsize.py --config configs/fiducial.yaml
python scripts/04_convergence_resolution.py --config configs/fiducial.yaml

# Reproduce all paper figures
python scripts/07_make_all_figures.py --config configs/fiducial.yaml
```

---

## Code origin and migration

This codebase was migrated and reorganised from:
- `Swanith_DP2_GitBranch` (commits up to `82208d7` / `526b1d7`)
- `Semester-6-Plots`

The old repositories are archived. This is the canonical repository going forward.

---

## Versioning

| Tag | Meaning |
|---|---|
| `v0.1` | First full end-to-end pipeline working |
| `submitted-v1` | Code state at journal submission |
| `accepted-v1` | Code state at acceptance |

At submission, this repository will be archived on Zenodo.

---

## Dependencies

- Python ≥ 3.10
- 21cmFAST v3 (tested on v3.3.1 / v3.4.0)
- numpy, scipy, matplotlib, astropy, pyyaml, h5py

See `environment.yml` for the full pinned environment.

---

## Authors

- Swanith Upadhye
- Supervisor: Prof. Girish Kulkarni (TIFR)
