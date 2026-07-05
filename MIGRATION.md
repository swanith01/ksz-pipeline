# Migration guide: old repos → ksz-pipeline

This file records exactly which files from the old repositories map to which
modules in this repo. Use it as a checklist when porting code.

---

## Old repositories

| Old repo | Status |
|---|---|
| `Swanith_DP2_GitBranch` | Archive after migration complete |
| `Semester-6-Plots` | Archive after migration complete |

---

## File mapping

### Lightcone kSZ (LOS integral)

| Old path (in `Swanith_DP2_GitBranch`) | New location |
|---|---|
| `Plots/Lightconer_v3/16Jun2026_copy_PatchyScreening_SkewedLOS_LightconeKSZ.py` | `src/ksz_pipeline/ksz/lightcone_integral.py` + `scripts/01_make_ksz_lightcone_maps.py` |
| `Plots/Lightconer_v3/28May2026_SkewedLOS.py` | `src/ksz_pipeline/ksz/skewed_los.py` (create this module) |
| `Plots/Lightconer_v3/Tomas_v3_Attempt.ipynb` | `notebooks/exploratory/lightcone_ksz_dev.ipynb` (exploration only) |

### Coeval-box Limber projection

| Old path | New location |
|---|---|
| `Plots/kSZ_v3_boxes/28May2026_kSZBoxes.py` | `src/ksz_pipeline/coeval/limber_projection.py` + `scripts/02_make_ksz_coeval_boxes.py` |
| `Plots/kSZ_v3_boxes/kSZ_v3_boxes.ipynb` | `notebooks/exploratory/coeval_ksz_dev.ipynb` |

### Convergence studies

| Old path | New location |
|---|---|
| `Plots/BoxSize_change_lightconev3/Box_Size_scan_21April2026.py` | `src/ksz_pipeline/convergence/boxsize_scan.py` + `scripts/03_convergence_boxsize.py` |
| `Plots/BoxSize_change_lightconev3/Resolution_scan_5May2026.py` | `src/ksz_pipeline/convergence/resolution_scan.py` + `scripts/04_convergence_resolution.py` |
| `Plots/kSZ_v3_boxes/ksz_boxsize_convergence.py` | merge into `src/ksz_pipeline/convergence/boxsize_scan.py` |
| `Plots/kSZ_v3_boxes/ksz_resolution_convergence.py` | merge into `src/ksz_pipeline/convergence/resolution_scan.py` |

### Patchy screening

| Old path | New location |
|---|---|
| Cells 4c, 5c, 6c of `Tomas_v3_Attempt.ipynb` | `src/ksz_pipeline/patchy/patchy_screening.py` + `scripts/05_patchy_screening.py` |

### Reionisation-history scans

| Old path | New location |
|---|---|
| `Plots/Reion_hist_change_v3_lightcone/HII_EFF_scan_6May2026.py` | `src/ksz_pipeline/reion_history/hii_eff_scan.py` + `scripts/06_reion_history_scan.py` |
| `Plots/Reion_hist_change_v3_lightcone/Lightcone_v3_Reion_hist_change.ipynb` | `notebooks/exploratory/reion_history_dev.ipynb` |

---

## Migration checklist (per file)

For each file above:
- [ ] Copy the file to the new location.
- [ ] Replace hardcoded paths with `cfg["data"]["..."]` from the YAML config.
- [ ] Replace inline physical constants with imports from `ksz_pipeline.utils.constants`.
- [ ] Move all plotting calls into `ksz_pipeline.plotting.*`.
- [ ] Strip the `__main__` block into the corresponding `scripts/` file.
- [ ] Verify the unit conventions match the pipeline standard (v_com/c, CGS prefactors).
- [ ] Add a docstring citing the relevant equation from the Semester-6 notes.
- [ ] Commit with a meaningful message, e.g.:
      `git commit -m "port: lightcone LOS integral from 16Jun2026 script"`

---

## TODO items carried over from Semester-6 notes

- [ ] Verify whether the stored velocity entering `q = (1+δ)x_e v` is strictly
      the comoving velocity or has an additional physical-velocity conversion downstream.
      (See §1.1 "TODO" in the notes.)
- [ ] Helium correction: replace `x_e` with `x_e^{H+He} = x_HII + 0.079 x_HeII + 0.158 x_HeIII`
      once helium data are available. Controlled by `cfg["21cmfast"]["include_helium"]`.
- [ ] The `load_coeval_snapshot` loader in `src/ksz_pipeline/io/loaders.py` has a
      placeholder for the velocity scaling (v_code → v_com/c). Fill this in.
