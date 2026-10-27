# Modifications to AMBER (local, for ksz-pipeline diagnostics)

AMBER — the Abundance Matching Box for the Epoch of Reionization — is
Copyright (C) Hy Trac, Nianyi Chen and collaborators, and is distributed
under the **GNU General Public License v3** (see `LICENSE` in
<https://github.com/hytrac/amber>). Please cite Trac et al. 2022 (ApJ 927,
186) and Chen et al. 2023 (ApJ 943, 138) for any result that uses it.

This directory does **not** contain AMBER. `build_amber.sh` clones the
original from `hytrac/amber` at a pinned commit, applies the patches below,
and builds locally. The patch files contain modified AMBER source and are
therefore themselves covered by the GPL v3. Anyone given an `amber.x` built
from them must also be given this directory (the patches) and the upstream
commit, as the GPL requires.

These are local diagnostic changes made for the ksz-pipeline project. They
have not been submitted to, reviewed by, or endorsed by the AMBER
developers; problems they introduce should not be attributed to AMBER.

| Patch | Date | File(s) changed | Change |
|---|---|---|---|
| `0001-dump-fields-for-ksz-pipeline.patch` | 2026-09-10 | `src/cmbreion.f90` | Adds `cmb_dumpfields`: after each CMB shell's density and velocity meshes are built, writes `mesh%rho2` and `mesh%mom2` to `output/cmb/fields_<z>.dat`. Output only; no change to AMBER's own calculations. |
| `0002-dump-shell-ksz-maps.patch` | 2026-10-27 | `src/cmbreion.f90` | Adds `cmb_dumpkszmap`: when `Map = make` or `write`, after each shell's `map_make`, writes the running (accumulated) HEALPix kSZ map and the shell's redshift and distance limits to `output/cmb/mapksz_<z>.dat`. The map of a single shell is the difference of consecutive files. Output only; no change to AMBER's own calculations. |

Other files here are not AMBER code: `healpix_stub.f90` (stand-ins for the
HEALPix routines AMBER calls, used only when no HEALPix library is found),
`Makefile.ksz` (build flags for this cluster) and `build_amber.sh`.

Upstream commit tested: `9703f515a70b17e9bac38457d4891dad9abf7f84`.
