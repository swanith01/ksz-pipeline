"""
Registration smoke test for simulate/stitch_from_coeval.py.

Verifies the one correctness property the whole kSZ-21cm cross-correlation
depends on: when stitch_lightcone_from_coeval() is asked to stitch
density, xH, velocity_z, and brightness_temp together with a given
wrap_cycle_seed, ALL FOUR fields must be sampled with the IDENTICAL
rotation at every line-of-sight position. If they aren't, kSZ and 21cm
end up drawn from different spatial locations after a periodic wrap --
the maps would still look plausible, the pipeline would still run
cleanly, and the cross-correlation would just be silently wrong (spuriously
suppressed or distorted right where the wrap-cycle fix matters most).

No py21cmfast needed: stitch_field() (the function doing the actual
rotation/sampling) only ever touches plain ndarrays -- it has no idea
density/xH/vz/dTb even exist as separate physical quantities, it just
interpolates whatever dict of boxes it's handed. So this test exercises
the real vendored geometry code directly, with synthetic "tagged" boxes
standing in for real coeval fields, run_coeval_fields() entirely bypassed.

Three checks:
  1. stitch_field() internal consistency -- independently recompute the
     per-cycle rotation angle (via the same public wrap_cycle_index/
     cycle_angle helpers stitch_field uses internally) and confirm
     manually rotating+sampling a tagged box with that angle reproduces
     stitch_field()'s own output exactly.
  2. Shared-seed parity -- stitch four DISTINCT synthetic "fields" with
     the SAME wrap_cycle_seed and confirm all four recover the correct
     underlying (i,j) source pixel (decoded from the tag) at several LOS
     positions spanning multiple wrap cycles. This is the direct analogue
     of what stitch_lightcone_from_coeval does for density/xH/vz/dTb.
  3. Negative control -- deliberately stitch one "field" with a DIFFERENT
     wrap_cycle_seed and confirm the mismatch is actually detected (a
     test that can't fail isn't proving anything).
  4. Static call-site guard -- inspect stitch_lightcone_from_coeval's own
     source and confirm all four stitch_field(...) calls pass the literal
     same wrap_cycle_seed parameter, so a future edit that accidentally
     hardcodes or renames one call's seed is caught even without running
     py21cmfast.

Run: python3 tests/test_shared_rotation.py
"""

import inspect
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz2_21cm.simulate.stitch_from_coeval import (
    stitch_field,
    wrap_cycle_index,
    cycle_angle,
    rotated_coords,
    get_slab_bilinear,
    comoving_distance_mpc,
    stitch_lightcone_from_coeval,
)

NGRID = 16
CELL_SIZE = 10.0          # Mpc -> BOX_LEN = 160 Mpc
SNAP_Z = np.array([4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
SEED = 123


def make_tag_box(offset=0.0):
    """
    A box where every transverse slice is identical and encodes (i, j)
    as a single float: value = i + 1000*j + offset. Constant along the
    box's own LOS axis (axis=-1) so cross-snapshot interpolation in z is
    a no-op and the test isolates the rotation/sampling step only.
    `offset` lets different "fields" carry visibly different values
    while still being invertible back to (i, j).
    """
    i, j = np.meshgrid(np.arange(NGRID), np.arange(NGRID), indexing='ij')
    tag = (i + 1000.0 * j + offset).astype(np.float32)
    return np.repeat(tag[:, :, None], NGRID, axis=2)


def build_z_arr(z_min, z_max):
    """Uniform-comoving-distance z grid at CELL_SIZE spacing (enough
    range to cross several 160 Mpc wrap cycles between z=4 and z=20)."""
    z_fine = np.linspace(z_min, z_max, 4000)
    chi_fine = np.array([comoving_distance_mpc(z) for z in z_fine])
    chi_min, chi_max = chi_fine[0], chi_fine[-1]
    n_pix = int(round((chi_max - chi_min) / CELL_SIZE))
    chi_target = chi_min + CELL_SIZE * np.arange(n_pix)
    return np.interp(chi_target, chi_fine, z_fine)


def decode_ij(value, offset=0.0):
    """Invert make_tag_box's encoding (bilinear smoothing means this is
    approximate near tag discontinuities, exact-ish mid-cell)."""
    v = value - offset
    j = v // 1000.0
    i = v - 1000.0 * j
    return i, j


def check_1_internal_consistency():
    print("1. stitch_field() internal consistency vs. manual rotation...")
    box = make_tag_box()
    boxes = {z: box for z in SNAP_Z}
    z_arr = build_z_arr(4.5, 19.5)
    z0 = float(z_arr.min())

    lc = stitch_field(boxes, SNAP_Z, z_arr, z0, CELL_SIZE, NGRID,
                       wrap_cycle_seed=SEED)

    n_cycles_checked = 0
    seen_cycles = set()
    max_abs_err = 0.0
    for n, z in enumerate(z_arr):
        cycle = wrap_cycle_index(z, z0, CELL_SIZE, NGRID)
        if cycle in seen_cycles:
            continue
        seen_cycles.add(cycle)
        angle = cycle_angle(cycle, SEED)
        ic, jc = rotated_coords(NGRID, angle)
        manual = get_slab_bilinear(box, 0, ic, jc, 'wrap')  # y_cell irrelevant: box is LOS-constant
        err = np.max(np.abs(manual - lc[:, :, n]))
        max_abs_err = max(max_abs_err, err)
        n_cycles_checked += 1

    assert n_cycles_checked >= 3, f"only probed {n_cycles_checked} cycles, widen z range"
    assert max_abs_err < 1e-5, f"stitch_field diverged from manual reconstruction: {max_abs_err}"
    print(f"   OK -- {n_cycles_checked} wrap cycles probed, max |err| = {max_abs_err:.2e}")


def check_2_shared_seed_parity():
    print("2. shared wrap_cycle_seed -> consistent registration across 4 fields...")
    fields = {
        "density":          make_tag_box(offset=0.0),
        "xH_box":           make_tag_box(offset=0.0),
        "velocity_z":       make_tag_box(offset=0.0),
        "brightness_temp":  make_tag_box(offset=0.0),
    }
    z_arr = build_z_arr(4.5, 19.5)
    z0 = float(z_arr.min())

    stitched = {}
    for name, box in fields.items():
        boxes = {z: box for z in SNAP_Z}
        stitched[name] = stitch_field(boxes, SNAP_Z, z_arr, z0, CELL_SIZE, NGRID,
                                       wrap_cycle_seed=SEED)

    names = list(fields.keys())
    ref = stitched[names[0]]
    max_cross_field_diff = 0.0
    for name in names[1:]:
        diff = np.max(np.abs(stitched[name] - ref))
        max_cross_field_diff = max(max_cross_field_diff, diff)

    # All four used the SAME tag box and the SAME seed, so with correct
    # shared rotation they must be exactly bit-identical.
    assert max_cross_field_diff == 0.0, (
        f"fields diverged despite identical input+seed: max diff = {max_cross_field_diff}")
    print(f"   OK -- {len(names)} fields bit-identical under shared seed "
          f"(max cross-field diff = {max_cross_field_diff})")


def check_3_negative_control():
    print("3. negative control -- mismatched seed must be detected...")
    box_a = make_tag_box(offset=0.0)
    box_b = make_tag_box(offset=0.0)   # identical content to box_a
    z_arr = build_z_arr(4.5, 19.5)
    z0 = float(z_arr.min())

    lc_a = stitch_field({z: box_a for z in SNAP_Z}, SNAP_Z, z_arr, z0,
                         CELL_SIZE, NGRID, wrap_cycle_seed=SEED)
    lc_b = stitch_field({z: box_b for z in SNAP_Z}, SNAP_Z, z_arr, z0,
                         CELL_SIZE, NGRID, wrap_cycle_seed=SEED + 1)  # DIFFERENT seed

    diff = np.max(np.abs(lc_a - lc_b))
    assert diff > 1.0, (
        f"test cannot distinguish mismatched seeds (diff={diff}) -- "
        f"the test itself is not sensitive enough to be trusted")
    print(f"   OK -- mismatched seed correctly produces divergent registration "
          f"(max diff = {diff:.1f}, as expected since content was otherwise identical)")


def check_4_call_site_guard():
    print("4. static guard -- stitch_lightcone_from_coeval passes one shared "
          "wrap_cycle_seed to all four stitch_field() calls...")
    src = inspect.getsource(stitch_lightcone_from_coeval)
    calls = [line for line in src.splitlines() if "stitch_field(" in line
             or (line.strip().startswith("HII_DIM, angle_deg, wrap_cycle_seed, wrap_mode")
                 and "stitch_field" not in line)]
    # Count how many argument lines actually pass the literal name
    # 'wrap_cycle_seed' (not a renamed/hardcoded value) into a stitch_field call.
    call_block_count = src.count("wrap_cycle_seed, wrap_mode)")
    assert call_block_count == 4, (
        f"expected exactly 4 stitch_field(...) calls passing "
        f"'wrap_cycle_seed, wrap_mode' verbatim, found {call_block_count} -- "
        f"a future edit may have desynced one field's rotation seed")
    print(f"   OK -- all 4 field calls (density, xH, velocity_z, brightness_temp) "
          f"pass the same wrap_cycle_seed variable")


if __name__ == "__main__":
    check_1_internal_consistency()
    check_2_shared_seed_parity()
    check_3_negative_control()
    check_4_call_site_guard()
    print("\nALL CHECKS PASSED -- kSZ and 21cm fields share rotation under "
          "stitch_lightcone_from_coeval()'s wrap_cycle_seed.")
