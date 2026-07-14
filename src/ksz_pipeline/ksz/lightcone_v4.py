"""
py21cmfast v4 (4.1.0) lightcone construction: rectilinear and angular,
sharing one InputParameters build so the comparison between them is
apples-to-apples.

Built against py21cmfast 4.1.0's ACTUAL API, confirmed via inspect.signature
probes run on the cluster (13Jul2026) -- not from documentation memory,
given how much of this session came from exactly that kind of assumption
turning out wrong even for the well-known v3 API. Specifically confirmed:

  - RectilinearLightconer.with_equal_cdist_slices is DEPRECATED; use
    between_redshifts(min_redshift, max_redshift, resolution, cosmo=...).
  - LightCone has no flat .density/.velocity/.xH_box attributes (the v3
    convention) -- fields live in .lightcones, accessed by name.
  - LightCone.get_fields(inputs) -> tuple returns the ACTUAL field name
    strings for a given InputParameters, without running anything.
    Confirmed for a minimal InputParameters(random_seed=1,
    simulation_options=SimulationOptions(HII_DIM=16, BOX_LEN=50.0)):
        ('log10_mturn_acg', 'log10_mturn_mcg', 'density', 'velocity_z',
         'neutral_fraction', 'ionisation_rate_G12', 'mean_free_path',
         'z_reion', 'kinetic_temperature', 'brightness_temp',
         'halo_sfr', 'n_ion')
    i.e. 'neutral_fraction' not 'xH_box', 'velocity_z' not 'lowres_vz'.
  - generate_lightcone's `regenerate` parameter DEFAULTS TO TRUE --
    opposite of what you want for caching; must be explicitly set False
    or every run recomputes regardless of a populated cache.
  - CacheConfig defaults to caching everything (all fields True) -- a
    safer default than v3's coeval/fields.py, which needed write=False
    set explicitly and turned out to still cache sub-components anyway.
  - AngularLightconer.like_rectilinear takes simulation_options (the
    SimulationOptions sub-object, not the full InputParameters) plus
    match_at_z.

NOT yet confirmed, and the reason this module leads with a diagnostic
check rather than assuming it's right: whether LightCone.lightcones is
dict-style access (lightcone.lightcones['density']) or something else.
get_fields() confirmed the field NAMES; it says nothing about the access
pattern. check_lightcone_fields() below fails loudly and immediately if
this assumption is wrong, rather than silently producing bad arrays.
"""

import os

import numpy as np


def build_inputs(random_seed, HII_DIM, BOX_LEN, node_redshifts,
                  HII_EFF_FACTOR=None, astro_params=None, N_THREADS=1):
    """
    Build one InputParameters object, shared by both the rectilinear and
    angular runs below so they're genuinely the same simulation setup.

    node_redshifts is REQUIRED here, not optional -- confirmed via a real
    run (13Jul2026), not the docstring: InputParameters.__doc__ claims
    it defaults to a log-(1+z)-spaced grid "if evolution is required",
    but run_lightcone raised ValueError: "You are attempting to run a
    lightcone with no node_redshifts" when left unset. Use
    build_node_redshifts() below for a sensible default grid, and pass
    it explicitly.

    Parameters
    ----------
    random_seed : int
    HII_DIM, BOX_LEN : int, float
    node_redshifts : array-like of float -- the redshifts at which
        coeval-like boxes are actually computed; the lightcone
        interpolates between these. See build_node_redshifts().
    HII_EFF_FACTOR : float, optional -- convenience for the one astro
        param this pipeline has touched elsewhere (matches
        configs/fiducial.yaml's astrophysics.HII_EFF_FACTOR); ignored if
        astro_params is given instead
    astro_params : dict, optional -- full override, takes precedence
        over HII_EFF_FACTOR if both given
    N_THREADS : int

    Returns
    -------
    py21cmfast.InputParameters
    """
    import py21cmfast as p21c

    if astro_params is None and HII_EFF_FACTOR is not None:
        astro_params = {"HII_EFF_FACTOR": float(HII_EFF_FACTOR)}

    kwargs = dict(
        random_seed=random_seed,
        node_redshifts=np.asarray(node_redshifts, dtype=float),
        simulation_options=p21c.SimulationOptions(
            HII_DIM=int(HII_DIM), BOX_LEN=float(BOX_LEN), N_THREADS=int(N_THREADS)),
    )
    if astro_params is not None:
        kwargs["astro_params"] = astro_params

    return p21c.InputParameters(**kwargs)


def build_node_redshifts(z_min, z_max, n_nodes=30, margin=0.5):
    """
    Default node_redshifts grid: log-spaced in (1+z) between
    z_min-margin and z_max+margin, matching the spacing PHILOSOPHY the
    InputParameters docstring describes for its (non-functioning, see
    build_inputs' docstring) auto-default -- finer steps at low z,
    coarser at high z, which is the physically sensible choice
    (structure evolves faster at low z).

    margin exists because of a second real error (13Jul2026):
    RectilinearLightconer.between_redshifts(min_redshift=z_min,
    max_redshift=z_max, ...) does NOT produce a lightcone whose actual
    redshift range exactly equals [z_min, z_max] -- fixed-comoving-
    distance stepping means the last slice can overshoot max_redshift.
    Confirmed concretely at HII_DIM=32/BOX_LEN=100 (cell_size=3.125 Mpc):
    requested [4.0, 20.0], actual lightcone range came out
    [4.000000000136674, 20.00103882804506] -- node_redshifts must fully
    CONTAIN the lightcone's actual range or run_lightcone raises
    ValueError. The overshoot's exact size depends on resolution/cell_size,
    not yet checked at fiducial (800Mpc/128^3) scale -- margin=0.5 is a
    generous, not precisely-derived, safety buffer.

    Parameters
    ----------
    z_min, z_max : float -- the LOS redshift range you actually want
    n_nodes : int -- number of coeval-like boxes computed; more nodes
        means finer interpolation but proportionally more py21cmfast
        compute. 30 is a reasonable starting point, not tuned against
        any convergence test yet -- unlike this pipeline's v3 z_snapshots
        grid, which the dz convergence sweep (script 06) actually checked.
    margin : float -- extra redshift range on both ends, see above

    Returns
    -------
    ndarray, ascending
    """
    return np.logspace(np.log10(1.0 + z_min - margin),
                        np.log10(1.0 + z_max + margin), n_nodes) - 1.0


# py21cmfast's own internal lightcone driver (_run_lightcone_from_perturbed_fields)
# unconditionally reads lightcone.lightcones["brightness_temp"], regardless of
# what quantities the caller asked for -- confirmed via a real KeyError
# (13Jul2026) when it was left out of quantities=. RectilinearLightconer's
# own __init__ default, quantities=('brightness_temp',), was the hint this
# is effectively mandatory, not just an example default. Always requested
# from the Lightconer below even if the caller doesn't want it back.
_REQUIRED_INTERNAL_QUANTITIES = ("brightness_temp",)


def check_lightcone_fields(lightcone, expected_fields):
    """
    Fail loudly, immediately, with a clear message, if `lightcone.lightcones`
    isn't the dict-style access this module assumes -- rather than let a
    wrong assumption propagate into silently-wrong D_ell downstream. See
    module docstring: this is the one thing NOT confirmed via the API
    probes, only inferred.

    Returns
    -------
    dict : field_name -> ndarray, exactly as found on lightcone.lightcones
    """
    lc_fields = lightcone.lightcones
    if not hasattr(lc_fields, "__getitem__") or not hasattr(lc_fields, "keys"):
        raise TypeError(
            f"lightcone.lightcones is a {type(lc_fields)}, not dict-like as "
            f"assumed. Run `print(type(lightcone.lightcones)); "
            f"print(dir(lightcone.lightcones))` and send the output -- "
            f"this module's field access needs updating, not the physics."
        )
    missing = [f for f in expected_fields if f not in lc_fields.keys()]
    if missing:
        raise KeyError(
            f"Expected fields {missing} not found in lightcone.lightcones "
            f"(has: {list(lc_fields.keys())}). Check the `quantities=` "
            f"passed to the Lightconer matches what you're trying to read."
        )
    print(f"  [check] lightcone.lightcones is dict-like with keys: "
          f"{list(lc_fields.keys())}")
    for f in expected_fields:
        arr = lc_fields[f]
        print(f"  [check] '{f}': shape={arr.shape} dtype={arr.dtype} "
              f"mean={np.mean(arr):.4e} rms={np.sqrt(np.mean(arr**2)):.4e}")
    return {f: lc_fields[f] for f in expected_fields}


def run_rectilinear(inputs, z_min, z_max, cache_dir, resolution_mpc=None,
                     quantities=("density", "velocity_z", "neutral_fraction")):
    """
    Build and run a RectilinearLightconer (fixed comoving transverse grid
    -- the v4 equivalent of what 01_make_ksz_lightcone_maps.py does in
    v3, but via the proper v4 Lightconer API, not skewed_los.py-style
    extraction).

    Parameters
    ----------
    inputs         : InputParameters, from build_inputs()
    z_min, z_max   : float
    cache_dir      : str
    resolution_mpc : float, optional -- comoving Mpc per LOS slice;
                     defaults to the box's own cell size
                     (BOX_LEN/HII_DIM), matching this pipeline's existing
                     dx convention
    quantities     : tuple of str, field names -- must be valid per
                     LightCone.get_fields(inputs); check that first if
                     unsure

    Returns
    -------
    lightcone : py21cmfast.LightCone
    fields    : dict, from check_lightcone_fields()
    """
    import py21cmfast as p21c
    import astropy.units as u

    if resolution_mpc is None:
        resolution_mpc = inputs.simulation_options.BOX_LEN / inputs.simulation_options.HII_DIM

    lightconer = p21c.RectilinearLightconer.between_redshifts(
        min_redshift=z_min, max_redshift=z_max,
        resolution=resolution_mpc * u.Mpc,
        quantities=tuple(set(quantities) | set(_REQUIRED_INTERNAL_QUANTITIES)),
    )

    os.makedirs(cache_dir, exist_ok=True)
    lightcone = p21c.run_lightcone(
        lightconer=lightconer, inputs=inputs,
        cache=p21c.OutputCache(direc=cache_dir),
        regenerate=False,  # default is True -- would ignore the cache entirely
        progressbar=False,
    )
    fields = check_lightcone_fields(lightcone, quantities)
    return lightcone, fields


def run_angular(inputs, match_at_z, z_max, cache_dir,
                 quantities=("density", "velocity_z", "neutral_fraction")):
    """
    Build and run an AngularLightconer (fixed field of view), pixel-size
    matched to a rectilinear lightconer at match_at_z -- the actual
    comparison basis: same InputParameters, same quantities, same
    effective resolution at one reference redshift, only the geometry
    differs.

    Parameters
    ----------
    inputs      : InputParameters, SAME object passed to run_rectilinear
                  for a genuine apples-to-apples comparison
    match_at_z  : float, redshift at which pixel sizes are matched
                  between the two lightconer types (pick something near
                  the middle of the patchy kSZ weight, e.g. z~7-8)
    z_max       : float
    cache_dir   : str, can be the SAME dir as the rectilinear run --
                  py21cmfast's own caching keys off simulation parameters,
                  not lightconer type, so the underlying coeval-like
                  boxes are shared/reused between the two runs
    quantities  : tuple of str

    Returns
    -------
    lightcone : py21cmfast.LightCone
    fields    : dict, from check_lightcone_fields()
    """
    import py21cmfast as p21c

    lightconer = p21c.AngularLightconer.like_rectilinear(
        simulation_options=inputs.simulation_options,
        match_at_z=match_at_z,
        max_redshift=z_max,
        quantities=tuple(set(quantities) | set(_REQUIRED_INTERNAL_QUANTITIES)),
    )

    os.makedirs(cache_dir, exist_ok=True)
    lightcone = p21c.run_lightcone(
        lightconer=lightconer, inputs=inputs,
        cache=p21c.OutputCache(direc=cache_dir),
        regenerate=False,
        progressbar=False,
    )
    fields = check_lightcone_fields(lightcone, quantities)
    return lightcone, fields
