"""
ksz_pipeline/amber/shell_maps.py

Analysis of AMBER's native HEALPix kSZ lightcone shell by shell, using the
running maps written by external/amber_patch/0002 (output/cmb/mapksz_*.dat).

AMBER's map_make deposits, for every CMB shell (high z first), the LPT
particles of a plainly tiled periodic box (observer at the origin, integer
tile offsets, no rotations -- cmbreion.f90 map_make) into one HEALPix map
that accumulates over shells. 0002 writes that running map after each
shell, so the map of shell iz alone is cum(iz) - cum(iz+1).

Three diagnostics, all from the same shell maps:

  1. P_diag / P_off on the sphere. With C_l^{ij} the cross-spectrum of
     shells i and j, the native map's spectrum is
        C_l^total = sum_i C_l^{ii} (diag) + sum_{i!=j} C_l^{ij} (off).
     Off is split by separation into 'near' (|chi_i - chi_j| < L/2) and
     'far' (>= L/2). A non-periodic universe has little far power except
     at the lowest ell; plain tiling adds far power at lattice lags.
  2. r_l(dchi): normalised cross-correlation of shell pairs vs dchi / L,
     in ell bands. Tiling gives returns at dchi = L|m| (1, sqrt2, sqrt3, 2).
  3. Lattice-image correlation: for each pixel p of shell i (3D point
     chi_i n_p), shift by L*m for a lattice vector m, find the shell j and
     pixel p' containing the image, and correlate map values over all
     (i, p) <-> (j, p') pairs. Plain tiling: the two are the same box
     point, so correlation is high; a real universe gives ~0; a half-lattice
     shift is the control.

Distances: AMBER writes r in comoving Mpc/h; pass L in Mpc/h too.
"""
import glob
import os

import numpy as np


def read_kszmap(path):
    """One 0002 file -> dict(nside, iz, nz, z (3,), r (3,) [Mpc/h], map)."""
    raw = np.fromfile(path, dtype=np.uint8)
    hdr = np.frombuffer(raw[:24], dtype='<i8')
    nside, iz, nz = (int(v) for v in hdr)
    zr = np.frombuffer(raw[24:72], dtype='<f8')
    m = np.frombuffer(raw[72:], dtype='<f8')
    if m.size != 12 * nside**2:
        raise ValueError(f"{path}: {m.size} map values, expected 12*Nside^2 "
                         f"= {12 * nside**2} (not a 0002 kSZ map file?)")
    return dict(nside=nside, iz=iz, nz=nz, z=zr[:3].copy(), r=zr[3:].copy(),
                map=m.copy())


def load_shell_maps(cmb_dir, dtype=np.float64):
    """
    Read every mapksz_*.dat in cmb_dir and difference the running maps.

    Returns dict with
      maps   : (Nshell, Npix) single-shell maps, ordered by increasing
               distance (low z first)
      chi    : (Nshell,) shell mid distance [Mpc/h]
      chi_lo, chi_hi : shell edges [Mpc/h]
      z      : (Nshell,) shell mid redshift
      total  : (Npix,) final running map (= sum of maps)
      nside
    """
    files = glob.glob(os.path.join(cmb_dir, 'mapksz_*.dat'))
    if not files:
        raise FileNotFoundError(f"no mapksz_*.dat in {cmb_dir} -- run AMBER "
                                f"built with patch 0002 and Map = make/write")
    recs = sorted((read_kszmap(f) for f in files), key=lambda r: -r['iz'])
    iz = [r['iz'] for r in recs]
    if iz != list(range(iz[0], iz[0] - len(iz), -1)):
        raise ValueError(f"shell indices not contiguous: {iz}")
    if iz[0] != recs[0]['nz']:
        raise ValueError(f"first running map is shell {iz[0]}, not the "
                         f"highest shell {recs[0]['nz']}: missing files?")
    nside = recs[0]['nside']
    cum = np.stack([r['map'] for r in recs])           # high z first
    shell = np.diff(np.vstack([np.zeros((1, cum.shape[1])), cum]), axis=0)
    shell = shell[::-1].astype(dtype)                  # low z first
    recs = recs[::-1]
    return dict(maps=shell,
                chi=np.array([r['r'][1] for r in recs]),
                chi_lo=np.array([r['r'][0] for r in recs]),
                chi_hi=np.array([r['r'][2] for r in recs]),
                z=np.array([r['z'][1] for r in recs]),
                total=cum[-1].copy(), nside=nside)


def shell_alms(maps, lmax, iter=1):
    import healpy as hp
    return np.array([hp.map2alm(m, lmax=lmax, iter=iter) for m in maps])


def cross_spectra(alms, lmax):
    """
    All C_l^{ij} at once: (Nshell, Nshell, lmax+1), from healpy-ordered
    alms (m >= 0 only stored; m > 0 counted twice), identical to
    hp.alm2cl(alm_i, alm_j) for every pair.
    """
    import healpy as hp
    ns = alms.shape[0]
    C = np.empty((ns, ns, lmax + 1))
    for ell in range(lmax + 1):
        idx = hp.Alm.getidx(lmax, ell, np.arange(ell + 1))
        A = alms[:, idx]
        w = np.full(ell + 1, 2.0)
        w[0] = 1.0
        C[:, :, ell] = np.real((A * w) @ A.conj().T) / (2 * ell + 1)
    return C


def decompose(C, chi, L):
    """
    D_ell-free decomposition (returns C_l arrays):
      total, diag, near (i != j, |dchi| < L/2), far (|dchi| >= L/2).
    total == diag + near + far.
    """
    d = np.abs(chi[:, None] - chi[None, :])
    off = ~np.eye(len(chi), dtype=bool)
    near = off & (d < 0.5 * L)
    far = off & (d >= 0.5 * L)
    return dict(total=C.sum(axis=(0, 1)),
                diag=np.einsum('iil->l', C),
                near=C[near].sum(axis=0),
                far=C[far].sum(axis=0))


def r_vs_dchi(C, chi, L, ell_band, dchi_edges_over_L):
    """
    Normalised shell-pair correlation in an ell band vs dchi / L:
    R(bin) = sum_{pairs in bin} C^{ij}_band / sum sqrt(C^{ii}_band C^{jj}_band),
    i != j, C_band = sum over ell in [ell_band[0], ell_band[1]).
    """
    l0, l1 = ell_band
    Cb = C[:, :, l0:l1].sum(axis=-1)
    dg = np.sqrt(np.clip(np.diag(Cb), 0, None))
    norm = dg[:, None] * dg[None, :]
    x = np.abs(chi[:, None] - chi[None, :]) / L
    off = ~np.eye(len(chi), dtype=bool)
    out = np.full(len(dchi_edges_over_L) - 1, np.nan)
    for k in range(len(out)):
        sel = off & (x >= dchi_edges_over_L[k]) & (x < dchi_edges_over_L[k + 1])
        if sel.any() and norm[sel].sum() > 0:
            out[k] = Cb[sel].sum() / norm[sel].sum()
    return out


def lattice_vectors(max_norm2=3):
    """One representative of each +-pair of integer vectors with
    0 < |m|^2 <= max_norm2, grouped by |m|^2 (1 axis, 2 face, 3 body)."""
    out = []
    rng = range(-2, 3)
    for v in ((a, b, c) for a in rng for b in rng for c in rng):
        n2 = v[0]**2 + v[1]**2 + v[2]**2
        if 0 < n2 <= max_norm2 and tuple(-x for x in v) not in out:
            out.append(v)
    return [np.array(v, float) for v in out]


def direction_class(nside, half_angle_deg=10.0):
    """Per pixel: 1 if within half_angle of a box axis, 2 of a face
    diagonal, 3 of a body diagonal, 0 otherwise."""
    import healpy as hp
    n = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T
    cls = np.zeros(len(n), int)
    c = np.cos(np.radians(half_angle_deg))
    for k, vecs in ((1, [v for v in lattice_vectors(1)]),
                    (2, [v for v in lattice_vectors(2) if (v**2).sum() == 2]),
                    (3, [v for v in lattice_vectors(3) if (v**2).sum() == 3])):
        u = np.array(vecs) / np.linalg.norm(vecs[0])
        near = (np.abs(n @ u.T) >= c).any(axis=1)
        cls[(cls == 0) & near] = k
    return cls


def lattice_image_correlation(sm, L, m, scale=1.0, pixel_mask=None,
                              shells=None):
    """
    Pearson correlation of shell-map values at (i, p) and at the image of
    chi_i n_p + scale*L*m, over all shells i (or `shells`) and pixels p (or
    those with pixel_mask True) whose image lies inside the shell range.
    scale=1 tests the lattice; scale=0.5 is the control. Maps are
    standardised per shell first. Returns (rho, n_pairs).
    """
    import healpy as hp
    maps = sm['maps']
    nside = sm['nside']
    npix = maps.shape[1]
    mu = maps.mean(axis=1, keepdims=True)
    sd = maps.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    zm = (maps - mu) / sd
    n = np.array(hp.pix2vec(nside, np.arange(npix))).T
    pix = np.arange(npix) if pixel_mask is None else np.flatnonzero(pixel_mask)
    lo, hi = sm['chi_lo'], sm['chi_hi']
    shift = scale * L * np.asarray(m, float)
    sx = sxx = sy = syy = sxy = 0.0
    cnt = 0
    for i in (range(len(sm['chi'])) if shells is None else shells):
        x = sm['chi'][i] * n[pix] + shift
        r = np.linalg.norm(x, axis=1)
        j = np.searchsorted(hi, r)                      # shell with lo<=r<hi
        ok = (j < len(hi)) & (r >= lo[np.clip(j, 0, len(lo) - 1)])
        if not ok.any():
            continue
        p2 = hp.vec2pix(nside, *x[ok].T)
        a = zm[i, pix[ok]]
        b = zm[j[ok], p2]
        sx += a.sum(); sy += b.sum(); sxx += (a * a).sum(); syy += (b * b).sum()
        sxy += (a * b).sum(); cnt += a.size
    if cnt < 2:
        return np.nan, cnt
    cov = sxy / cnt - (sx / cnt) * (sy / cnt)
    va = sxx / cnt - (sx / cnt)**2
    vb = syy / cnt - (sy / cnt)**2
    return float(cov / np.sqrt(va * vb)), cnt
