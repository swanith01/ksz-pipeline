"""FROZEN copy of coeval/momentum.py::qperp_power as it was BEFORE the
2026-09 AMBER refactor. Test fixture only -- do not import from src."""
import numpy as np


def qperp_power_frozen(delta, xH, vx, vy, vz, BOX_LEN, nbins=None):
    chi = 1.0 - xH
    w   = (1.0 + delta) * chi

    qx = w * vx
    qy = w * vy
    qz = w * vz

    N = qx.shape[0]
    L = float(BOX_LEN)
    d = L / N
    V = L**3

    kfreq      = np.fft.fftfreq(N, d=d) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    k2         = kx**2 + ky**2 + kz**2
    k_mag      = np.sqrt(k2)
    k2_safe    = np.where(k2 == 0.0, np.inf, k2)

    Qx = np.fft.fftn(qx) * d**3
    Qy = np.fft.fftn(qy) * d**3
    Qz = np.fft.fftn(qz) * d**3

    kdotQ_k2 = (Qx * kx + Qy * ky + Qz * kz) / k2_safe
    Qx_perp  = Qx - kdotQ_k2 * kx
    Qy_perp  = Qy - kdotQ_k2 * ky
    Qz_perp  = Qz - kdotQ_k2 * kz

    Qperp2 = (np.abs(Qx_perp)**2
              + np.abs(Qy_perp)**2
              + np.abs(Qz_perp)**2)
    p_flat = (Qperp2 / V / 2.0).ravel()
    k_flat = k_mag.ravel()

    if nbins is None:
        nbins = max(2, int(np.ceil(np.cbrt(N) * 8)))

    pos_k = np.abs(kfreq[kfreq > 0.0])
    kmin  = pos_k.min() if pos_k.size > 0 else 1e-6
    kmax  = np.abs(kfreq).max() * np.sqrt(3.0)
    if kmax <= kmin:
        kmax = kmin * 10.0

    bins  = np.geomspace(kmin, kmax, nbins)
    digit = np.digitize(k_flat, bins)

    k_bins, P_bins, P_std = [], [], []
    for i in range(1, len(bins)):
        mask = digit == i
        if not np.any(mask):
            continue
        k_bins.append(k_flat[mask].mean())
        P_bins.append(p_flat[mask].mean())
        P_std.append(p_flat[mask].std())

    return np.array(k_bins), np.array(P_bins), np.array(P_std)
