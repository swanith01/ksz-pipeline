"""
Reconstruct full D_ell(ell) curves for the coeval box-size, resolution,
and dz sweeps from already-cached per-redshift qperp_power .pkl files.
Pure math (compute_cell + build_dz_subsets, no py21cmfast) -- runs
locally, no cluster needed. Run from the repo root.
"""
import pickle
import numpy as np
import yaml

from ksz_pipeline.coeval.limber import compute_cell
from ksz_pipeline.convergence.dz_sweep import build_dz_subsets

CACHE_DIR = "data/cache"
OUT_DIR = "data/products"

# ---- box-size sweep ----
boxsize_tags = {"box200_N32": 200, "box400_N64": 400, "box600_N96": 600,
                 "box800_N128": 800, "box1000_N160": 1000}
save_dict = {}
x_values, tags_order = [], []
print("=== Box-size sweep ===")
for tag, L in boxsize_tags.items():
    path = f"{CACHE_DIR}/convergence_boxsize_coeval/qperp_{tag}.pkl"
    with open(path, 'rb') as f:
        results = pickle.load(f)
    ells, D_ell, sigma_D, *_ = compute_cell(results)
    save_dict[f"ell_{tag}"] = ells
    save_dict[f"Dl_{tag}"] = D_ell
    save_dict[f"sigma_{tag}"] = sigma_D
    x_values.append(L)
    tags_order.append(tag)
    print(f"  {tag}: D_3000 = {np.interp(3000, ells, D_ell):.4f} uK^2")
save_dict["x_values"] = np.array(x_values)
save_dict["tags"] = np.array(tags_order)
np.savez(f"{OUT_DIR}/convergence_coeval_boxsize_full.npz", **save_dict)
print(f"Saved -> {OUT_DIR}/convergence_coeval_boxsize_full.npz\n")

# ---- resolution sweep ----
resolution_Ns = [32, 64, 128, 256, 512]
save_dict = {}
x_values, tags_order = [], []
print("=== Resolution sweep ===")
for N in resolution_Ns:
    tag = f"res{N}"
    path = f"{CACHE_DIR}/qperp_{tag}.pkl"
    with open(path, 'rb') as f:
        results = pickle.load(f)
    ells, D_ell, sigma_D, *_ = compute_cell(results)
    save_dict[f"ell_{tag}"] = ells
    save_dict[f"Dl_{tag}"] = D_ell
    save_dict[f"sigma_{tag}"] = sigma_D
    x_values.append(N)
    tags_order.append(tag)
    print(f"  {tag}: D_3000 = {np.interp(3000, ells, D_ell):.4f} uK^2")
save_dict["x_values"] = np.array(x_values)
save_dict["tags"] = np.array(tags_order)
np.savez(f"{OUT_DIR}/convergence_coeval_resolution_full.npz", **save_dict)
print(f"Saved -> {OUT_DIR}/convergence_coeval_resolution_full.npz\n")

# ---- dz sweep ----
with open("configs/fiducial.yaml") as f:
    cfg = yaml.safe_load(f)
z_fine = cfg['coeval_ksz']['z_snapshots']
dz_multiples = cfg['convergence']['dz_multiples']

with open(f"{CACHE_DIR}/qperp_dz_fiducial.pkl", 'rb') as f:
    results_full = pickle.load(f)

subsets = build_dz_subsets(z_fine, dz_multiples)
save_dict = {}
x_values, tags_order = [], []
print("=== dz sweep ===")
for m in sorted(dz_multiples):
    label = f"dz_x{m}"
    subset_z = subsets[m]
    results_subset = {z: results_full[z] for z in subset_z if z in results_full}
    ells, D_ell, sigma_D, *_ = compute_cell(results_subset)
    save_dict[f"ell_{label}"] = ells
    save_dict[f"Dl_{label}"] = D_ell
    save_dict[f"sigma_{label}"] = sigma_D
    x_values.append(len(subset_z))
    tags_order.append(label)
    print(f"  {label}: n_z={len(subset_z)}  D_3000 = {np.interp(3000, ells, D_ell):.4f} uK^2")
save_dict["x_values"] = np.array(x_values)
save_dict["tags"] = np.array(tags_order)
np.savez(f"{OUT_DIR}/convergence_dz_coeval_full.npz", **save_dict)
print(f"Saved -> {OUT_DIR}/convergence_dz_coeval_full.npz")
