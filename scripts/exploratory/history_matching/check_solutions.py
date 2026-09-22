import json, numpy as np, py21cmfast as p21c
D = "/user1/swanith/hist_match/cache"
zs = np.arange(20.0, 4.99, -0.5)
sols = json.load(open("solutions_400_64.json"))
tgt = np.load("solutions_400_64_target.npy")
up = p21c.UserParams(HII_DIM=64, BOX_LEN=400.0, N_THREADS=8)
ic = p21c.initial_conditions(user_params=up, random_seed=37, direc=D)
pf = {z: p21c.perturb_field(redshift=z, init_boxes=ic, direc=D) for z in zs}
fl = p21c.FlagOptions()

def hist(zeta, tv, r):
    kw = dict(HII_EFF_FACTOR=zeta, ION_Tvir_MIN=tv)
    if r is not None: kw["R_BUBBLE_MAX"] = r
    ap = p21c.AstroParams(**kw)
    return np.array([1 - float(np.mean(p21c.ionize_box(
        redshift=z, init_boxes=ic, perturbed_field=pf[z], astro_params=ap,
        flag_options=fl, write=False).xH_box)) for z in zs])

res = {s["rmax"]: hist(s["zeta"], s["log10_Tvir"], s["rmax"]) - tgt for s in sols}
print("    z  x_e(tgt) " + " ".join(f"dx[Rmax={k}]".rjust(15) for k in res))
for i, z in enumerate(zs):
    print(f"{z:5.1f}  {tgt[i]:8.4f} " + " ".join(f"{res[k][i]:15.4f}" for k in res))
for k, r in res.items():
    i = np.argmax(np.abs(r))
    print(f"Rmax={k}: worst |dx|={abs(r[i]):.4f} at z={zs[i]}")
