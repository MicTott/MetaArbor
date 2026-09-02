"""Figure 6 sweep: what does the GW tree term contribute?

For each (cost calibration, rho) facet: tree-only GW (alpha=0), the fusion
sweep, and molecular-only unbalanced OT with the identical cost, marginals
and regularization. Reports the coupling, its row entropy, and the realized
molecular / structural objective components.

Usage: fugw_sweep.py <dir>
  reads  M_raw.csv M_clip.csv CA.csv CB.csv wA.csv wB.csv
  writes settings.csv (one row per solved setting) and pi_long.csv
"""
import os
import sys

import numpy as np
import ot

d = sys.argv[1]
read = lambda f: np.loadtxt(os.path.join(d, f), delimiter=",", ndmin=2)
M = {"raw": read("M_raw.csv"), "clip": read("M_clip.csv")}
CA = read("CA.csv")
CB = read("CB.csv")
wA = read("wA.csv").ravel()
wB = read("wB.csv").ravel()

ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
RHOS = [0.3, 1.0, 3.0]

def components(pi, Mc):
    r, c = pi.sum(1), pi.sum(0)
    lin = float((pi * Mc).sum())
    t1 = float((CA ** 2 @ r) @ r)
    t2 = float((CB ** 2 @ c) @ c)
    t3 = float(np.trace(CA @ pi @ CB @ pi.T))
    gw = t1 + t2 - 2 * t3
    rows = pi / np.maximum(pi.sum(1, keepdims=True), 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(rows > 0, rows * np.log(rows), 0), axis=1)
    return lin, gw, float(np.median(ent))

rows, pis = [], []
sid = 0
for cal, Mc in M.items():
    for rho in RHOS:
        for a in ALPHAS:
            ps, pf, _ = ot.gromov.fused_unbalanced_gromov_wasserstein(
                CA, CB, wx=wA, wy=wB, reg_marginals=rho, epsilon=0,
                divergence="kl", unbalanced_solver="mm",
                alpha=(a / (1 - a) if a > 0 else 0.0), M=Mc,
                max_iter=500, tol=1e-8, max_iter_ot=1000, tol_ot=1e-8, log=True)
            pi = (ps + pf) / 2
            lin, gw, ent = components(pi, Mc)
            rows.append((sid, cal, rho, a, "fugw" if a > 0 else "gw_only",
                         lin, gw, ent, float(pi.sum())))
            pis.append(pi); sid += 1
        # molecular-only UOT: identical cost, marginals, KL regularization
        pi = ot.unbalanced.mm_unbalanced(wA, wB, Mc, reg_m=rho, div="kl")
        lin, gw, ent = components(pi, Mc)
        rows.append((sid, cal, rho, 1.0, "uot_molecular", lin, gw, ent,
                     float(pi.sum())))
        pis.append(pi); sid += 1

with open(os.path.join(d, "settings.csv"), "w") as fh:
    fh.write("setting,calibration,rho,alpha,model,lin_component,gw_component,"
             "median_row_entropy,mass\n")
    for r in rows:
        fh.write(",".join(str(x) for x in r) + "\n")
with open(os.path.join(d, "pi_long.csv"), "w") as fh:
    fh.write("setting,i,j,value\n")
    for s, pi in enumerate(pis):
        nz = np.argwhere(pi > 1e-12)
        for i, j in nz:
            fh.write(f"{s},{i},{j},{pi[i, j]:.6e}\n")
print(f"solved {len(pis)} settings")
