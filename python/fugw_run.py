"""FUGW solver bridge for TreeNeighbor (DESIGN.md §3c).

Reads a problem directory written from R (CSV matrices + params.json),
solves fused unbalanced Gromov-Wasserstein with POT, writes pi.csv back.

POT's implementation optimizes a PAIR of couplings (P, Q) by block
coordinate descent — a relaxation whose couplings coincide at the exact
solution — and its `alpha` weights the linear term only: the linear part is
(alpha/2) * sum((P+Q) * M) with the quadratic GW term at coefficient 1.
Our design's alpha (cost = a*M + (1-a)*GW) maps to alpha_pot = a / (1 - a)
up to overall scale, which does not change the minimizer. The caller passes
alpha_pot directly.

Usage: fugw_run.py <problem_dir>
  problem_dir: M.csv (nA x nB), CA.csv, CB.csv, wA.csv, wB.csv, params.json
  writes:      pi.csv (nA x nB), info.json
"""
import json
import os
import sys

import numpy as np
import ot

d = sys.argv[1]
read = lambda f: np.loadtxt(os.path.join(d, f), delimiter=",", ndmin=2)
M = read("M.csv")
CA = read("CA.csv")
CB = read("CB.csv")
wA = read("wA.csv").ravel()
wB = read("wB.csv").ravel()
p = json.load(open(os.path.join(d, "params.json")))

eps = float(p["epsilon"])
# mm is stable at epsilon 0 (no entropic blur); positive epsilon needs the
# log-domain sinkhorn to avoid exp overflow when costs/epsilon is large
solver = "mm" if eps == 0 else "sinkhorn_log"
pi_samp, pi_feat, log = ot.gromov.fused_unbalanced_gromov_wasserstein(
    CA, CB, wx=wA, wy=wB,
    reg_marginals=float(p["rho"]), epsilon=eps,
    divergence="kl", unbalanced_solver=solver,
    alpha=float(p["alpha_pot"]), M=M,
    max_iter=500, tol=1e-8, max_iter_ot=1000, tol_ot=1e-8,
    log=True,
)
gap = float(np.abs(pi_samp - pi_feat).sum())
pi = (pi_samp + pi_feat) / 2.0
np.savetxt(os.path.join(d, "pi.csv"), pi, delimiter=",")
json.dump({"pi_gap": gap, "mass": float(pi.sum())},
          open(os.path.join(d, "info.json"), "w"))
print(f"fugw: coupling {pi.shape}, mass {pi.sum():.4f}, P-Q gap {gap:.2e}")
