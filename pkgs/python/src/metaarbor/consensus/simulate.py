"""Multi-donor simulation framework for MetaArbor-Consensus (DESIGN.md,
validation scenarios 1-4).

Generates K donor datasets from one shared latent tree with per-donor:
batch distortion, missing clades, private clades, annotation resolution
(labels cut at family or leaf level), abundance imbalance, and the
flagship case — a rare private branch (low prevalence) in one small donor.
Ground truth (latent tree, per-donor presence, eligibility) is returned so
the eligibility model's three-way calls (supported / private-or-absent /
unknown) can be scored exactly.
"""
from __future__ import annotations

import numpy as np


def latent_tree(n_family=4, n_sub=3):
    """Shared truth: families F1..Fk with subtypes Fi.sj."""
    fams = [f"F{i+1}" for i in range(n_family)]
    leaves = [f"{f}.s{j+1}" for f in fams for j in range(n_sub)]
    return {"families": fams, "leaves": leaves,
            "family_of": {l: l.split(".")[0] for l in leaves}}


def simulate_donors(K=3, n_family=4, n_sub=3, n_genes=1500,
                    cells_per_leaf=100, fam_lfc=1.2, sub_lfc=0.9,
                    batch_sd=0.5, seed=0,
                    missing=None, private=None, resolution=None,
                    abundance=None, rare_private=None):
    """Generate K donor datasets from the shared latent tree.

    missing:    {donor_idx: [latent clade]} — clade absent from that donor
    private:    {donor_idx: [name]} — donor-specific NEW leaves (their own
                expression program, present nowhere else)
    resolution: {donor_idx: "family" | "leaf"} — annotation depth (default
                "leaf")
    abundance:  {donor_idx: scale} — multiplies cells_per_leaf (imbalance)
    rare_private: dict(donor=idx, name=str, prevalence=float,
                parent=family) — the flagship case: a rare private subtype
                inside `parent`, at `prevalence` of that donor's parent
                cells (cells are relabeled from the parent's budget, so
                power depends on the donor's sampling)

    Returns dict with per-donor counts (cells x genes), labels, latent leaf
    per cell, and ground truth incl. per-(donor, clade) presence and the
    expected-prevalence table the eligibility model consumes.
    """
    rs = np.random.RandomState(seed)
    lt = latent_tree(n_family, n_sub)
    missing = missing or {}
    private = private or {}
    resolution = resolution or {}
    abundance = abundance or {}

    genes = np.arange(n_genes)
    base_mu = rs.lognormal(0, 1, n_genes)
    pool = rs.permutation(n_genes)
    fam_idx = {f: pool[i * 40:(i + 1) * 40]
               for i, f in enumerate(lt["families"])}
    off = 40 * n_family
    sub_idx = {l: pool[off + i * 15: off + (i + 1) * 15]
               for i, l in enumerate(lt["leaves"])}
    extra = pool[off + 15 * len(lt["leaves"]):]

    def leaf_mu(leaf, priv_block=None):
        mu = base_mu.copy()
        if priv_block is not None:
            mu[priv_block] *= np.exp(1.5)
            return mu
        mu[fam_idx[lt["family_of"][leaf]]] *= np.exp(fam_lfc)
        mu[sub_idx[leaf]] *= np.exp(sub_lfc)
        return mu

    donors, truth_presence, prevalence = [], {}, {}
    extra_used = 0
    for d in range(K):
        drop = set(missing.get(d, []))
        present = [l for l in lt["leaves"]
                   if l not in drop and lt["family_of"][l] not in drop]
        n_per = max(10, int(round(cells_per_leaf * abundance.get(d, 1.0))))
        batch = rs.lognormal(0, batch_sd, n_genes)
        blocks, latent = [], []
        for l in present:
            lam = np.outer(rs.gamma(10, 0.1, n_per), leaf_mu(l) * batch)
            blocks.append(rs.poisson(lam))
            latent += [l] * n_per
        for name in private.get(d, []):
            pb = extra[extra_used * 20:(extra_used + 1) * 20]
            extra_used += 1
            lam = np.outer(rs.gamma(10, 0.1, n_per),
                           leaf_mu(None, pb) * batch)
            blocks.append(rs.poisson(lam))
            latent += [name] * n_per
            truth_presence[(d, name)] = "private"
        if rare_private and rare_private["donor"] == d:
            par = rare_private["parent"]
            p = rare_private["prevalence"]
            par_cells = [i for i, l in enumerate(latent)
                         if lt["family_of"].get(l) == par]
            n_rare = max(1, int(round(p * len(par_cells))))
            take = rs.choice(par_cells, n_rare, replace=False)
            pb = extra[extra_used * 20:(extra_used + 1) * 20]
            extra_used += 1
            mu = leaf_mu(latent[take[0]])
            mu[pb] *= np.exp(1.5)
            X = np.vstack(blocks)
            lam = np.outer(rs.gamma(10, 0.1, n_rare), mu * batch)
            X[take] = rs.poisson(lam)
            blocks = [X]
            for i in take:
                latent[i] = rare_private["name"]
            truth_presence[(d, rare_private["name"])] = "private"
            prevalence[rare_private["name"]] = p
        X = np.vstack(blocks).astype(float)
        latent = np.asarray(latent)
        if resolution.get(d, "leaf") == "family":
            labels = np.asarray([lt["family_of"].get(l, l) for l in latent])
        else:
            labels = latent.copy()
        for l in lt["leaves"]:
            truth_presence.setdefault(
                (d, l), "present" if l in present else "absent")
        donors.append({"counts": X, "labels": labels, "latent": latent,
                       "resolution": resolution.get(d, "leaf")})
    return {"donors": donors, "latent": lt,
            "truth_presence": truth_presence,
            "prevalence": prevalence, "seed": seed}


def scenario(name, seed=0, **kw):
    """The four prespecified validation scenarios (DESIGN.md)."""
    if name == "batch":
        return simulate_donors(K=3, batch_sd=1.0, seed=seed, **kw)
    if name == "missing_unique":
        return simulate_donors(K=3, seed=seed, missing={1: ["F4"]},
                               private={2: ["P1"]}, **kw)
    if name == "resolution_imbalance":
        return simulate_donors(K=3, seed=seed, resolution={0: "family"},
                               abundance={0: 4.0, 2: 0.25}, **kw)
    if name == "rare_private":
        # the flagship: 2% private subtype inside F1, in the SMALLEST donor
        return simulate_donors(
            K=3, seed=seed, abundance={0: 5.0, 1: 1.0, 2: 0.3},
            rare_private=dict(donor=2, name="F1.rare", prevalence=0.02,
                              parent="F1"), **kw)
    raise ValueError(name)
