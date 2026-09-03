"""Tree-inference tests: family recovery on simulated data, polytomy
collapse under no structure, star policy, and the interpretation
annotation."""
import numpy as np
import pytest

from metaarbor import (annotate_star_relations, infer_tree, is_star_tree,
                       star_tree)


def _simulate(n_family=3, n_sub=3, n_genes=800, n_per=60, seed=5):
    rs = np.random.RandomState(seed)
    fams = [f"F{i+1}" for i in range(n_family)]
    leaves = [f"{f}.s{j+1}" for f in fams for j in range(n_sub)]
    base = rs.lognormal(0, 1, n_genes)
    pool = rs.permutation(n_genes)
    fam_idx = {f: pool[i * 40:(i + 1) * 40] for i, f in enumerate(fams)}
    off = 40 * n_family
    sub_idx = {l: pool[off + i * 15: off + (i + 1) * 15]
               for i, l in enumerate(leaves)}
    blocks, labels = [], []
    for l in leaves:
        mu = base.copy()
        mu[fam_idx[l.split(".")[0]]] *= np.exp(1.2)
        mu[sub_idx[l]] *= np.exp(0.9)
        lam = np.outer(rs.gamma(10, 0.1, n_per), mu)
        blocks.append(rs.poisson(lam))
        labels += [l] * n_per
    return np.vstack(blocks).astype(float), np.asarray(labels), fams, leaves


def test_recovers_latent_families():
    counts, labels, families, latent_leaves = _simulate()
    res = infer_tree(counts, labels, n_hvg=600, n_boot=30, seed=1)
    fams = {f: sorted(l for l in latent_leaves if l.startswith(f))
            for f in families}
    clade_sets = {tuple(v) for v in res["clades"].values()}
    hits = sum(tuple(v) in clade_sets for v in fams.values())
    assert hits == 3, (res["clades"], fams)
    assert all(s >= 0.7 for s in res["support"].values())
    assert not is_star_tree(res["tree"])


def test_no_structure_collapses_to_polytomy():
    rs = np.random.RandomState(0)
    counts = rs.poisson(2.0, size=(300, 300)).astype(float)
    labels = np.repeat([f"t{i}" for i in range(6)], 50)
    res = infer_tree(counts, labels, n_hvg=200, n_boot=30, seed=2)
    # iid noise: no clade should be strongly supported -> few/no internals
    assert res["provenance"]["n_internal_kept"] <= 1


def test_star_policy_and_annotation():
    with pytest.raises(ValueError):
        infer_tree(np.zeros((10, 5)), ["a"] * 5 + ["b"] * 5)  # < 3 labels
    st = star_tree(["x", "y", "z"])
    assert is_star_tree(st)
    rows = [{"query": "q", "walk_relation": "discordant"},
            {"query": "r", "walk_relation": "family"}]
    ann = annotate_star_relations(rows, st)
    assert ann[0]["walk_relation_note"] == "distributed_no_target_clade"
    assert "walk_relation_note" not in ann[1]
    assert ann[0]["walk_relation"] == "discordant"  # raw output preserved
