"""Projection prototype gates + the review's adversarial checks:

Core: vote parity with the kernel; held-out-batch accuracy with ~0
wrong-family; family-only cells abstain at family; novel-family cells
resolve shallow with separable OOR; determinism.

Adversarial (review round 1):
A. REFINEMENT NULL — one leaf vs ten exchangeable leaves under a pure
   null must NOT bias assignment or confidence toward the leaf-rich
   branch (multiplicity calibration).
B. TIES — identical A/B reference profiles must abstain, invariant to
   reference row order, label renaming, and atlas order.
C. QUERY COMPOSITION — a cell projected alone equals the same cell
   projected in any batch (fitted projector, not transductive).
D. INTERNAL-NODE LABELS — coarse labels inform splits above their node
   only; a coarse-only reference never resolves below its labels; the
   from_harmonize adapter round-trips.
E. SPARSE — scipy.sparse inputs equal dense results.
F. UNEQUAL COVERAGE — a reference missing one child's labels is
   excluded from that split but still contributes elsewhere.
"""
import numpy as np
import pytest
from scipy import sparse as sp

from metaarbor import tree_from_levels
from metaarbor.kernel import vote_cache
from metaarbor.projection import (_rank_norm, build_projector,
                                  from_harmonize, project)

GENES = [f"g{i}" for i in range(900)]
FAM = {f"F{i}.s{j}": f"F{i}" for i in (1, 2, 3) for j in (1, 2)}


def sim(leaves, batch_seed, n_per=60, fam_only=None, extra_latents=None,
        rename=None):
    rs = np.random.RandomState(batch_seed)
    base = np.random.RandomState(99).lognormal(0, 1, len(GENES))
    pool = np.random.RandomState(98).permutation(len(GENES))
    fams = sorted(set(FAM.values()) | set(extra_latents or []))
    fam_idx = {f: pool[i * 40:(i + 1) * 40] for i, f in enumerate(fams)}
    subs = sorted(FAM)
    sub_idx = {s: pool[len(fams) * 40 + i * 15:
                       len(fams) * 40 + (i + 1) * 15]
               for i, s in enumerate(subs)}
    batch = rs.lognormal(0, 0.4, len(GENES))
    blocks, labels = [], []
    for l in leaves:
        f = FAM.get(l, l)
        mu = base.copy()
        mu[fam_idx[f]] *= np.exp(1.2)
        if l in FAM and f not in (fam_only or []):
            mu[sub_idx[l]] *= np.exp(1.3)
        lam = np.outer(rs.gamma(10, 0.1, n_per), mu * batch)
        blocks.append(rs.poisson(lam))
        labels += [(rename or {}).get(l, l)] * n_per
    return (np.vstack(blocks).astype(float), np.asarray(labels))


def ref_tree():
    return tree_from_levels([(FAM[l], l) for l in sorted(FAM)],
                            ["family", "leaf"])


@pytest.fixture(scope="module")
def world():
    tree = ref_tree()
    xa, la = sim(sorted(FAM), batch_seed=11)
    xb, lb = sim(sorted(FAM), batch_seed=12)
    proj = build_projector(
        [{"counts": xa, "labels": la, "gene_names": GENES, "name": "A"},
         {"counts": xb, "labels": lb, "gene_names": GENES, "name": "B"}],
        tree, cap_per_label=40, seed=0)
    return tree, proj


# ---- core gates -----------------------------------------------------------
def test_global_vote_parity_with_kernel(world):
    tree, proj = world
    xq, _ = sim(sorted(FAM), batch_seed=2, n_per=8)
    out = project(proj, xq, GENES)
    # recompute one reference's global label votes via kernel.vote_cache
    ref = proj["refs"][0]
    from metaarbor.kernel import lognorm
    qpos = {g: i for i, g in enumerate(GENES)}
    ridx = np.asarray([i for i, g in enumerate(ref["hvg_names"])])
    qidx = np.asarray([qpos[g] for g in ref["hvg_names"]])
    qn = _rank_norm(lognorm(xq)[:, qidx])
    rn = _rank_norm(ref["ln_hvg"][:, ridx])
    kv = vote_cache(qn, rn, ref["labels"])
    mean_kernel = kv["V"] / kv["leaf_sizes"][None, :]
    # our label_vote averages over 2 refs; recompute ref-1 side and
    # verify the two-ref average reproduces
    ref2 = proj["refs"][1]
    qidx2 = np.asarray([qpos[g] for g in ref2["hvg_names"]])
    qn2 = _rank_norm(lognorm(xq)[:, qidx2])
    rn2 = _rank_norm(ref2["ln_hvg"])
    kv2 = vote_cache(qn2, rn2, ref2["labels"])
    mean2 = kv2["V"] / kv2["leaf_sizes"][None, :]
    lab = out["labels"]
    want = (mean_kernel[:, [kv["leaves"].index(l) for l in lab]] +
            mean2[:, [kv2["leaves"].index(l) for l in lab]]) / 2
    assert np.allclose(out["label_vote"], want, atol=1e-10)


def test_heldout_accuracy_and_no_wrong_family(world):
    tree, proj = world
    xq, lq = sim(sorted(FAM), batch_seed=33)
    out = project(proj, xq, GENES)
    fam_true = np.asarray([FAM[l] for l in lq])
    fam_pred = np.asarray([FAM.get(b, b) for b in out["best_label"]])
    assert np.mean(out["best_label"] == lq) >= 0.9
    assert np.mean(fam_pred == fam_true) >= 0.98
    deep = out["resolved_depth"] >= 2
    if deep.any():
        assert np.mean(fam_pred[deep] != fam_true[deep]) <= 0.01
    assert np.mean(deep) >= 0.45, np.mean(deep)


def test_family_only_cells_abstain_at_family(world):
    tree, proj = world
    xq, _ = sim([l for l in sorted(FAM) if FAM[l] == "F1"],
                batch_seed=44, fam_only=["F1"])
    out = project(proj, xq, GENES)
    fam_pred = np.asarray([FAM.get(b, b) for b in out["best_label"]])
    assert np.mean(fam_pred == "F1") >= 0.95
    # never resolves BEYOND the family (root stops are conservative,
    # not wrong: these cells' profiles genuinely lack the subtype
    # component the reference carries)
    assert np.mean(out["resolved_depth"] <= 1) >= 0.9
    assert np.mean(out["resolved_depth"] >= 2) <= 0.1


def test_novel_family_shallow_and_low_oor(world):
    tree, proj = world
    xq, _ = sim(["FX"], batch_seed=55, extra_latents=["FX"])
    out_nov = project(proj, xq, GENES)
    xk, _ = sim(sorted(FAM), batch_seed=56)
    out_in = project(proj, xk, GENES)
    assert (np.median(out_nov["resolved_depth"]) <
            np.median(out_in["resolved_depth"]))
    from metaarbor.kernel import auroc
    score = np.concatenate([out_in["max_label_vote"],
                            out_nov["max_label_vote"]])
    is_in = np.concatenate([np.ones(len(xk)) if False else
                            np.ones(out_in["max_label_vote"].shape),
                            np.zeros(out_nov["max_label_vote"].shape)])
    assert auroc(score, is_in == 1) >= 0.8
    assert np.mean(out_nov["resolved_depth"] >= 2) <= 0.1


def test_determinism_and_blocking(world):
    tree, proj = world
    xq, _ = sim(sorted(FAM), batch_seed=77, n_per=15)
    a = project(proj, xq, GENES)
    b = project(proj, xq, GENES, block=17)      # odd block size
    assert list(a["best_label"]) == list(b["best_label"])
    assert list(a["resolved_node"]) == list(b["resolved_node"])
    assert np.allclose(a["path_margins"], b["path_margins"],
                       equal_nan=True)


# ---- A. refinement null ---------------------------------------------------
def test_refinement_null_no_multiplicity_bias():
    """One leaf under A, ten exchangeable leaves under B, ALL drawn
    from the same latent: assignment must stay ~balanced and almost
    nothing may pass the root confidently."""
    rs = np.random.RandomState(0)
    base = np.random.RandomState(99).lognormal(0, 1, len(GENES))
    def draw(n, seed):
        r = np.random.RandomState(seed)
        batch = r.lognormal(0, 0.3, len(GENES))
        return r.poisson(np.outer(r.gamma(10, 0.1, n), base * batch))
    labels = ["A1"] + [f"B{j}" for j in range(1, 11)]
    tree = tree_from_levels([("A", "A1")] +
                            [("B", f"B{j}") for j in range(1, 11)],
                            ["branch", "leaf"])
    X = np.vstack([draw(40, 10 + i) for i in range(len(labels))])
    L = np.concatenate([[l] * 40 for l in labels])
    proj = build_projector([{"counts": X.astype(float), "labels": L,
                             "gene_names": GENES}], tree,
                           cap_per_label=40)
    xq = draw(300, 999).astype(float)
    out = project(proj, xq, GENES)
    into_b = np.mean([b.startswith("B") for b in out["best_label"]])
    assert 0.25 <= into_b <= 0.75, into_b       # no leaf-count pull
    conf_deep = np.mean(out["resolved_depth"] >= 1)
    assert conf_deep <= 0.15, conf_deep         # null: almost all abstain


def test_refinement_relabeling_same_cells():
    """The stronger invariant: the SAME reference cells, with one
    population relabeled as 1, 2, or 10 pseudo-leaves. Parent CHOICE
    must be unchanged for signal cells, and the null pass-rate must
    stay bounded under every relabeling. (Margin magnitudes shift with
    block size — the statistic is multiplicity-ADJUSTED, and exact
    refinement invariance of magnitudes is not claimed.)"""
    rs = np.random.RandomState(0)
    base = np.random.RandomState(99).lognormal(0, 1, len(GENES))
    pool = np.random.RandomState(98).permutation(len(GENES))
    a_idx, b_idx = pool[:40], pool[40:80]

    def draw(n, seed, prog=None):
        r = np.random.RandomState(seed)
        mu = base.copy()
        if prog is not None:
            mu[prog] *= np.exp(1.3)
        return r.poisson(np.outer(r.gamma(10, 0.1, n), mu))

    XA = draw(60, 1, a_idx)
    XB = draw(120, 2, b_idx)                  # ONE homogeneous B pool
    X = np.vstack([XA, XB]).astype(float)
    xq_sig = draw(80, 7, b_idx).astype(float)   # true B cells
    xq_nul = draw(200, 8, None).astype(float)   # neither program

    prev_choice = None
    for k in (1, 2, 10):
        bl = np.concatenate([np.full(60, "A1", dtype=object),
                             np.asarray([f"B{j % k}" for j in
                                         range(120)], dtype=object)])
        rows = [("A", "A1")] + [("B", f"B{j}") for j in range(k)]
        tree = tree_from_levels(rows, ["branch", "leaf"])
        proj = build_projector([{"counts": X, "labels": bl,
                                 "gene_names": GENES}], tree,
                               cap_per_label=200)
        o_sig = project(proj, xq_sig, GENES)
        o_nul = project(proj, xq_nul, GENES)
        choice = np.asarray([b.startswith("B") or b == "B"
                             for b in o_sig["best_label"]])
        assert choice.mean() >= 0.95, (k, choice.mean())
        if prev_choice is not None:
            assert np.mean(choice == prev_choice) >= 0.95
        prev_choice = choice
        # bounded null leakage under every relabeling
        assert np.mean(o_nul["resolved_depth"] >= 1) <= 0.15, k


def test_from_harmonize_maps_affiliates():
    harm = {"tree": {
        "MA-C0001": {"parent": None, "members": {"A": "A|F1",
                                                 "B": "B|F1"}},
        "MA-C0002": {"parent": "MA-C0001",
                     "members": {"A": "A|F1.s1", "B": "B|F1.s1"}},
    }, "affiliates": [{"dataset": "B", "node": "B|twin",
                       "attached_to": "MA-C0002",
                       "candidate_id": "cand:0001"}]}
    tree, lmaps = from_harmonize(harm)
    assert lmaps["B"]["B|twin"] == "MA-C0002"
    assert lmaps["B"]["B|F1.s1"] == "MA-C0002"   # member wins/coexists
    assert lmaps["A"]["A|F1"] == "MA-C0001"


# ---- B. ties and permutation invariance -----------------------------------
def test_ties_abstain_and_order_invariance():
    """Two reference labels with IDENTICAL profiles: the split must
    abstain; results must not change under reference row permutation,
    label renaming, or atlas order."""
    rs = np.random.RandomState(3)
    base = np.random.RandomState(99).lognormal(0, 1, len(GENES))
    def draw(n, seed):
        r = np.random.RandomState(seed)
        return r.poisson(np.outer(r.gamma(10, 0.1, n), base))
    X = np.vstack([draw(40, 1), draw(40, 2)]).astype(float)
    L = np.asarray(["A"] * 40 + ["B"] * 40)
    tree = tree_from_levels([("A",), ("B",)], ["leaf"])
    xq = draw(60, 7).astype(float)

    p1 = build_projector([{"counts": X, "labels": L,
                           "gene_names": GENES}], tree)
    o1 = project(p1, xq, GENES)
    assert np.mean(o1["resolved_depth"] == 0) >= 0.9   # ties -> abstain

    perm = rs.permutation(len(L))
    p2 = build_projector([{"counts": X[perm], "labels": L[perm],
                           "gene_names": GENES}], tree)
    o2 = project(p2, xq, GENES)
    assert list(o1["resolved_node"]) == list(o2["resolved_node"])
    assert np.allclose(o1["path_margins"], o2["path_margins"],
                       equal_nan=True, atol=1e-9)

    # label renaming: swap names A<->Z (Z sorts after B)
    ren = np.where(L == "A", "Z", "B")
    tree_r = tree_from_levels([("Z",), ("B",)], ["leaf"])
    p3 = build_projector([{"counts": X, "labels": ren,
                           "gene_names": GENES}], tree_r)
    o3 = project(p3, xq, GENES)
    assert np.mean(o3["resolved_depth"] == 0) >= 0.9


def test_atlas_order_invariance(world):
    tree, _ = world
    xa, la = sim(sorted(FAM), batch_seed=11)
    xb, lb = sim(sorted(FAM), batch_seed=12)
    ra = {"counts": xa, "labels": la, "gene_names": GENES, "name": "A"}
    rb = {"counts": xb, "labels": lb, "gene_names": GENES, "name": "B"}
    xq, _ = sim(sorted(FAM), batch_seed=21, n_per=12)
    o_ab = project(build_projector([ra, rb], tree, cap_per_label=40),
                   xq, GENES)
    o_ba = project(build_projector([rb, ra], tree, cap_per_label=40),
                   xq, GENES)
    assert list(o_ab["best_label"]) == list(o_ba["best_label"])
    assert list(o_ab["resolved_node"]) == list(o_ba["resolved_node"])


# ---- C. query composition invariance --------------------------------------
def test_query_composition_invariance(world):
    tree, proj = world
    xq, _ = sim(sorted(FAM), batch_seed=88, n_per=5)
    batch = project(proj, xq, GENES)
    for i in (0, 7, 13, 29):
        single = project(proj, xq[i:i + 1], GENES)
        assert single["best_label"][0] == batch["best_label"][i]
        assert single["resolved_node"][0] == batch["resolved_node"][i]
        assert np.allclose(single["path_margins"][0],
                           batch["path_margins"][i], equal_nan=True)


# ---- D. internal-node labels + harmonize adapter --------------------------
def test_coarse_only_reference_never_resolves_below_its_labels():
    tree = ref_tree()
    x, l = sim(sorted(FAM), batch_seed=5)
    fam_labels = np.asarray([FAM[i] for i in l])
    lmap = {f: f"family:{f}" for f in sorted(set(fam_labels))}
    proj = build_projector([{"counts": x, "labels": fam_labels,
                             "gene_names": GENES}], tree,
                           label_maps=[lmap])
    xq, lq = sim(sorted(FAM), batch_seed=6, n_per=20)
    out = project(proj, xq, GENES)
    # family split resolvable; below it, no reference signal exists
    assert np.mean(out["resolved_depth"] <= 1) == 1.0
    # CONTRACT: best_label is a real reference label; best_node is a
    # VALID TREE NODE (never an invented leaf under a coarse reference)
    valid_nodes = set(tree["parent"]) | {"root"}
    assert set(out["best_label"]) <= set(fam_labels)
    assert set(out["best_node"]) <= valid_nodes
    assert all(b.startswith("family:") for b in out["best_node"])
    fam_true = np.asarray([FAM[i] for i in lq])
    got_fam = np.asarray(
        [r.split(":")[-1] if r != "root" else "root"
         for r in out["resolved_node"]])
    ok = got_fam[out["resolved_depth"] == 1] == \
        fam_true[out["resolved_depth"] == 1]
    assert ok.mean() >= 0.95


def test_mixed_resolution_references(world):
    """Fine ref + coarse ref together: subtype splits are informed by
    the fine ref alone; the coarse ref still shapes the family split."""
    tree = ref_tree()
    xa, la = sim(sorted(FAM), batch_seed=11)
    xc, lc = sim(sorted(FAM), batch_seed=13)
    fam_c = np.asarray([FAM[i] for i in lc])
    lmap = {f: f"family:{f}" for f in sorted(set(fam_c))}
    proj = build_projector(
        [{"counts": xa, "labels": la, "gene_names": GENES},
         {"counts": xc, "labels": fam_c, "gene_names": GENES}],
        tree, label_maps=[None, lmap] if False else
        [{l: l for l in set(la)}, lmap])
    xq, lq = sim(sorted(FAM), batch_seed=14, n_per=20)
    out = project(proj, xq, GENES)
    assert np.mean(out["best_label"] == lq) >= 0.85
    deep = out["resolved_depth"] >= 2
    fam_pred = np.asarray([FAM.get(b, b) for b in out["best_label"]])
    fam_true = np.asarray([FAM[i] for i in lq])
    if deep.any():
        assert np.mean(fam_pred[deep] != fam_true[deep]) <= 0.02


def test_from_harmonize_roundtrip(world):
    from metaarbor.consensus.harmonize import harmonize
    fam = {f"F{i}.s{j}": f"F{i}" for i in (1, 2) for j in (1, 2)}
    leaves = sorted(fam)
    trees = {k: tree_from_levels([(f"{k}|{fam[l]}", f"{k}|{l}")
                                  for l in leaves], ["family", "leaf"])
             for k in ("A", "B")}
    ds = {}
    for i, k in enumerate(("A", "B")):
        x, l = sim(leaves, batch_seed=30 + i)
        ds[k] = {"counts": x,
                 "labels": np.asarray([f"{k}|{v}" for v in l]),
                 "gene_names": GENES}
    harm = harmonize(ds, trees, n_hvg=600, n_boot=60, trust_trees=True)
    tree, lmaps = from_harmonize(harm)
    proj = build_projector(
        [{"counts": ds["A"]["counts"], "labels": ds["A"]["labels"],
          "gene_names": GENES}], tree, label_maps=[lmaps["A"]])
    xq, lq = sim(leaves, batch_seed=41, n_per=10)
    out = project(proj, xq, GENES)
    assert len(out["best_label"]) == len(lq)     # runs end to end
    assert set(out["resolved_node"]) <= (set(harm["tree"]) | {"root"})


# ---- E. sparse inputs -----------------------------------------------------
def test_sparse_equals_dense(world):
    tree, _ = world
    xa, la = sim(sorted(FAM), batch_seed=11)
    xq, _ = sim(sorted(FAM), batch_seed=19, n_per=8)
    pd_ = build_projector([{"counts": xa, "labels": la,
                            "gene_names": GENES}], tree,
                          cap_per_label=40)
    ps = build_projector([{"counts": sp.csr_matrix(xa), "labels": la,
                           "gene_names": GENES}], tree,
                         cap_per_label=40)
    od = project(pd_, xq, GENES)
    os_ = project(ps, sp.csr_matrix(xq), GENES)
    assert list(od["best_label"]) == list(os_["best_label"])
    assert np.allclose(od["label_vote"], os_["label_vote"], atol=1e-10)


# ---- F. unequal coverage --------------------------------------------------
def test_unequal_coverage_excluded_per_split(world):
    """A second reference lacking family F2 entirely must not distort
    the F2-subtype split (it is excluded there) while still informing
    the root split for the families it has."""
    tree = ref_tree()
    xa, la = sim(sorted(FAM), batch_seed=11)
    keepers = [l for l in sorted(FAM) if FAM[l] != "F2"]
    xb, lb = sim(keepers, batch_seed=12)
    proj = build_projector(
        [{"counts": xa, "labels": la, "gene_names": GENES},
         {"counts": xb, "labels": lb, "gene_names": GENES}], tree,
        cap_per_label=40)
    xq, lq = sim(sorted(FAM), batch_seed=61)
    out = project(proj, xq, GENES)
    f2 = np.asarray([FAM[l] == "F2" for l in lq])
    acc_f2 = np.mean(out["best_label"][f2] == lq[f2])
    assert acc_f2 >= 0.85, acc_f2


# ---- G. crossed partial coverage (round-4 P1) -----------------------------
def test_crossed_coverage_split_abstains_and_full_ref_governs():
    """Three children; ref A covers F1+F2, ref B covers F2+F3 (crossed
    partial coverage). Under the formal combination rule NEITHER
    contributes to the root split (children must be compared on
    identical reference subsets), so the split abstains — no child can
    be favored by having more covering atlases. Adding a reference C
    that covers all three children makes the projection EQUAL to the
    C-only projection at that split (only full-coverage refs govern)."""
    tree = ref_tree()
    leaves_a = [l for l in sorted(FAM) if FAM[l] != "F3"]
    leaves_b = [l for l in sorted(FAM) if FAM[l] != "F1"]
    xa, la = sim(leaves_a, batch_seed=11)
    xb, lb = sim(leaves_b, batch_seed=12)
    xc, lc = sim(sorted(FAM), batch_seed=13)
    xq, lq = sim(sorted(FAM), batch_seed=71, n_per=15)

    crossed = build_projector(
        [{"counts": xa, "labels": la, "gene_names": GENES, "name": "A"},
         {"counts": xb, "labels": lb, "gene_names": GENES, "name": "B"}],
        tree, cap_per_label=40)
    oc = project(crossed, xq, GENES)
    # root split uncovered by every reference -> universal abstention
    assert np.all(oc["resolved_depth"] == 0)
    # and best_node is always a REAL tree node, never invented
    valid = set(tree["parent"]) | {"root"}
    assert set(oc["best_node"]) <= valid

    both = build_projector(
        [{"counts": xa, "labels": la, "gene_names": GENES, "name": "A"},
         {"counts": xb, "labels": lb, "gene_names": GENES, "name": "B"},
         {"counts": xc, "labels": lc, "gene_names": GENES,
          "name": "C"}], tree, cap_per_label=40)
    only_c = build_projector(
        [{"counts": xc, "labels": lc, "gene_names": GENES,
          "name": "C"}], tree, cap_per_label=40)
    ob = project(both, xq, GENES)
    os_ = project(only_c, xq, GENES)
    # at the ROOT split only C governs in both projectors, so the
    # root-level choice must be identical
    root_choice_b = np.asarray([FAM.get(b, b) for b in ob["best_label"]])
    root_choice_c = np.asarray([FAM.get(b, b)
                                for b in os_["best_label"]])
    assert np.mean(root_choice_b == root_choice_c) >= 0.95
    assert np.allclose(ob["path_margins"][:, 0],
                       os_["path_margins"][:, 0], equal_nan=True)


# ---- H. label collisions (round-4 P1) -------------------------------------
def test_label_collision_across_atlases_raises():
    tree = ref_tree()
    xa, la = sim(sorted(FAM), batch_seed=11)
    xb, lb = sim(sorted(FAM), batch_seed=12)
    # same label string, DIFFERENT nodes: must raise
    lmapA = {l: l for l in set(la)}
    lmapB = {l: l for l in set(lb)}
    lmapB["F1.s1"] = "F1.s2"                 # conflict with A's mapping
    with pytest.raises(ValueError, match="maps to node"):
        build_projector(
            [{"counts": xa, "labels": la, "gene_names": GENES,
              "name": "A"},
             {"counts": xb, "labels": lb, "gene_names": GENES,
              "name": "B"}], tree, label_maps=[lmapA, lmapB],
            cap_per_label=20)
    # same label, SAME node (shared taxonomy): allowed
    build_projector(
        [{"counts": xa, "labels": la, "gene_names": GENES, "name": "A"},
         {"counts": xb, "labels": lb, "gene_names": GENES,
          "name": "B"}], tree, cap_per_label=20)


def test_gene_panel_fixes_feature_space():
    tree = ref_tree()
    xa, la = sim(sorted(FAM), batch_seed=11)
    panel = GENES[:400]
    p1 = build_projector([{"counts": xa, "labels": la,
                           "gene_names": GENES, "name": "A"}], tree,
                         cap_per_label=20, gene_panel=panel)
    assert set(p1["refs"][0]["hvg_names"]) <= set(panel)
    p2 = build_projector([{"counts": xa, "labels": la,
                           "gene_names": GENES, "name": "A"}], tree,
                         cap_per_label=40, gene_panel=panel)
    # with a supplied panel the feature space no longer depends on cap
    assert p1["refs"][0]["hvg_names"] == p2["refs"][0]["hvg_names"]
