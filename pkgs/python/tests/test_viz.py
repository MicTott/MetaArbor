"""Smoke + convention gates for metaarbor.viz (the audit renderer)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metaarbor.viz import (audit_legend, canonical_order,
                           nested_from_harmonize,
                           nested_from_input_tree, plot_audit_tree,
                           split_root_dumped)

GROUPS = {"a|x": "G1", "a|y": "G1", "b|x": "G2", "b|y": "G2"}
LABELS = set(GROUPS)


def nodes():
    return {"C1": {"parent": None, "status": "backbone",
                   "members": {"a": "a|x", "b": "b|x"}},
            "U1": {"parent": "C1", "status": "single_atlas",
                   "members": {"a": "a|y"}},
            "U2": {"parent": "C1", "status": "single_atlas",
                   "members": {"b": "b|y"}}}


def test_builders_and_smoke(tmp_path):
    t = nested_from_harmonize(nodes(), LABELS,
                              matched_calls={"a|y"})
    # edge kinds assigned per convention
    kids = {c["label"]: c for c in t["children"][0:1][0]["children"]} \
        if t["children"] and t["children"][0]["children"] else {}
    fig, ax = plt.subplots()
    order = canonical_order(GROUPS)
    plot_audit_tree(ax, t, order=order, groups=GROUPS,
                    title="t", subtitle="s")
    audit_legend(fig, groups=GROUPS, dataset_names=["a", "b"])
    fig.savefig(tmp_path / "smoke.png")
    plt.close(fig)


def test_edge_kinds():
    t = nested_from_harmonize(nodes(), LABELS,
                              matched_calls={"a|y"})
    top = t["children"][0]
    assert top["edge"] == "certified"
    by = {c["label"]: c["edge"] for c in top["children"]}
    assert by["a|y"] == "containment"      # matched one-way call
    assert by["b|y"] == "ancestry"         # no call


def test_no_groups_degrades(tmp_path):
    t = nested_from_input_tree(
        {"children": {"root": ["n1"], "n1": ["a|x", "a|y"],
                      "a|x": [], "a|y": []}}, LABELS)
    fig, ax = plt.subplots()
    plot_audit_tree(ax, t, order=sorted(LABELS))   # no groups: bands off
    fig.savefig(tmp_path / "nogroups.png")
    plt.close(fig)


def test_root_dumped_split():
    nested = {"label": "", "children": [
        {"label": "a|x&b|x", "children": [], "edge": "ancestry"},
        {"label": "a|y", "children": [], "edge": "ancestry"},
        {"label": "", "children": [
            {"label": "b|y", "children": [], "edge": "ancestry"}],
         "edge": "ancestry"}], "edge": "ancestry"}
    t, dumped = split_root_dumped(nested, LABELS)
    assert dumped == ["a|y"]
    assert len(t["children"]) == 2         # merge + subtree kept


def test_order_shared_layout_invariance():
    """Same canonical order -> identical leaf y assignment
    irrespective of child insertion order."""
    def tree(rev):
        ch = [{"label": "a|y", "children": [], "edge": "ancestry"},
              {"label": "a|x", "children": [], "edge": "ancestry"}]
        if rev:
            ch = ch[::-1]
        return {"label": "", "children": ch, "edge": "ancestry"}
    order = sorted(LABELS)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    t1, t2 = tree(False), tree(True)
    plot_audit_tree(ax1, t1, order=order)
    plot_audit_tree(ax2, t2, order=order)
    y1 = {c["label"]: c["_y"] for c in t1["children"]}
    y2 = {c["label"]: c["_y"] for c in t2["children"]}
    assert y1 == y2
    plt.close(fig)
