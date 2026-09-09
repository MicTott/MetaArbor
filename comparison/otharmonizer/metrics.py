"""Self-contained port of OTHarmonizer's evaluation metrics — TEDS,
PCBS, AH-F1 — faithful to github.com/Duck-Boss/OTHarmonizer
(OTHarmonizer/benchmark.py + tree.py), so MetaArbor comparisons can be
scored with THEIR metrics without cloning their repository.

Fidelity notes (quirks preserved deliberately; parity-tested against
their code in test_metrics.py):
- TEDS normalizes by (size1 + size2 - 1) where each size is the zss
  distance of the tree to a single node labeled with the SET {'root'}
  (their exact construction), and children are ordered by their
  convert_to_zssNode sort followed by addkid_with_order's
  first-character insertion.
- PCBS is their compute_match_score variant (mean best-Jaccard-overlap
  of parent-branch label sets, both directions), not the RFS variant.
- AH-F1 averages F1 over parent-child / equal / none-relation edge
  sets extracted from '&'-merged labels. ONE deliberate deviation:
  upstream count_f1 raises ZeroDivisionError when either tree lacks a
  relation category entirely (e.g. no '&' equals anywhere); this port
  scores that component 0.0 instead.

Input trees are nested dicts {"label": str, "children": [...]} — the
format of the committed oth_tree_*.json files — or MyNode objects.
'&' inside a label denotes merged/equal annotations.
"""
from __future__ import annotations

import itertools

import zss


class MyNode:
    def __init__(self, label):
        self.label = label
        self.parent = None
        self.children = []

    def addkid(self, node):
        node.parent = self
        self.children.append(node)
        return self

    def get_all_descendants_labels(self):
        out = [self.label]
        for c in self.children:
            out.extend(c.get_all_descendants_labels())
        return out


def from_nested(d):
    n = MyNode(str(d["label"]))
    for c in d.get("children", []):
        n.addkid(from_nested(c))
    return n


# ---- their zss conversion, verbatim behavior ------------------------------
class _ZssNode(zss.Node):
    def __init__(self, label):
        super().__init__(label)
        self.children = []

    def addkid_with_order(self, node):
        idx = 0
        while idx < len(self.children):
            if list(node.label)[0] < list(self.children[idx].label)[0]:
                break
            idx += 1
        self.children.insert(idx, node)
        return self


def _to_zss(my_node):
    labels_sorted = sorted(my_node.label.split("&"))
    z = _ZssNode("&".join(labels_sorted))
    for child in sorted(my_node.children,
                        key=lambda x: "&".join(sorted(
                            x.label.split("&")))):
        z.addkid_with_order(_to_zss(child))
    return z


def teds(constructed, ref):
    """(TED, TEDS) exactly as their TEDS()."""
    root_node = zss.Node({"root"})
    cz, rz = _to_zss(constructed), _to_zss(ref)
    size_c = zss.simple_distance(cz, root_node)
    size_r = zss.simple_distance(rz, root_node)
    dist = zss.simple_distance(cz, rz)
    return dist, 1 - dist / (size_c + size_r - 1)


# ---- PCBS ----------------------------------------------------------------
def _copy_setlabel(node):
    n = MyNode({" & ".join(sorted(node.label))})
    for c in node.children:
        n.addkid(_copy_setlabel(c))
    return n


def _collect_sets(node):
    sets = []
    labels = node.label
    if node.children:
        for child in node.children:
            labels = labels.union(child.label)
            sets.extend(_collect_sets(child))
        sets.append(labels)
    return sets


def pcbs(constructed, ref):
    """(PCB, RFS, PCBS) exactly as their PCBS()."""
    nc, nr = _copy_setlabel(constructed), _copy_setlabel(ref)
    lc, lr = _collect_sets(nc), _collect_sets(nr)
    diffs = [s for s in lc if s not in lr] + \
            [s for s in lr if s not in lc]
    total = len(lc) + len(lr)
    rfs = 1 - len(diffs) / total
    score = 0.0
    for s1 in lc:
        best = max((len(s1 & s2) for s2 in lr), default=0)
        score += best / len(s1)
    for s2 in lr:
        best = max((len(s1 & s2) for s1 in lc), default=0)
        score += best / len(s2)
    return len(diffs), rfs, score / (len(lc) + len(lr))


# ---- AH-F1 ---------------------------------------------------------------
def _relationships(node, rel=None, seen_equal=None):
    if rel is None:
        rel = {"parent-child": [], "equal": [], "none-relation": []}
        seen_equal = set()
    parents = node.label.split("&") if "&" in node.label else \
        [node.label]
    for child in node.children:
        parts = child.label.split("&") if "&" in child.label else \
            [child.label]
        for p in parents:
            for l_ in parts:
                rel["parent-child"].append(f"{p}:{l_}")
        _relationships(child, rel, seen_equal)
    for label in node.get_all_descendants_labels():
        if "&" in label:
            parts = label.split("&")
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    pair = tuple(sorted([parts[i], parts[j]]))
                    if pair not in seen_equal:
                        seen_equal.add(pair)
                        rel["equal"].append({parts[i], parts[j]})
    return rel


def _none_relations(node, rel):
    all_nodes = set()
    for label in node.get_all_descendants_labels():
        all_nodes.update(label.split("&") if "&" in label else [label])
    out = []
    for a, b in itertools.product(sorted(all_nodes), repeat=2):
        if a == b:
            continue
        if f"{a}:{b}" in rel["parent-child"]:
            continue
        if f"{b}:{a}" in rel["parent-child"]:
            continue
        if {a, b} in rel["equal"]:
            continue
        if {a, b} not in out:
            out.append({a, b})
    return out


def _f1(query, ref):
    common = [x for x in query if x in ref]
    if not ref or not query:
        return 0.0
    recall = len(common) / len(ref)
    precision = len(common) / len(query)
    if recall == 0 and precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def ah_f1(constructed, ref):
    rq = _relationships(constructed)
    rr = _relationships(ref)
    rq["none-relation"] = _none_relations(constructed, rq)
    rr["none-relation"] = _none_relations(ref, rr)
    return (_f1(rq["parent-child"], rr["parent-child"]) +
            _f1(rq["equal"], rr["equal"]) +
            _f1(rq["none-relation"], rr["none-relation"])) / 3


def score_all(constructed, ref):
    """{'TED', 'TEDS', 'PCB', 'RFS', 'PCBS', 'AH_F1'} for two MyNode
    trees (or nested dicts)."""
    if isinstance(constructed, dict):
        constructed = from_nested(constructed)
    if isinstance(ref, dict):
        ref = from_nested(ref)
    ted, teds_ = teds(constructed, ref)
    pcb, rfs, pcbs_ = pcbs(constructed, ref)
    return {"TED": float(ted), "TEDS": float(teds_), "PCB": int(pcb),
            "RFS": float(rfs), "PCBS": float(pcbs_),
            "AH_F1": float(ah_f1(constructed, ref))}
