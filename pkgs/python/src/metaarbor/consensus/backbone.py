"""Hierarchical greedy consensus backbone with eligibility v2 (DESIGN.md
steps 6-7, corrected).

FROZEN THRESHOLDS (fixed before the simulation gates ran; changing them
against gate results un-freezes the method):

  POWER            0.95   detection probability required to count a
                          dataset as powered (eligible) for absence calls
  CRED_Q           0.05   lower credible bound quantile for single-donor
                          prevalence
  MIN_SUPPORT      0.5    minimum supported / eligible ratio for backbone
  MIN_DATASETS     2      minimum supporting datasets for backbone
  STABILITY_FLOOR  0.7    minimum within-dataset stability for private

Processing: ancestors before descendants (topological over the candidate
ancestry relation); within each parent context candidates rank by
(1) eligible-dataset support ratio, (2) number of supporting datasets,
(3) clique completeness (seed invariance), (4) mean bootstrap support,
(5) deterministic tie-break on sorted members.

Eligibility calls per (dataset, candidate), all RAW rows emitted:
  supported               dataset holds a member
  unresolved_in_dataset   no member, and EITHER (a) the parent member is
                          TERMINAL in this dataset's canonical tree (the
                          dataset cannot express anything below the
                          parent — resolution limitation), OR (b) the
                          dataset has FREE canonical structure below the
                          parent (nodes unclaimed by any multi-dataset
                          meta-clade): absence cannot then be
                          distinguished from unresolved local structure.
                          Only when everything below the parent is
                          claimed does powered absence become genuine
                          evidence. One-way walk landings are recorded
                          as asymmetric evidence but never drive this
                          call. Excluded from the support denominator,
                          never counted as absent. [Corrected twice by
                          the gates: the original parent-containment
                          rule structurally precluded private detection;
                          the landing-based revision let weak twins be
                          mis-called private.]
  private_or_absent       powered (LOO Beta-binomial posterior detection
                          probability >= POWER) and absent
  unknown                 unpowered; excluded from the denominator

Singletons: AFFILIATE first — a singleton whose one-way selection lands
exactly on a spoken-for member of an accepted, ancestry-compatible
meta-clade that has NO member in the singleton's own dataset attaches to
that meta-clade as an asymmetric affiliate (it plausibly IS the missing
member; this removes the duplicate-private residual). If the landing
clade already holds a distinct member in the singleton's dataset, the
landing is absorption and evaluation proceeds: PRIVATE only when at
least one other dataset was powered to detect it (and stability >=
STABILITY_FLOOR); otherwise UNKNOWN. Private subtrees keep their complete internal topology
(`subtree_parent`): absorbed never means discarded.

Refused merge edges are classified by ancestry analysis:
  resolution_mismatch  the refused endpoint relates to the group's member
                       as ancestor/descendant in its own tree
  genuine_conflict     disjoint branches
  ambiguity            interleaved or undeterminable
"""
from __future__ import annotations

import numpy as np

from metaarbor import leaves_under

from .eligibility import p_detect_posterior, prevalence_lower, prevalence_posterior
from .poset import compatible, pair_relation, relation

FROZEN = {"POWER": 0.95, "CRED_Q": 0.05, "MIN_SUPPORT": 0.5,
          "MIN_DATASETS": 2, "STABILITY_FLOOR": 0.7}


def _cells_under(datasets, trees, ds, node):
    labels = np.asarray(datasets[ds]["labels"])
    if node is None or node == "root":
        return int(len(labels))
    lv = set(leaves_under(trees[ds], node))
    return int(np.isin(labels, list(lv)).sum())


def _mean_boot_support(cand):
    vals = [v for e in cand["reciprocal_edges"]
            for v in (e["support_ij"], e["support_ji"])
            if np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


def classify_edge_conflicts(edge_conflicts, candidates, trees):
    """Merge collisions -> resolution_mismatch / genuine_conflict /
    ambiguity via ancestry analysis in the endpoint's own tree."""
    node_cand = {}
    for c in candidates:
        if c["kind"] == "multi":
            for ds, n in c["members"].items():
                node_cand[(ds, n)] = c
    out = []
    for ec in edge_conflicts:
        ki, ni, kj, nj = ec["edge"]
        verdicts = []
        for (ka, na, kb, nb) in ((ki, ni, kj, nj), (kj, nj, ki, ni)):
            g = node_cand.get((kb, nb))
            if g and ka in g["members"] and g["members"][ka] != na:
                rel = relation(trees[ka], na, g["members"][ka])
                verdicts.append({"ancestor": "resolution_mismatch",
                                 "descendant": "resolution_mismatch",
                                 "disjoint": "genuine_conflict",
                                 "interleaved": "ambiguity",
                                 "equal": "ambiguity"}[rel])
        if not verdicts:
            cls = "ambiguity"
        elif len(set(verdicts)) == 1:
            cls = verdicts[0]
        else:
            cls = "ambiguity"
        out.append(dict(ec, **{"class": cls}))
    return out


def _eligibility_row(cand, ds, datasets, trees, parent_members, selections,
                     frozen, terminal_in=None, has_free_below=None):
    """Raw eligibility calculation for one (dataset, candidate) pair."""
    members = cand["members"]
    parent_member = parent_members.get(ds)
    n_parent = _cells_under(datasets, trees, ds, parent_member)
    if ds in members:
        k = _cells_under(datasets, trees, ds, members[ds])
        return {"dataset": ds, "call": "supported", "k": k,
                "n_parent": n_parent, "power": None, "posterior": None}
    # (a) resolution limitation: ds cannot express anything below parent
    if parent_member is not None and terminal_in is not None and             (ds, parent_member) in terminal_in:
        return {"dataset": ds, "call": "unresolved_in_dataset", "k": 0,
                "n_parent": n_parent, "power": None, "posterior": None,
                "via": ("terminal_parent", parent_member)}
    # (b) free structure below the parent: absence is indistinguishable
    # from unresolved local structure while unclaimed nodes remain there;
    # one-way landings stay recorded as asymmetric evidence only
    if parent_member is not None and has_free_below is not None and \
            has_free_below.get((ds, parent_member)):
        return {"dataset": ds, "call": "unresolved_in_dataset", "k": 0,
                "n_parent": n_parent, "power": None, "posterior": None,
                "via": ("free_structure_below", parent_member)}
    # LOO prevalence from supporting datasets
    ks, ns = [], []
    for mds, mnode in members.items():
        pk = _cells_under(datasets, trees, mds, mnode)
        pn = _cells_under(datasets, trees, mds, parent_members.get(mds))
        ks.append(pk)
        ns.append(max(pn, pk))
    if len(ks) >= 2:
        a, b = prevalence_posterior(ks, ns)
        power = p_detect_posterior(a, b, n_parent)
        post = ("beta", float(a), float(b))
    elif len(ks) == 1:
        p_lo = prevalence_lower(ks[0], ns[0], q=frozen["CRED_Q"])
        power = 1.0 - (1.0 - p_lo) ** max(n_parent, 0)
        post = ("lower_bound", float(p_lo))
    else:
        power, post = 0.0, None
    call = ("private_or_absent" if power >= frozen["POWER"] else "unknown")
    return {"dataset": ds, "call": call, "k": 0, "n_parent": n_parent,
            "power": float(power), "posterior": post}


def greedy_backbone(cand_out, trees, datasets, selections=None,
                    stability=None, frozen=FROZEN):
    """Hierarchical greedy consensus. Returns dict with:
    nodes (accepted, MA-C#### in acceptance order, with status
    backbone|private, parent links, members, eligibility), rejected,
    unknown, conflicts (classified edge collisions + ancestry
    disagreements + compatibility rejections), eligibility_table (raw
    rows for EVERY donor-candidate pair), provenance rows."""
    candidates = cand_out["candidates"]
    keys = sorted(trees)
    stability = stability or {}

    from .candidates import canonical_nodes
    # terminal canonical nodes per dataset (no canonical children)
    terminal_in = set()
    for k in keys:
        nodes_k, _ = canonical_nodes(trees[k])
        from metaarbor.branch_fit import _collapse_chains
        rp, _ = _collapse_chains(trees[k])
        has_child = {p for p in rp.values()}
        for n_ in nodes_k:
            if n_ not in has_child:
                terminal_in.add((k, n_))
    node_owner = {(ds, n_): ci for ci, c in enumerate(candidates)
                  if c["kind"] == "multi"
                  for ds, n_ in c["members"].items()}
    # structure is "claimed" only by ACCEPTED multi-dataset claims — a
    # claim that is later rejected must not screen eligibility, so
    # free-structure-below is evaluated DYNAMICALLY at each candidate's
    # processing time against the accepted-so-far set (ancestors are
    # processed first, so their claims are in place when descendants and
    # siblings are adjudicated)
    claimed = set()
    kids_by = {}
    for k in keys:
        rp, _ = _collapse_chains(trees[k])
        kof = {}
        for c_, p_ in rp.items():
            kof.setdefault(p_, []).append(c_)
        kids_by[k] = kof

    class _FreeBelow:
        """dict-shaped view (.get) computing free-structure-below
        against the live accepted-claim set."""

        def get(self, key):
            k, p_ = key
            stack = list(kids_by.get(k, {}).get(p_, []))
            while stack:
                v = stack.pop()
                if (k, v) not in claimed:
                    return True
                stack.extend(kids_by[k].get(v, []))
            return False

    has_free_below = _FreeBelow()

    # candidate ancestry relation (co-present datasets); disagreements
    # recorded as conflicts and treated as unrelated for ordering
    n = len(candidates)
    anc = np.zeros((n, n), dtype=bool)
    conflicts = classify_edge_conflicts(
        cand_out.get("edge_conflicts", []), candidates, trees)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rel, disagreement = pair_relation(candidates[i]["members"],
                                              candidates[j]["members"],
                                              trees)
            if disagreement:
                if i < j:
                    conflicts.append({"type": "ancestry_disagreement",
                                      "candidates": (i, j),
                                      "detail": disagreement,
                                      "class": "genuine_conflict"})
                continue
            if rel == "ancestor":
                anc[i, j] = True

    # eligibility per candidate (parent context = nearest ACCEPTED
    # ancestor's members at processing time; root before any acceptance)
    accepted, rejected, unknown_out = [], [], []
    accepted_members = []           # list of member dicts, accept order
    node_list = []
    affiliates = []
    eligibility_table = []
    ids = {}
    processed = set()

    def parent_context(ci):
        """Deepest accepted ancestor's members (per dataset), else root.
        Also returns the MAXIMAL accepted ancestors: more than one means
        two incomparable parent candidates claim this child (ambiguous
        parentage — surfaced as a conflict at acceptance time)."""
        best, best_idx, maximal = None, None, []
        for k, aidx in enumerate(accepted):
            if anc[aidx, ci]:
                maximal = [m for m in maximal if not anc[m, aidx]]
                if not any(anc[aidx, m] for m in maximal):
                    maximal.append(aidx)
                if best is None or anc[best_idx, aidx]:
                    best, best_idx = accepted_members[k], aidx
        ctx = {ds: None for ds in keys}
        if best:
            ctx.update(best)
        return ctx, best_idx, maximal

    order_round = 0
    while len(processed) < n:
        ready = [i for i in range(n) if i not in processed and
                 all(j in processed for j in range(n) if anc[j, i])]
        if not ready:
            # ancestry constraints among the unprocessed candidates are
            # CYCLIC: contradictory cross-atlas ancestry evidence. Emit
            # an explicit conflict and reject every cycle member (the
            # claims die; their labels fall back via rejection routing)
            # — never silently linearize by index.
            rem = [i for i in range(n) if i not in processed]
            reach = {i: {j for j in rem if j != i and anc[i, j]}
                     for i in rem}
            changed = True
            while changed:
                changed = False
                for i in rem:
                    ext = set().union(*(reach[j] for j in reach[i]))                         if reach[i] else set()
                    if not ext <= reach[i]:
                        reach[i] |= ext
                        changed = True
            in_cycle = sorted({i for i in rem if i in reach[i]})
            comps, seen = [], set()
            for i in in_cycle:
                if i in seen:
                    continue
                comp = sorted({i} | {j for j in in_cycle
                                     if j in reach[i] and i in reach[j]})
                seen.update(comp)
                comps.append(comp)
            if not comps:                   # safety net; cannot happen
                ready = [min(rem)]
            else:
                for comp in comps:
                    conflicts.append({
                        "type": "ancestry_cycle",
                        "candidates": [candidates[i]["candidate_id"]
                                       for i in comp],
                        "class": "genuine_conflict"})
                    for i in comp:
                        processed.add(i)
                        rejected.append({"candidate": candidates[i],
                                         "reason": "ancestry_cycle",
                                         "support": (0, 0)})
                continue
        # group by parent context, rank within it
        def rank_key(ci):
            c = candidates[ci]
            rows = []
            ctx, _, _ = parent_context(ci)
            for ds in keys:
                rows.append(_eligibility_row(c, ds, datasets, trees, ctx,
                                             selections, frozen,
                                             terminal_in=terminal_in,
                                             has_free_below=has_free_below))
            supp = sum(r["call"] == "supported" for r in rows)
            elig = sum(r["call"] in ("supported", "private_or_absent")
                       for r in rows)
            ratio = supp / elig if elig else 0.0
            return (-ratio, -supp, -c["seed_invariance"],
                    -_mean_boot_support(c),
                    tuple(sorted(c["members"].items()))), rows
        keyed = []
        for ci in ready:
            k_, rows = rank_key(ci)
            keyed.append((k_, ci, rows))
        keyed.sort(key=lambda t: t[0])
        for k_, ci, _rank_rows in keyed:
            processed.add(ci)
            c = candidates[ci]
            # ranking used cohort-start state; ADJUDICATION re-evaluates
            # eligibility fresh so claims accepted earlier in this same
            # cohort are visible (no stale shared state within a round)
            _k2, rows = rank_key(ci)
            for r in rows:
                eligibility_table.append(
                    dict(r, candidate=c["candidate_id"]))
            supp = sum(r["call"] == "supported" for r in rows)
            elig = sum(r["call"] in ("supported", "private_or_absent")
                       for r in rows)
            ratio = supp / elig if elig else 0.0
            stab = min((stability.get((ds, node), 1.0)
                        for ds, node in c["members"].items()), default=1.0)
            if supp >= frozen["MIN_DATASETS"]:
                status = ("backbone" if ratio >= frozen["MIN_SUPPORT"]
                          else None)
                reason = None if status else "insufficient_support"
            elif supp == 1:
                # affiliate check: does the singleton's one-way walk land
                # on an accepted meta-clade missing a member in the
                # singleton's own dataset?
                (sk, snode), = c["members"].items()
                affiliate_target = None
                sub_nodes = c.get("provenance", {}).get(
                    "subtree_nodes") or []
                if selections is not None and len(sub_nodes) <= 1:
                    # a consolidated subtree is never affiliated: an
                    # alias cannot carry topology, so subtree claims go
                    # through the private/unknown routes (which expand)
                    landings = {}
                    for d in keys:
                        if d == sk:
                            continue
                        sel = selections.get((sk, d), {}).get(snode)
                        v = sel.get("selected") if sel else None
                        oi = node_owner.get((d, v)) if v else None
                        if oi is None or oi not in ids or \
                                sk in candidates[oi]["members"]:
                            continue
                        _rel, dis = pair_relation(
                            c["members"], candidates[oi]["members"],
                            trees)
                        if not dis:
                            landings[d] = oi
                    targets = sorted(set(landings.values()))
                    if len(targets) == 1:
                        affiliate_target = ids[targets[0]]
                    elif len(targets) > 1:
                        comparable = all(
                            anc[a_, b_] or anc[b_, a_]
                            for x_, a_ in enumerate(targets)
                            for b_ in targets[x_ + 1:])
                        if comparable:
                            # nested resolution difference: attach to
                            # the COARSEST landing (the common ancestor
                            # claim) — the conservative statement
                            coarse = [t for t in targets
                                      if all(anc[t, o] for o in targets
                                             if o != t)]
                            if coarse:
                                affiliate_target = ids[coarse[0]]
                        if affiliate_target is None:
                            conflicts.append({
                                "type": "affiliate_incompatible_landings",
                                "candidate": c["candidate_id"],
                                "targets": [candidates[t]["candidate_id"]
                                            for t in targets],
                                "class": "ambiguity"})
                if affiliate_target is not None:
                    affiliates.append({
                        "dataset": sk, "node": snode,
                        "attached_to": affiliate_target,
                        "candidate_id": c["candidate_id"]})
                    continue
                powered_others = sum(r["call"] == "private_or_absent"
                                     for r in rows)
                if powered_others >= 1 and stab >= frozen[
                        "STABILITY_FLOOR"]:
                    status, reason = "private", None
                else:
                    status, reason = None, "unknown_no_power"
            else:
                status, reason = None, "no_support"
            if status is None:
                (unknown_out if reason == "unknown_no_power"
                 else rejected).append(
                    {"candidate": c, "reason": reason,
                     "support": (supp, elig)})
                continue
            ok, comp_conflicts = compatible(
                accepted_members, c["members"], trees)
            if not ok:
                conflicts.append({"type": "ancestry_incompatible",
                                  "candidate": c["candidate_id"],
                                  "detail": comp_conflicts,
                                  "class": "genuine_conflict"})
                rejected.append({"candidate": c,
                                 "reason": "ancestry_incompatible",
                                 "support": (supp, elig)})
                continue
            _ctx, parent_idx, maximal = parent_context(ci)
            if len(maximal) > 1:
                conflicts.append({
                    "type": "ambiguous_parent",
                    "candidate": c["candidate_id"],
                    "parents": [candidates[m]["candidate_id"]
                                for m in sorted(maximal)],
                    "class": "ambiguity"})
                # the diagnostic must govern the placement: an ambiguous
                # candidate is NOT forced under an arbitrary parent — it
                # attaches at the deepest accepted ancestor COMMON to
                # every maximal parent (a polytomy), or at root
                common = [a for a in accepted
                          if anc[a, ci] and
                          all(anc[a, m] for m in maximal)]
                parent_idx = None
                for a in common:
                    if parent_idx is None or anc[parent_idx, a]:
                        parent_idx = a
            ma_id = f"MA-C{len(accepted)+1:04d}"
            ids[ci] = ma_id
            node = {"id": ma_id, "status": status,
                    "members": dict(c["members"]),
                    "parent": ids.get(parent_idx),
                    "support": (supp, elig),
                    "seed_invariance": c["seed_invariance"],
                    "mean_boot_support": _mean_boot_support(c),
                    "stability": stab,
                    "candidate_id": c["candidate_id"]}
            if c["kind"] == "private_subtree":
                # absorbed never means discarded: full internal topology
                from metaarbor.branch_fit import _collapse_chains
                ds = next(iter(c["members"]))
                red_parent, _ = _collapse_chains(trees[ds])
                sub = set(c["provenance"].get("subtree_nodes", []))
                node["subtree_parent"] = {
                    x: (red_parent.get(x) if red_parent.get(x) in sub
                        else None) for x in sorted(sub)}
            accepted.append(ci)
            accepted_members.append(c["members"])
            if c["kind"] == "multi":
                claimed.update((ds, n_) for ds, n_ in
                               c["members"].items())
            node_list.append(node)
        order_round += 1
        if order_round > n + 2:
            break
    for aff in affiliates:      # attach aliases onto their target nodes
        for nd in node_list:
            if nd["id"] == aff["attached_to"]:
                nd.setdefault("affiliates", []).append(
                    {"dataset": aff["dataset"], "node": aff["node"]})
    return {"nodes": node_list, "rejected": rejected,
            "unknown": unknown_out, "conflicts": conflicts,
            "affiliates": affiliates,
            "eligibility_table": eligibility_table,
            "frozen": dict(frozen)}


def provenance_table(backbone_out):
    """Every source label -> MA id -> display name (modal member name) —
    original names are synonyms, never destroyed."""
    rows = []
    for nd in backbone_out["nodes"]:
        names = [n.split("|", 1)[-1] for n in nd["members"].values()]
        display = sorted(names)[0] if names else nd["id"]
        for ds, node in sorted(nd["members"].items()):
            rows.append({"dataset": ds, "source_node": node,
                         "ma_id": nd["id"], "display_name": display,
                         "status": nd["status"]})
    return rows
