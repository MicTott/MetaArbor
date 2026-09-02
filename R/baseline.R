# Baseline estimator: hierarchical MetaNeighbor by direct selection
# (DESIGN.md §3b). Top-down descent per query population; hypotheses gated by
# the descent (children are tested only where the walk reaches them). Formal
# hierarchical-FDR calibration is a prototype TODO (DESIGN.md §3b item 4).
#
# *** FROZEN 2026-09-02 for validation ***
# The decision rules and their defaults (vote-guided navigation, sibling
# contrast margin = 0.01, vote_override = 0.9, alpha = 0.05, n_boot = 200,
# min_auroc = 0.6, min_compact = 0.7) were developed against the
# 10Xv2-vs-10Xv3 PL-ILA-ORB run and are frozen through the batch-condition
# validation (random / donor-held-out / cross-platform). Changing them in
# response to validation results turns the validation back into development.
#
# Two structural facts discovered and relied on during prototyping:
#
# 1. With mean rank-standardized votes, the union of ALL training leaves gives
#    every test cell the same score, so the root's AUROC is exactly 0.5 and
#    over-merged nodes decay toward chance. The classic "root wins" failure
#    of score-maximizing tree search cannot occur.
#
# 2. One-vs-all AUROCs saturate near 1 for parent and child alike when the
#    background is easy, so a parent-vs-child test alone cannot resolve the
#    level. The discriminating question is the SIBLING CONTRAST: a leaf-level
#    query separates the best child from the second-best (Delta-AUROC > 0),
#    while a family-level query relates to the children exchangeably
#    (Delta approx 0) regardless of saturation. Per-cell argmax preferences
#    (an earlier design) are underpowered for this — sibling similarity makes
#    individual cells' votes noisy even when the population-level contrast is
#    clear — so the descent tests AUROC contrasts and reports per-cell shares
#    only as diagnostics.

#' Vote score vector for a node (sum of cached leaf columns; Appendix A).
tn_node_scores <- function(cache, node_leaves) {
  if (length(node_leaves) == 1L) cache$V[, node_leaves]
  else rowSums(cache$V[, node_leaves, drop = FALSE])
}

#' Mean-rank score of a node (per training cell), comparable across sibling
#' nodes of different sizes. Used for descriptive preference shares.
tn_node_mean_scores <- function(cache, node_leaves) {
  tn_node_scores(cache, node_leaves) / sum(cache$leaf_sizes[node_leaves])
}

#' Paired bootstrap of Delta-AUROC = auroc(scores1) - auroc(scores2) for one
#' query, resampling test cells (stratified positives/negatives), both score
#' vectors evaluated on the same resample.
tn_boot_delta <- function(scores1, scores2, positive, n_boot = 200L) {
  ip <- which(positive); ineg <- which(!positive)
  delta <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    sp <- sample(ip, length(ip), replace = TRUE)
    sn <- sample(ineg, length(ineg), replace = TRUE)
    idx <- c(sp, sn)
    pos <- c(rep(TRUE, length(sp)), rep(FALSE, length(sn)))
    delta[b] <- tn_auroc(scores1[idx], pos) - tn_auroc(scores2[idx], pos)
  }
  delta
}

#' Descriptive per-cell preference shares at a split: which child does each
#' positive cell vote for (argmax of mean-rank score)? Reported, not tested.
tn_child_shares <- function(cache, tree, kids, positive) {
  ms <- sapply(kids, function(k) tn_node_mean_scores(cache, tn_leaves_under(tree, k)))
  pref <- kids[max.col(ms[positive, , drop = FALSE], ties.method = "first")]
  prop.table(table(factor(pref, levels = kids)))
}

#' Descend `tree` (over the training atlas's leaves) for one query population
#' from the test atlas. At each split, order the children by one-vs-all AUROC
#' and descend into the best child only when (a) the sibling contrast is
#' significant — the best child's AUROC exceeds the second-best's, paired
#' bootstrap lower bound > 0 — and (b) the parent's AUROC is not
#' significantly better than the best child's. Otherwise stop: the current
#' node is the finest reproducible level for this query. Single-child chains
#' are followed without a test.
#'
#' @param alpha one-sided level for both tests at each visited split
#' @param min_auroc below this, the best achievable node is called unmatched
#' @param margin practical-significance margin on the sibling contrast, in
#'   AUROC units: descend only when the best child beats the second-best by
#'   more than this. With tens of thousands of test cells, the bootstrap
#'   makes differences of 0.001 statistically significant, and a walk that
#'   descends on statistical significance alone overshoots to single leaves;
#'   equivalence within `margin` means "this query does not meaningfully
#'   distinguish the siblings" and the walk stops at the finest reproducible
#'   level. Observed directly on the Allen benchmark (NOTES.md).
#' @param vote_override descend without the AUROC test when this fraction of
#'   the query's cells votes for one child's subtree — clear votes beat
#'   union-AUROC saturation at coarse nodes (Allen: distinct vascular
#'   subclasses both saturate one-vs-all, but the cells' votes are ~98%
#'   unanimous)
#' @param trace when TRUE, attach a per-visited-split diagnostics table
#'   (child votes, child AUROCs, tests, decision). Recording only: it adds no
#'   RNG draws and alters no decision — verified by identity against the
#'   saved benchmark maps (analysis/07_fig_walks.R).
tn_select_node <- function(cache, test_labels, query, tree,
                           alpha = 0.05, n_boot = 200L, min_auroc = 0.6,
                           margin = 0.01, vote_override = 0.9, trace = FALSE) {
  positive <- test_labels == query
  node_scores <- function(id) tn_node_scores(cache, tn_leaves_under(tree, id))
  node_auc <- function(id) tn_auroc(node_scores(id), positive)
  # per-cell leaf votes, computed once: each positive cell votes for its
  # argmax mean-rank training leaf; a child collects the votes whose leaf
  # sits in its subtree. Size-robust, unlike one-vs-all AUROC of a union,
  # which dilutes toward chance for large heterogeneous children
  # (root-neutrality at scale) and misranks children of unequal size.
  ms <- sweep(cache$V[positive, , drop = FALSE], 2, cache$leaf_sizes, "/")
  top_leaf <- cache$leaves[max.col(ms, ties.method = "first")]
  child_votes <- function(kids) {
    vapply(kids, function(k)
      sum(top_leaf %in% tn_leaves_under(tree, k)), numeric(1)) / length(top_leaf)
  }
  path <- data.frame(id = character(0), auroc = numeric(0), vote = numeric(0),
                     sib_lo = numeric(0), par_lo = numeric(0),
                     stopped = logical(0))
  current <- "root"
  split_shares <- NULL
  trace_rows <- if (trace) list() else NULL
  repeat {
    kids <- tn_children(tree, current)
    if (!length(kids)) break                       # reached a leaf
    if (length(kids) == 1L) { current <- kids; next }  # chain: follow
    votes <- child_votes(kids)
    ord <- order(votes, decreasing = TRUE)
    best <- kids[ord[1]]; second <- kids[ord[2]]
    sib_lo <- par_lo <- NA_real_
    override_used <- votes[ord[1]] >= vote_override
    if (override_used) {
      stop_here <- FALSE
    } else {
      d_sib <- tn_boot_delta(node_scores(best), node_scores(second),
                             positive, n_boot)
      sib_lo <- unname(quantile(d_sib, alpha))
      concentrated <- sib_lo > margin
      if (current != "root" && concentrated) {
        # root parent is uninformative by construction (AUROC exactly 0.5)
        d_par <- tn_boot_delta(node_scores(current), node_scores(best),
                               positive, n_boot)
        par_lo <- unname(quantile(d_par, alpha))
      }
      stop_here <- !concentrated || (!is.na(par_lo) && par_lo > 0)
    }
    if (trace) {
      kid_auc <- vapply(kids, node_auc, numeric(1))  # no RNG: recording only
      trace_rows[[length(trace_rows) + 1L]] <- data.frame(
        query = query, split_at = current, child = kids,
        vote = unname(votes), child_auroc = unname(kid_auc),
        is_best = kids == best, is_second = kids == second,
        auc_gap = unname(kid_auc[match(best, kids)]) -
                  unname(kid_auc[match(second, kids)]),
        sib_lo = sib_lo, par_lo = par_lo,
        margin_pass = !override_used && !is.na(sib_lo) && sib_lo > margin,
        override = override_used,
        decision = if (stop_here) "stop" else "descend")
    }
    path <- rbind(path, data.frame(
      id = if (stop_here) current else best,
      auroc = if (stop_here) { if (current == "root") NA_real_ else node_auc(current) }
              else node_auc(best),
      vote = votes[ord[1]], sib_lo = sib_lo, par_lo = par_lo,
      stopped = stop_here))
    if (stop_here) {
      split_shares <- setNames(votes, kids)
      break
    }
    current <- best
  }
  sel_auc <- if (current == "root") NA_real_ else node_auc(current)
  matched <- is.finite(sel_auc) && sel_auc >= min_auroc
  list(query = query, selected = if (matched) current else NA_character_,
       auroc = sel_auc, matched = matched, at_root = current == "root",
       path = path, split_shares = split_shares,
       trace = if (trace) do.call(rbind, trace_rows) else NULL,
       final = current)
}

#' Context compactness (DESIGN.md §2.2 "discordant"), cell-vote form: the
#' fraction of the query's positive cells whose argmax training LEAF
#' (mean-rank score) lies inside the selected node's PARENT subtree.
#' Scatter among the selected node's siblings is expected — that is what
#' family structure means; scatter across the parent's boundary is
#' discordance. When the parent is the root (top-level selection), the
#' selected subtree itself is the context.
#'
#' Per-cell votes, not affinity mass: an earlier affinity form
#' (sum of max(0, S - 0.5) shares) collapses when the background holds ~100
#' populations — tiny above-chance slivers across many off-target leaves
#' swamp the on-target affinity. One vote per cell is scale-invariant in the
#' number of background populations (NOTES.md).
tn_compactness <- function(cache, positive, tree, selected) {
  parent <- tree$parent[match(selected, tree$id)]
  ctx <- if (is.na(parent) || parent == "root") selected else parent
  ms <- sweep(cache$V[positive, , drop = FALSE], 2, cache$leaf_sizes, "/")
  top_leaf <- cache$leaves[max.col(ms, ties.method = "first")]
  mean(top_leaf %in% tn_leaves_under(tree, ctx))
}

#' Run the baseline in one direction: every test-atlas population against the
#' training atlas's tree. `S_dir` is the symmetrized similarity with rows =
#' test populations, cols = training leaves (costs$S oriented accordingly).
tn_baseline_map <- function(cache, test_labels, tree, S_dir,
                            alpha = 0.05, n_boot = 200L, min_auroc = 0.6,
                            min_compact = 0.7, margin = 0.01,
                            vote_override = 0.9) {
  queries <- sort(unique(as.character(test_labels)))
  res <- lapply(queries, function(q) {
    sel <- tn_select_node(cache, test_labels, q, tree, alpha, n_boot,
                          min_auroc, margin, vote_override)
    comp <- if (sel$matched)
      tn_compactness(cache, test_labels == q, tree, sel$selected) else NA_real_
    has_signal <- max(S_dir[q, ]) >= min_auroc
    is_leaf <- sel$matched && isTRUE(tree$is_leaf[match(sel$selected, tree$id)])
    relation <- if (sel$at_root) { if (has_signal) "discordant" else "unmatched" }
                else if (!sel$matched) "unmatched"
                else if (!is.na(comp) && comp < min_compact) "discordant"
                else if (is_leaf) "leaf" else "family"
    data.frame(query = q, selected = sel$selected, auroc = sel$auroc,
               compactness = comp, relation = relation)
  })
  out <- do.call(rbind, res)
  rownames(out) <- NULL
  out
}
