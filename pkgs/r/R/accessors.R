# Stable accessors for MetaArbor's underlying matrices — read-only views
# over quantities the estimators already compute.
#
# Raw one-vs-all node AUROC (ma_node_auroc_matrix) is an INTERPRETABILITY
# measure, not a replacement for the sibling-relative Walk decision: large
# heterogeneous unions dilute toward 0.5 (root neutrality at scale) and
# easy backgrounds saturate parent and child alike near 1 — the effects the
# Walk's vote-guided navigation and sibling contrast exist to sidestep.

#' Query x target-leaf argmax-vote fractions (rows sum to 1).
ma_vote_matrix <- function(cache, test_labels) {
  queries <- sort(unique(as.character(test_labels)))
  ms <- sweep(cache$V, 2, cache$leaf_sizes, "/")
  top <- cache$leaves[max.col(ms, ties.method = "first")]
  M <- t(vapply(queries, function(q)
    as.numeric(table(factor(top[test_labels == q], levels = cache$leaves))) /
      sum(test_labels == q), numeric(length(cache$leaves))))
  dimnames(M) <- list(queries, cache$leaves)
  M
}

#' Query x target-node one-vs-all AUROCs. `nodes` defaults to the cache
#' leaves; pass internal node ids for aggregated views. See file header for
#' size/saturation caveats.
ma_node_auroc_matrix <- function(cache, test_labels, tree, nodes = NULL) {
  queries <- sort(unique(as.character(test_labels)))
  if (is.null(nodes)) nodes <- cache$leaves
  M <- matrix(NA_real_, length(queries), length(nodes),
              dimnames = list(queries, nodes))
  for (n in nodes) {
    lv <- if (n %in% cache$leaves) n
          else intersect(ma_leaves_under(tree, n), cache$leaves)
    if (!length(lv)) next
    s <- ma_node_scores(cache, lv)
    for (q in queries) M[q, n] <- ma_auroc(s, test_labels == q)
  }
  M
}

#' Family/node-aggregated transport mass; rows normalized to 1 by default.
ma_family_mass <- function(pi, family_of_leaf, normalize = TRUE) {
  fams <- sort(unique(unname(family_of_leaf[colnames(pi)])))
  P <- pi
  if (normalize) {
    tot <- rowSums(P)
    tot[tot == 0] <- 1
    P <- P / tot
  }
  M <- vapply(fams, function(f)
    rowSums(P[, unname(family_of_leaf[colnames(pi)]) == f, drop = FALSE]),
    numeric(nrow(pi)))
  dimnames(M) <- list(rownames(pi), fams)
  M
}

#' Complete Walk decision traces for every query (same per-query seeds as
#' ma_baseline_map; decisions identical by construction).
ma_walk_traces <- function(cache, test_labels, tree, base_seed = 7, ...) {
  queries <- sort(unique(as.character(test_labels)))
  out <- lapply(seq_along(queries), function(i)
    ma_select_node(cache, test_labels, queries[i], tree,
                   seed = base_seed + i - 1, trace = TRUE, ...)$trace)
  res <- do.call(rbind, out)
  rownames(res) <- NULL
  res
}
