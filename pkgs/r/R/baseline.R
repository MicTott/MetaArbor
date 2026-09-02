# The frozen MetaArbor-Walk estimator (repo NOTES.md items 4, 7-9;
# frozen 2026-09-02). Votes navigate, AUROC contrasts decide.
#
# *** FROZEN defaults: alpha = 0.05, n_boot = 200, min_auroc = 0.6,
# margin = 0.01, vote_override = 0.9, min_compact = 0.7. Changing them on
# validation data un-freezes the estimator. ***
#
# Package difference from the research-repo copy: bootstraps draw from the
# portable MINSTD stream (R/rng.R) with a per-query seed
# (base_seed + query rank in the sorted query list), in a documented order
# (per iteration: n_pos positive draws, then n_neg negative draws), matching
# the python package bit-for-bit. Children at a split are sorted so tie
# handling is language-identical.

#' Vote score vector for a node (sum of cached leaf columns).
ma_node_scores <- function(cache, node_leaves) {
  if (length(node_leaves) == 1L) cache$V[, node_leaves]
  else rowSums(cache$V[, node_leaves, drop = FALSE])
}

#' Mean-rank score of a node (per training cell).
ma_node_mean_scores <- function(cache, node_leaves) {
  ma_node_scores(cache, node_leaves) / sum(cache$leaf_sizes[node_leaves])
}

#' Paired bootstrap of Delta-AUROC using the portable MINSTD stream.
ma_boot_delta <- function(scores1, scores2, positive, rng, n_boot = 200L) {
  ip <- which(positive); ineg <- which(!positive)
  np <- length(ip); nn <- length(ineg)
  posmask <- c(rep(TRUE, np), rep(FALSE, nn))
  delta <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    sp <- ip[ma_minstd_indices(rng, np, np) + 1]
    sn <- ineg[ma_minstd_indices(rng, nn, nn) + 1]
    idx <- c(sp, sn)
    delta[b] <- ma_auroc(scores1[idx], posmask) - ma_auroc(scores2[idx], posmask)
  }
  delta
}

#' Frozen hierarchical selection for one query population.
#' @param seed MINSTD seed for this query's bootstrap stream
ma_select_node <- function(cache, test_labels, query, tree, seed,
                           alpha = 0.05, n_boot = 200L, min_auroc = 0.6,
                           margin = 0.01, vote_override = 0.9, trace = FALSE) {
  positive <- test_labels == query
  rng <- ma_minstd_new(seed)
  node_scores <- function(id) ma_node_scores(cache, ma_leaves_under(tree, id))
  node_auc <- function(id) ma_auroc(node_scores(id), positive)
  ms <- sweep(cache$V[positive, , drop = FALSE], 2, cache$leaf_sizes, "/")
  top_leaf <- cache$leaves[max.col(ms, ties.method = "first")]
  child_votes <- function(kids) {
    vapply(kids, function(k)
      sum(top_leaf %in% ma_leaves_under(tree, k)), numeric(1)) / length(top_leaf)
  }
  path <- NULL
  trace_rows <- if (trace) list() else NULL
  current <- "root"
  repeat {
    kids <- sort(ma_children(tree, current))
    if (!length(kids)) break
    if (length(kids) == 1L) { current <- kids; next }
    votes <- child_votes(kids)
    ord <- order(-votes)                    # stable; kids sorted => ties match py
    best <- kids[ord[1]]; second <- kids[ord[2]]
    sib_lo <- par_lo <- NA_real_
    override <- votes[ord[1]] >= vote_override
    if (override) {
      stop_here <- FALSE
    } else {
      d_sib <- ma_boot_delta(node_scores(best), node_scores(second),
                             positive, rng, n_boot)
      sib_lo <- unname(quantile(d_sib, alpha))
      concentrated <- sib_lo > margin
      if (current != "root" && concentrated) {
        d_par <- ma_boot_delta(node_scores(current), node_scores(best),
                               positive, rng, n_boot)
        par_lo <- unname(quantile(d_par, alpha))
      }
      stop_here <- !concentrated || (!is.na(par_lo) && par_lo > 0)
    }
    path <- rbind(path, data.frame(
      id = if (stop_here) current else best,
      vote = votes[ord[1]], sib_lo = sib_lo, par_lo = par_lo,
      override = override, stopped = stop_here))
    if (trace) {
      kid_auc <- vapply(kids, node_auc, numeric(1))
      trace_rows[[length(trace_rows) + 1L]] <- data.frame(
        query = query, split_at = current, child = kids,
        vote = unname(votes), child_auroc = unname(kid_auc),
        is_best = kids == best, sib_lo = sib_lo, par_lo = par_lo,
        override = override,
        decision = if (stop_here) "stop" else "descend", row.names = NULL)
    }
    if (stop_here) break
    current <- best
  }
  sel_auc <- if (current == "root") NA_real_ else node_auc(current)
  matched <- is.finite(sel_auc) && sel_auc >= min_auroc
  list(query = query, selected = if (matched) current else NA_character_,
       auroc = sel_auc, matched = matched, at_root = current == "root",
       final = current, path = path,
       trace = if (trace) do.call(rbind, trace_rows) else NULL)
}

#' Context compactness: fraction of positive cells whose argmax leaf lies in
#' the selected node's parent-context subtree.
ma_compactness <- function(cache, positive, tree, selected) {
  parent <- tree$parent[match(selected, tree$id)]
  ctx <- if (is.na(parent) || parent == "root") selected else parent
  ms <- sweep(cache$V[positive, , drop = FALSE], 2, cache$leaf_sizes, "/")
  top_leaf <- cache$leaves[max.col(ms, ties.method = "first")]
  mean(top_leaf %in% ma_leaves_under(tree, ctx))
}

#' Map every test population onto `tree` with the frozen walk.
#' Per-query MINSTD seed = base_seed + rank in the sorted query list.
ma_baseline_map <- function(cache, test_labels, tree, S_dir, base_seed = 7,
                            alpha = 0.05, n_boot = 200L, min_auroc = 0.6,
                            min_compact = 0.7, margin = 0.01,
                            vote_override = 0.9) {
  queries <- sort(unique(as.character(test_labels)))
  res <- lapply(seq_along(queries), function(i) {
    q <- queries[i]
    sel <- ma_select_node(cache, test_labels, q, tree, seed = base_seed + i - 1,
                          alpha, n_boot, min_auroc, margin, vote_override)
    comp <- if (sel$matched)
      ma_compactness(cache, test_labels == q, tree, sel$selected) else NA_real_
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
