# Common node-evidence table: one row per (query, visited split, child) —
# directional AUROC, sibling contrast with bootstrap lower bound, vote
# fraction, override/margin verdicts, and the query's selected relation.
# (MetaArbor-Transport mass is python-only; join the python table on
# query/child when a coupling is available.)
#
# Evidence rows are re-derived with the same per-query seeds as
# ma_baseline_map, so recorded decisions are exactly the map's decisions.

#' @param cache,test_labels,tree,S_dir as for ma_baseline_map
#' @param cache_rev,labels_rev optional reverse-fold cache (leaves = query
#'   populations) for the second directional AUROC
ma_node_evidence <- function(cache, test_labels, tree, S_dir, base_seed = 7,
                             cache_rev = NULL, labels_rev = NULL, ...) {
  queries <- sort(unique(as.character(test_labels)))
  map <- ma_baseline_map(cache, test_labels, tree, S_dir,
                         base_seed = base_seed, ...)
  out <- list()
  for (i in seq_along(queries)) {
    q <- queries[i]
    sel <- ma_select_node(cache, test_labels, q, tree,
                          seed = base_seed + i - 1, trace = TRUE, ...)
    tr <- sel$trace
    if (is.null(tr)) next
    tr$n_leaves <- vapply(tr$child, function(ch)
      length(ma_leaves_under(tree, ch)), numeric(1))
    tr$selected <- map$selected[map$query == q]
    tr$relation <- map$relation[map$query == q]
    tr$auroc_rev <- NA_real_
    if (!is.null(cache_rev) && q %in% cache_rev$leaves) {
      for (r in seq_len(nrow(tr))) {
        pos <- labels_rev %in% ma_leaves_under(tree, tr$child[r])
        tr$auroc_rev[r] <- ma_auroc(cache_rev$V[, q], pos)
      }
    }
    out[[i]] <- tr
  }
  res <- do.call(rbind, out)
  rownames(res) <- NULL
  res
}
