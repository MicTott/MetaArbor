# MetaArbor measurement kernel (DESIGN.md §3a).
#
# Native reimplementation of the MetaNeighbor voting kernel (the Bioconductor
# package is deprecated as of Bioc 3.23), exposing the per-cell vote cache the
# packaged functions do not. Scores are mean rank-standardized network weights;
# vote scores are additive over disjoint training unions, AUROCs are not
# (Appendix A of DESIGN.md) — every union AUROC is recomputed from summed scores.

#' Normalize a counts matrix to log1p-CPM. Accepts an already-log matrix
#' untouched when `assume_log = TRUE`. When the matrix carries only a gene
#' subset, pass the full-gene per-cell totals as `lib` so CPM is not
#' distorted by the subsetting.
ma_lognorm <- function(counts, assume_log = FALSE, lib = NULL) {
  if (assume_log) return(counts)
  if (is.null(lib)) lib <- colSums(counts)
  lib[lib == 0] <- 1
  log1p(t(t(counts) / lib) * 1e6)
}

#' Joint highly variable genes: intersection of each dataset's top-variance
#' genes on log-normalized expression (variableGenes-style, simplified).
ma_variable_genes <- function(expr_a, expr_b, n_top = 1000) {
  shared <- intersect(rownames(expr_a), rownames(expr_b))
  stopifnot(length(shared) > 10)
  va <- matrixStats::rowVars(expr_a[shared, , drop = FALSE])
  vb <- matrixStats::rowVars(expr_b[shared, , drop = FALSE])
  top_a <- shared[order(va, decreasing = TRUE)[seq_len(min(n_top, length(shared)))]]
  top_b <- shared[order(vb, decreasing = TRUE)[seq_len(min(n_top, length(shared)))]]
  hvg <- intersect(top_a, top_b)
  if (length(hvg) < 50)
    warning("only ", length(hvg), " joint HVGs; costs may be unstable")
  hvg
}

#' Rank each cell's expression profile within itself (Spearman preparation),
#' then center and L2-normalize so that crossprod() gives the cell-cell
#' Spearman correlation.
ma_rank_normalize <- function(expr) {
  r <- apply(expr, 2, rank, ties.method = "average")
  r <- sweep(r, 2, colMeans(r), "-")
  n2 <- sqrt(colSums(r^2))
  n2[n2 == 0] <- 1
  sweep(r, 2, n2, "/")
}

#' The vote cache: for every test cell, the summed rank-standardized network
#' weight to each *training leaf* (DESIGN.md §3a, Appendix A).
#'
#' For each test cell c, cross-dataset Spearman correlations to all training
#' cells are ranked within c (rank / n_train, i.e. standardized to (0, 1]),
#' and summed per training leaf. Column sums over any set of leaves give the
#' vote score of that union; only the ordering across test cells matters for
#' AUROC, so no re-normalization is needed.
#'
#' @param test_norm,train_norm outputs of ma_rank_normalize on the joint HVGs
#' @param train_labels factor of training leaf labels (length = ncol(train_norm))
#' @param chunk test cells per block, to bound the correlation matrix in memory
#' @return list(V = n_test x n_leaf vote matrix, leaves = leaf names)
ma_vote_cache <- function(test_norm, train_norm, train_labels, chunk = 2000L) {
  train_labels <- droplevels(as.factor(train_labels))
  leaves <- levels(train_labels)
  n_test <- ncol(test_norm)
  n_train <- ncol(train_norm)
  # leaf indicator for fast per-leaf summation of ranks
  ind <- Matrix::sparseMatrix(
    i = seq_len(n_train), j = as.integer(train_labels), x = 1,
    dims = c(n_train, length(leaves))
  )
  V <- matrix(0, n_test, length(leaves), dimnames = list(colnames(test_norm), leaves))
  starts <- seq(1L, n_test, by = chunk)
  for (s in starts) {
    idx <- s:min(s + chunk - 1L, n_test)
    co <- crossprod(test_norm[, idx, drop = FALSE], train_norm)  # |idx| x n_train
    w <- t(apply(co, 1, rank, ties.method = "average")) / n_train
    V[idx, ] <- as.matrix(w %*% ind)
  }
  list(V = V, leaves = leaves,
       leaf_sizes = setNames(as.integer(table(train_labels)[leaves]), leaves))
}

#' Aggregate a vote cache to a coarser training labeling (Appendix A used
#' as an algorithm): the coarse cache is exact column sums of the fine one,
#' with no recomputation of the network. `mapping` is a named character
#' vector fine-leaf -> coarse-label covering all cache leaves.
ma_aggregate_cache <- function(cache, mapping) {
  stopifnot(all(cache$leaves %in% names(mapping)))
  coarse <- sort(unique(unname(mapping[cache$leaves])))
  V <- sapply(coarse, function(cl) {
    cols <- cache$leaves[mapping[cache$leaves] == cl]
    if (length(cols) == 1L) cache$V[, cols]
    else rowSums(cache$V[, cols, drop = FALSE])
  })
  sizes <- vapply(coarse, function(cl) {
    sum(cache$leaf_sizes[cache$leaves[mapping[cache$leaves] == cl]])
  }, numeric(1))
  list(V = V, leaves = coarse, leaf_sizes = sizes)
}

#' Vectorized Mann-Whitney AUROC of `scores` for a logical positive mask.
ma_auroc <- function(scores, positive) {
  n_pos <- sum(positive)
  n_neg <- sum(!positive)
  if (n_pos == 0L || n_neg == 0L) return(NA_real_)
  r <- rank(scores, ties.method = "average")
  (sum(r[positive]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

#' AUROC of a query test population against a training node (set of leaves),
#' from the cache. `negatives` restricts the background (one_vs_best-style);
#' default is one-vs-all.
ma_node_auroc <- function(cache, test_labels, query, node_leaves, negatives = NULL) {
  scores <- if (length(node_leaves) == 1L) cache$V[, node_leaves]
            else rowSums(cache$V[, node_leaves, drop = FALSE])
  pos <- test_labels == query
  keep <- if (is.null(negatives)) rep(TRUE, length(pos)) else (pos | negatives)
  ma_auroc(scores[keep], pos[keep])
}

#' Full symmetrized leaf-cost matrix (DESIGN.md §3a):
#'   M[i, j] = 1 - (AUC_{A->B}(i,j) + AUC_{B->A}(j,i)) / 2
#' computed from the two directional caches. Also returns both directional
#' AUROC matrices (documented as asymmetric in the MetaNeighbor vignette).
ma_leaf_costs <- function(cache_a, labels_a, cache_b, labels_b) {
  # cache_a: A cells scored against B leaves (train = B); AUC_{B->A}
  # cache_b: B cells scored against A leaves (train = A); AUC_{A->B}
  la <- sort(unique(as.character(labels_a)))
  lb <- cache_a$leaves
  stopifnot(identical(sort(cache_b$leaves), la))
  auc_b_to_a <- matrix(NA_real_, length(la), length(lb), dimnames = list(la, lb))
  for (i in la) for (j in lb)
    auc_b_to_a[i, j] <- ma_node_auroc(cache_a, labels_a, i, j)
  auc_a_to_b <- matrix(NA_real_, length(la), length(lb), dimnames = list(la, lb))
  for (j in unique(as.character(labels_b))) for (i in la)
    auc_a_to_b[i, j] <- ma_node_auroc(cache_b, labels_b, j, i)
  S <- (auc_a_to_b + auc_b_to_a) / 2
  list(M = 1 - S, S = S, auc_a_to_b = auc_a_to_b, auc_b_to_a = auc_b_to_a)
}

#' Convenience: run the whole measurement layer for two labeled datasets.
ma_measure <- function(expr_a, labels_a, expr_b, labels_b,
                       n_hvg = 1000, assume_log = FALSE, chunk = 2000L,
                       lib_a = NULL, lib_b = NULL) {
  ea <- ma_lognorm(expr_a, assume_log, lib_a)
  eb <- ma_lognorm(expr_b, assume_log, lib_b)
  hvg <- ma_variable_genes(ea, eb, n_hvg)
  na <- ma_rank_normalize(ea[hvg, , drop = FALSE])
  nb <- ma_rank_normalize(eb[hvg, , drop = FALSE])
  cache_a <- ma_vote_cache(na, nb, labels_b, chunk)  # A cells vs B leaves
  cache_b <- ma_vote_cache(nb, na, labels_a, chunk)  # B cells vs A leaves
  costs <- ma_leaf_costs(cache_a, labels_a, cache_b, labels_b)
  list(hvg = hvg, cache_a = cache_a, cache_b = cache_b,
       labels_a = labels_a, labels_b = labels_b, costs = costs)
}
