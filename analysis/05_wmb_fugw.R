# Stage 6 (DESIGN.md §4.3): FUGW on the WMB PL-ILA-ORB benchmark, against
# the same ground truth as the baseline — the deciding comparison.
#
# Terminal populations: A = 25 subclasses (10Xv2), B = kept clusters (10Xv3).
# Molecular cost: 1 - S_sub from analysis/03 (symmetrized MetaNeighbor).
# Structure: class->subclass tree distances (A), full curated tree (B).
#
# Requires: results/wmb_similarity.rds from analysis/03_wmb_benchmark.R
# Run: TN_PYTHON=<python-with-POT> Rscript analysis/05_wmb_fugw.R

root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), ".."))
for (f in list.files(file.path(root, "R"), full.names = TRUE)) source(f)
script <- file.path(root, "python", "fugw_run.py")

S_sub <- readRDS(file.path(root, "results", "wmb_similarity.rds"))$S_sub
cells_a <- read.csv(file.path(root, "data", "wmb_plilaorb", "cells_10Xv2.csv"))
cells_b <- read.csv(file.path(root, "data", "wmb_plilaorb", "cells_10Xv3.csv"))

tree_a <- tn_tree_from_levels(unique(cells_a[, c("class", "subclass")]))
tree_b <- tn_tree_from_levels(unique(cells_b[, c("class", "subclass",
                                                 "supertype", "cluster")]))
CA <- tn_leaf_path_dist(tree_a)[rownames(S_sub), rownames(S_sub)]
CB <- tn_leaf_path_dist(tree_b)[colnames(S_sub), colnames(S_sub)]
M <- 1 - S_sub

truth_leaves <- lapply(rownames(S_sub), function(s)
  sort(unique(cells_b$cluster[cells_b$subclass == s])))
names(truth_leaves) <- rownames(S_sub)

score <- function(assign) {
  sel <- lapply(seq_len(nrow(assign)), function(k) {
    if (is.na(assign$node[k]) || assign$node[k] == "root") return(NA)
    sort(tn_leaves_under(tree_b, assign$node[k]))
  })
  mapply(function(s, q) identical(s, truth_leaves[[q]]), sel, assign$query)
}

results <- list()
for (alpha in c(0.3, 0.5, 0.7, 0.9)) {
  fit <- tn_fugw(M, CA, CB, alpha = alpha, rho = 1, epsilon = 0,
                 script = script)
  assign <- tn_fugw_assign(fit$pi, tree_b, q = 0.9)
  ok <- score(assign)
  # mass concentration on true subclass subtrees
  own <- vapply(rownames(M), function(s) {
    row <- fit$pi[s, ]
    sum(row[names(row) %in% truth_leaves[[s]]]) / sum(row)
  }, numeric(1))
  cat(sprintf("alpha %.1f | exact node %d/%d | median own-subtree mass %.3f | P-Q gap %.1e\n",
              alpha, sum(ok), length(ok), median(own), fit$pi_gap))
  results[[as.character(alpha)]] <- data.frame(
    alpha = alpha, query = assign$query, node = assign$node,
    share = assign$share, own_mass = own[assign$query], correct = ok)
}
out <- do.call(rbind, results)
write.csv(out, file.path(root, "results", "wmb_fugw_assignments.csv"),
          row.names = FALSE)

base <- read.csv(file.path(root, "results", "wmb_map_subclass_to_tree.csv"))
cat(sprintf("\nbaseline (stage 5): %d/%d exact | FUGW best alpha: %d/%d exact\n",
            sum(base$correct), nrow(base),
            max(tapply(out$correct, out$alpha, sum)), nrow(base)))
cat("per-query disagreements at best alpha are the cases to inspect by hand\n")
