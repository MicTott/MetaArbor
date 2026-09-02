# Batch-severity robustness curve with the FROZEN estimator (see R/baseline.R
# header). Three conditions:
#
#   random   — stratified random half-split of 10Xv2: near-zero batch
#   donor    — donor-held-out split of 10Xv2 (largest donor vs other two):
#              mild real batch
#   platform — 10Xv2 vs 10Xv3 (the development run; metrics recomputed here
#              from its saved similarity where possible)
#
# Per condition: cluster self-RBH recovery, median self-AUROC, sibling margin
# (median over clusters of self-AUROC minus best off-target AUROC), forward
# family accuracy (subclass -> exact cluster leaf-set), reverse parent
# recovery (cluster -> subclass), median compactness both directions, and a
# classification of reverse failures: wrong_branch (different class),
# adjacent (same class), premature_stop (no selection, root stop), unmatched.
#
# NO estimator parameters are tuned here. Run: Rscript analysis/06_batch_conditions.R

suppressPackageStartupMessages(library(Matrix))
root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), ".."))
for (f in list.files(file.path(root, "R"), full.names = TRUE)) source(f)
dir_in <- file.path(root, "data", "wmb_plilaorb")
dir_out <- file.path(root, "results")

SIDE_FLOOR <- 15   # min cells per cluster per side within a condition

counts <- as.matrix(readMM(file.path(dir_in, "counts_10Xv2.mtx")))
rownames(counts) <- readLines(file.path(dir_in, "genes.txt"))
cells <- read.csv(file.path(dir_in, "cells_10Xv2.csv"))
colnames(counts) <- cells$cell_label
lib <- as.numeric(readLines(file.path(dir_in, "lib_10Xv2.txt")))
class_of <- setNames(cells$class, cells$subclass)[!duplicated(cells$subclass)]

classify_failures <- function(map, truth_parent, class_of) {
  bad <- map[!map$correct, ]
  if (!nrow(bad)) return(table(factor(character(0),
    levels = c("wrong_branch", "adjacent", "premature_stop", "unmatched"))))
  cls <- vapply(seq_len(nrow(bad)), function(k) {
    sel <- bad$selected[k]
    if (is.na(sel)) {
      if (bad$relation[k] == "unmatched") "unmatched" else "premature_stop"
    } else if (identical(unname(class_of[sel]),
                         unname(class_of[truth_parent[bad$query[k]]]))) {
      "adjacent"
    } else "wrong_branch"
  }, character(1))
  table(factor(cls, levels = c("wrong_branch", "adjacent",
                               "premature_stop", "unmatched")))
}

run_condition <- function(name, idx_a, idx_b) {
  ca <- cells[idx_a, ]; cb <- cells[idx_b, ]
  keep_cl <- intersect(names(which(table(ca$cluster) >= SIDE_FLOOR)),
                       names(which(table(cb$cluster) >= SIDE_FLOOR)))
  ia <- idx_a[ca$cluster %in% keep_cl]; ib <- idx_b[cb$cluster %in% keep_cl]
  ca <- cells[ia, ]; cb <- cells[ib, ]
  cat(sprintf("\n=== %s | A %d cells, B %d cells, %d clusters ===\n",
              name, length(ia), length(ib), length(keep_cl)))
  meas <- tn_measure(counts[, ia], ca$cluster, counts[, ib], cb$cluster,
                     n_hvg = 1000, lib_a = lib[ia], lib_b = lib[ib])

  S <- meas$costs$S
  self <- diag(S[, rownames(S)])
  offbest <- vapply(rownames(S), function(i)
    max(S[i, setdiff(colnames(S), i)]), numeric(1))
  best_ab <- colnames(S)[max.col(S, ties.method = "first")]
  best_ba <- rownames(S)[max.col(t(S), ties.method = "first")]
  rbh <- sum(vapply(seq_len(nrow(S)), function(i)
    best_ab[i] == rownames(S)[i] &&
    best_ba[match(rownames(S)[i], colnames(S))] == rownames(S)[i], logical(1)))

  subclasses <- sort(unique(ca$subclass))
  sub_of <- setNames(ca$subclass, ca$cluster)[!duplicated(ca$cluster)]
  cache_b_sub <- tn_aggregate_cache(meas$cache_b, sub_of)
  S_sub <- matrix(NA_real_, length(subclasses), ncol(S),
                  dimnames = list(subclasses, colnames(S)))
  for (i in subclasses) for (j in colnames(S)) {
    S_sub[i, j] <- (tn_node_auroc(meas$cache_a, ca$subclass, i, j) +
                    tn_node_auroc(cache_b_sub, cb$cluster, j, i)) / 2
  }

  tree_b <- tn_tree_from_levels(unique(cb[, c("class", "subclass",
                                              "supertype", "cluster")]))
  set.seed(7)
  fwd <- tn_baseline_map(meas$cache_a, ca$subclass, tree_b, S_sub)
  truth_lv <- lapply(subclasses, function(s)
    sort(unique(cb$cluster[cb$subclass == s])))
  names(truth_lv) <- subclasses
  fwd$correct <- mapply(function(id, q) {
    if (is.na(id)) return(FALSE)
    identical(sort(tn_leaves_under(tree_b, id)), truth_lv[[q]])
  }, fwd$selected, fwd$query)

  tree_a <- tn_tree_from_levels(data.frame(leaf = subclasses))
  set.seed(7)
  rev <- tn_baseline_map(cache_b_sub, cb$cluster, tree_a, t(S_sub))
  truth_parent <- setNames(cb$subclass, cb$cluster)[!duplicated(cb$cluster)]
  rev$correct <- !is.na(rev$selected) & rev$selected == truth_parent[rev$query]
  fail <- classify_failures(rev, truth_parent, class_of)

  res <- data.frame(
    condition = name, n_clusters = length(keep_cl),
    self_rbh = rbh, self_rbh_pct = round(100 * rbh / nrow(S)),
    med_self_auroc = round(median(self), 3),
    sib_margin = round(median(self - offbest), 3),
    fwd_exact = sprintf("%d/%d", sum(fwd$correct), nrow(fwd)),
    rev_parent = sprintf("%d/%d", sum(rev$correct), nrow(rev)),
    med_fwd_compact = round(median(fwd$compactness, na.rm = TRUE), 2),
    med_rev_compact = round(median(rev$compactness, na.rm = TRUE), 2),
    wrong_branch = fail[["wrong_branch"]], adjacent = fail[["adjacent"]],
    premature_stop = fail[["premature_stop"]], unmatched = fail[["unmatched"]])
  print(res, row.names = FALSE)
  write.csv(fwd, file.path(dir_out, sprintf("cond_%s_forward.csv", name)),
            row.names = FALSE)
  write.csv(rev, file.path(dir_out, sprintf("cond_%s_reverse.csv", name)),
            row.names = FALSE)
  res
}

set.seed(20260902)
## condition 1: stratified random half-split (near-zero batch)
idx_by_cl <- split(seq_len(nrow(cells)), cells$cluster)
half <- lapply(idx_by_cl, function(ix) sample(ix, floor(length(ix) / 2)))
idx_a1 <- sort(unlist(half))
idx_b1 <- setdiff(seq_len(nrow(cells)), idx_a1)

## condition 2: donor-held-out (largest donor vs the other two)
donors <- names(sort(table(cells$donor_label), decreasing = TRUE))
idx_a2 <- which(cells$donor_label == donors[1])
idx_b2 <- which(cells$donor_label != donors[1])

summary <- rbind(run_condition("random", idx_a1, idx_b1),
                 run_condition("donor", idx_a2, idx_b2))

## condition 3: cross-platform, recomputed from the development run's outputs
sims <- readRDS(file.path(dir_out, "wmb_similarity.rds"))
S3 <- sims$S
self3 <- diag(S3[, rownames(S3)])
off3 <- vapply(rownames(S3), function(i)
  max(S3[i, setdiff(colnames(S3), i)]), numeric(1))
best_ab <- colnames(S3)[max.col(S3, ties.method = "first")]
best_ba <- rownames(S3)[max.col(t(S3), ties.method = "first")]
rbh3 <- sum(vapply(seq_len(nrow(S3)), function(i)
  best_ab[i] == rownames(S3)[i] &&
  best_ba[match(rownames(S3)[i], colnames(S3))] == rownames(S3)[i], logical(1)))
fwd3 <- read.csv(file.path(dir_out, "wmb_map_subclass_to_tree.csv"))
rev3 <- read.csv(file.path(dir_out, "wmb_map_cluster_to_subclass.csv"))
truth_parent3 <- setNames(cells$subclass, cells$cluster)[!duplicated(cells$cluster)]
fail3 <- classify_failures(rev3, truth_parent3, class_of)
summary <- rbind(summary, data.frame(
  condition = "platform", n_clusters = nrow(S3),
  self_rbh = rbh3, self_rbh_pct = round(100 * rbh3 / nrow(S3)),
  med_self_auroc = round(median(self3), 3),
  sib_margin = round(median(self3 - off3), 3),
  fwd_exact = sprintf("%d/%d", sum(fwd3$correct), nrow(fwd3)),
  rev_parent = sprintf("%d/%d", sum(rev3$correct), nrow(rev3)),
  med_fwd_compact = round(median(fwd3$compactness, na.rm = TRUE), 2),
  med_rev_compact = round(median(rev3$compactness, na.rm = TRUE), 2),
  wrong_branch = fail3[["wrong_branch"]], adjacent = fail3[["adjacent"]],
  premature_stop = fail3[["premature_stop"]], unmatched = fail3[["unmatched"]]))

cat("\n=== robustness curve (frozen estimator) ===\n")
print(summary, row.names = FALSE)
write.csv(summary, file.path(dir_out, "batch_conditions_summary.csv"),
          row.names = FALSE)

cat("\ncross-platform reverse failures, classified:\n")
bad3 <- rev3[!rev3$correct, c("query", "selected", "auroc", "compactness", "relation")]
bad3$true_parent <- truth_parent3[bad3$query]
print(bad3, row.names = FALSE)
