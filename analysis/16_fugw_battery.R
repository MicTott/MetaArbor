# FROZEN FUGW through the three-condition batch battery (NOTES.md item 14a).
#
# Frozen configuration (chosen before the intrinsic comparison; NOT a
# retrospective 23/23 setting): cost = 1 - S_sub (raw), rho = 0.3,
# alpha = 0.9, epsilon = 0 (mm solver), refinement-invariant marginals
# (tn_tree_weights) on both sides. Readouts: argmax-family accuracy,
# mass-based confidence categories, coupling concentration (family- and
# leaf-level entropy), singleton status, and cross-condition assignment
# stability. NOTHING is tuned here.
#
# Conditions replicate analysis/06 exactly (same seed, same floors):
#   random   — stratified half-split of 10Xv2
#   donor    — donor-held-out split of 10Xv2
#   platform — 10Xv2 vs 10Xv3 (existing measurement)
#
# Run: TN_PYTHON=<python-with-POT> Rscript analysis/16_fugw_battery.R

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "fig_common.R"))
source(file.path(fig_root, "analysis", "fugw_diag.R"))
suppressPackageStartupMessages(library(Matrix))

FROZEN <- list(rho = 0.3, alpha = 0.9, epsilon = 0)

counts <- as.matrix(readMM(file.path(fig_root, "data", "wmb_plilaorb", "counts_10Xv2.mtx")))
rownames(counts) <- readLines(file.path(fig_root, "data", "wmb_plilaorb", "genes.txt"))
cells <- read.csv(file.path(fig_root, "data", "wmb_plilaorb", "cells_10Xv2.csv"))
colnames(counts) <- cells$cell_label
lib <- as.numeric(readLines(file.path(fig_root, "data", "wmb_plilaorb", "lib_10Xv2.txt")))
SIDE_FLOOR <- 15

## splits identical to analysis/06 -------------------------------------------
set.seed(20260902)
idx_by_cl <- split(seq_len(nrow(cells)), cells$cluster)
half <- lapply(idx_by_cl, function(ix) sample(ix, floor(length(ix) / 2)))
idx_a1 <- sort(unlist(half))
idx_b1 <- setdiff(seq_len(nrow(cells)), idx_a1)
donors <- names(sort(table(cells$donor_label), decreasing = TRUE))
idx_a2 <- which(cells$donor_label == donors[1])
idx_b2 <- which(cells$donor_label != donors[1])

condition_S_sub <- function(name, idx_a, idx_b) {
  meas_file <- file.path(dir_res, sprintf("cond_meas_%s.rds", name))
  if (file.exists(meas_file)) return(readRDS(meas_file))
  ca <- cells[idx_a, ]; cb <- cells[idx_b, ]
  keep_cl <- intersect(names(which(table(ca$cluster) >= SIDE_FLOOR)),
                       names(which(table(cb$cluster) >= SIDE_FLOOR)))
  ia <- idx_a[ca$cluster %in% keep_cl]; ib <- idx_b[cb$cluster %in% keep_cl]
  ca <- cells[ia, ]; cb <- cells[ib, ]
  meas <- tn_measure(counts[, ia], ca$cluster, counts[, ib], cb$cluster,
                     n_hvg = 1000, lib_a = lib[ia], lib_b = lib[ib])
  subcl <- sort(unique(ca$subclass))
  sub_of <- setNames(ca$subclass, ca$cluster)[!duplicated(ca$cluster)]
  cache_b_sub <- tn_aggregate_cache(meas$cache_b, sub_of)
  S_sub <- matrix(NA_real_, length(subcl), length(keep_cl),
                  dimnames = list(subcl, sort(keep_cl)))
  for (i in subcl) for (j in colnames(S_sub)) {
    S_sub[i, j] <- (tn_node_auroc(meas$cache_a, ca$subclass, i, j) +
                    tn_node_auroc(cache_b_sub, cb$cluster, j, i)) / 2
  }
  out <- list(S_sub = S_sub, cells_a = ca, cells_b = cb)
  saveRDS(out, meas_file)
  out
}

conditions <- list(
  random = condition_S_sub("random", idx_a1, idx_b1),
  donor = condition_S_sub("donor", idx_a2, idx_b2),
  platform = list(S_sub = sims$S_sub, cells_a = cells_a, cells_b = cells_b))

## frozen FUGW per condition -------------------------------------------------
rows <- list(); assign_tbl <- list(); perq_all <- list()
for (nm in names(conditions)) {
  cond <- conditions[[nm]]
  qn <- rownames(cond$S_sub); ln <- colnames(cond$S_sub)
  tp <- setNames(cond$cells_b$subclass, cond$cells_b$cluster)[!duplicated(cond$cells_b$cluster)]
  tl <- lapply(qn, function(s) sort(unique(cond$cells_b$cluster[cond$cells_b$subclass == s])))
  names(tl) <- qn
  tax <- unique(cond$cells_b[, c("class", "subclass", "supertype", "cluster")])
  tax <- tax[order(tax$class, tax$subclass, tax$supertype, tax$cluster), ]
  tr_b <- tn_tree_from_levels(tax)
  tr_a <- tn_tree_from_levels(unique(cond$cells_a[, c("class", "subclass")]))
  wA <- tn_tree_weights(tr_a)[qn]
  wB <- tn_tree_weights(tr_b)[ln]
  fit <- tn_fugw(1 - cond$S_sub, tn_leaf_path_dist(tr_a)[qn, qn],
                 tn_leaf_path_dist(tr_b)[ln, ln], wA = wA, wB = wB,
                 alpha = FROZEN$alpha, rho = FROZEN$rho,
                 epsilon = FROZEN$epsilon,
                 script = file.path(fig_root, "python", "fugw_run.py"))
  d <- fugw_decompose(fit$pi, qn, ln, tp = tp, tl = tl)
  d$category <- fugw_classify(d)
  d$condition <- nm
  perq_all[[nm]] <- d
  singles <- qn[vapply(qn, function(q) length(tl[[q]]) == 1, logical(1))]
  # argmax family per query, for stability
  fam_of_leaf <- unname(tp[ln])
  assign_tbl[[nm]] <- setNames(vapply(qn, function(q) {
    p <- fit$pi[q, ] / sum(fit$pi[q, ])
    names(which.max(tapply(p, fam_of_leaf, sum)))
  }, character(1)), qn)
  rows[[nm]] <- data.frame(
    condition = nm, n_queries = length(qn), n_clusters = length(ln),
    argmax = sum(d$argmax_correct),
    confident = sum(d$category == "confident_correct"),
    underconf = sum(d$category == "underconfident_correct"),
    diffuse = sum(d$category == "diffuse_correct"),
    cross_family = sum(d$category == "cross_family_failure"),
    singleton_ok = sum(d$argmax_correct[d$query %in% singles]),
    n_singletons = length(singles),
    med_H_family = round(median(d$H_family, na.rm = TRUE), 3),
    med_eff_ratio = round(median(d$eff_ratio, na.rm = TRUE), 2),
    pi_gap = signif(fit$pi_gap, 2))
}
summary <- do.call(rbind, rows)
write.csv(summary, file.path(dir_res, "fugw_battery_summary.csv"), row.names = FALSE)
write.csv(do.call(rbind, perq_all), file.path(dir_res, "fugw_battery_perquery.csv"),
          row.names = FALSE)

cat("=== frozen FUGW, three-condition battery ===\n")
print(summary, row.names = FALSE)

## cross-condition assignment stability --------------------------------------
common <- Reduce(intersect, lapply(assign_tbl, names))
ag <- sapply(assign_tbl, function(a) a[common])
stable <- rowSums(ag == ag[, 1]) == ncol(ag)
cat(sprintf("\nassignment stability: %d/%d queries identical argmax family across all three conditions\n",
            sum(stable), length(common)))
if (any(!stable)) print(ag[!stable, , drop = FALSE])

cat("\nwalk-estimator battery for comparison (results/batch_conditions_summary.csv):\n")
print(read.csv(file.path(dir_res, "batch_conditions_summary.csv"))[,
      c("condition", "fwd_exact", "rev_parent")], row.names = FALSE)
