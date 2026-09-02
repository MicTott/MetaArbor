# Unit tests for the measurement kernel, chiefly the Appendix A claim:
# with a fixed network, merging training leaves and recomputing the cache
# equals summing the cached leaf columns — exactly — and node AUROCs from
# summed scores equal AUROCs from a merged-label rerun.
#
# Run:  Rscript tests/test_kernel.R

root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), ".."))
for (f in list.files(file.path(root, "R"), full.names = TRUE)) source(f)

set.seed(11)
sim <- tn_simulate_pair(n_family = 3, n_sub = 2, cells_per_leaf = 40,
                        n_genes = 600, batch_sd = 0.4, seed = 11)
ea <- tn_lognorm(sim$A$counts); eb <- tn_lognorm(sim$B$counts)
hvg <- tn_variable_genes(ea, eb, 400)
na <- tn_rank_normalize(ea[hvg, ]); nb <- tn_rank_normalize(eb[hvg, ])

# fine-label cache vs merged-label (family) cache over the same network
cache_fine <- tn_vote_cache(na, nb, sim$B$leaf, chunk = 500L)
cache_fam  <- tn_vote_cache(na, nb, sim$B$family, chunk = 500L)

for (fam in sim$families) {
  fine_cols <- grep(paste0("^", fam, "\\."), cache_fine$leaves, value = TRUE)
  summed <- rowSums(cache_fine$V[, fine_cols, drop = FALSE])
  merged <- cache_fam$V[, fam]
  stopifnot(max(abs(summed - merged)) < 1e-9)
  a1 <- tn_auroc(summed, sim$A$family == fam)
  a2 <- tn_auroc(merged, sim$A$family == fam)
  stopifnot(isTRUE(all.equal(a1, a2)))
}
cat("additivity: cached leaf-column sums == merged-label rerun (exact)\n")

# tn_aggregate_cache must reproduce the merged-label cache exactly
mapping <- setNames(sub("\\..*", "", cache_fine$leaves), cache_fine$leaves)
agg <- tn_aggregate_cache(cache_fine, mapping)
stopifnot(identical(agg$leaves, cache_fam$leaves),
          max(abs(agg$V - cache_fam$V[, agg$leaves])) < 1e-9,
          identical(unname(agg$leaf_sizes),
                    as.numeric(cache_fam$leaf_sizes[agg$leaves])))
cat("aggregate cache: equals merged-label rerun (exact)\n")

# the all-leaves union is uninformative: every test cell gets the same score,
# so the root AUROC is exactly 0.5. Floating-point accumulation breaks exact
# ties, so round to restore them before ranking (mathematically the scores
# are identical: the sum of all ranks per test cell is a constant).
tot <- rowSums(cache_fine$V)
stopifnot(diff(range(tot)) < 1e-9)
stopifnot(isTRUE(all.equal(tn_auroc(round(tot, 6), sim$A$family == sim$families[1]), 0.5)))
cat("root neutrality: full union scores constant, AUROC exactly 0.5\n")

# AUROC sanity: perfect separation and chance
stopifnot(tn_auroc(c(1, 2, 3, 4), c(FALSE, FALSE, TRUE, TRUE)) == 1)
stopifnot(tn_auroc(c(1, 2, 3, 4), c(TRUE, TRUE, FALSE, FALSE)) == 0)
stopifnot(tn_auroc(rep(1, 4), c(TRUE, FALSE, TRUE, FALSE)) == 0.5)
cat("auroc: boundary cases correct\n")

# directional asymmetry exists (the vignette-documented property):
meas <- tn_measure(sim$A$counts, sim$A$family, sim$B$counts, sim$B$leaf, n_hvg = 400)
stopifnot(!isTRUE(all.equal(meas$costs$auc_a_to_b, meas$costs$auc_b_to_a)))
cat("directionality: the two AUROC folds differ, as documented\n")

cat("\nall kernel tests passed\n")
