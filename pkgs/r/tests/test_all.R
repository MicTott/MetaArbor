# Package tests: kernel invariants, tree-weight leak criteria, MINSTD
# stream, and a small end-to-end walk on simulated data.
# R CMD check runs this via tests/; it stops on the first failure.

library(MetaArbor)

## kernel invariants ---------------------------------------------------------
sim <- ma_simulate_pair(n_family = 3, n_sub = 2, cells_per_leaf = 40,
                        n_genes = 600, batch_sd = 0.4, seed = 11)
ea <- ma_lognorm(sim$A$counts); eb <- ma_lognorm(sim$B$counts)
hvg <- ma_variable_genes(ea, eb, 400)
na <- ma_rank_normalize(ea[hvg, ]); nb <- ma_rank_normalize(eb[hvg, ])
cache_fine <- ma_vote_cache(na, nb, sim$B$leaf, chunk = 500L)
cache_fam <- ma_vote_cache(na, nb, sim$B$family, chunk = 500L)
mapping <- setNames(sub("\\..*", "", cache_fine$leaves), cache_fine$leaves)
agg <- ma_aggregate_cache(cache_fine, mapping)
stopifnot(identical(agg$leaves, cache_fam$leaves),
          max(abs(agg$V - cache_fam$V[, agg$leaves])) < 1e-9)
tot <- rowSums(cache_fine$V)
stopifnot(diff(range(tot)) < 1e-9)
stopifnot(isTRUE(all.equal(ma_auroc(round(tot, 6),
                                    sim$A$family == sim$families[1]), 0.5)))
stopifnot(ma_auroc(c(1, 2, 3, 4), c(FALSE, FALSE, TRUE, TRUE)) == 1,
          ma_auroc(c(1, 2, 3, 4), c(TRUE, TRUE, FALSE, FALSE)) == 0)

## tree weights: leak criteria -----------------------------------------------
toy <- ma_tree_from_levels(data.frame(
  fam = c("A", "A", "A", "A", "B", "C", "C"),
  leaf = c("a1", "a2", "a3", "a4", "b1", "c1", "c2")))
w <- ma_tree_weights(toy)
stopifnot(abs(sum(w) - 1) < 1e-12, abs(w[["b1"]] - 1/3) < 1e-12,
          abs(w[["a1"]] - 1/12) < 1e-12)
toy_ref <- ma_tree_from_levels(data.frame(
  fam = c(rep("A", 4), rep("B", 10), "C", "C"),
  leaf = c("a1", "a2", "a3", "a4", paste0("b1_", 1:10), "c1", "c2")))
stopifnot(abs(sum(ma_tree_weights(toy_ref)[paste0("b1_", 1:10)]) - 1/3) < 1e-12)
wa <- ma_tree_weights(toy, all_nodes = TRUE)
for (v in toy$id[!toy$is_leaf]) {
  kids <- ma_children(toy, v)
  if (length(kids)) stopifnot(abs(wa[v] - sum(wa[kids])) < 1e-12)
}

## MINSTD determinism --------------------------------------------------------
r1 <- ma_minstd_new(42); r2 <- ma_minstd_new(42)
stopifnot(identical(ma_minstd_indices(r1, 20, 7), ma_minstd_indices(r2, 20, 7)))

## end-to-end walk on simulation ---------------------------------------------
sim2 <- ma_simulate_pair(batch_sd = 0.5, seed = 42)
m <- ma_measure(sim2$A$counts, sim2$A$family, sim2$B$counts, sim2$B$leaf,
                n_hvg = 1000)
tree_b <- ma_tree_from_levels(data.frame(
  family = sub("\\..*", "", sim2$leaves), leaf = sim2$leaves))
map <- ma_baseline_map(m$cache_a, m$labels_a, tree_b, m$costs$S)
stopifnot(identical(map$selected, paste0("family:", sim2$families)),
          all(map$relation == "family"))
cat("all MetaArbor package tests passed\n")
