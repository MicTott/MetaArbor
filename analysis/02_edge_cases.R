# Edge cases the design must not mishandle (DESIGN.md §2.2):
#   novel      — a coarse type with no counterpart in the fine atlas
#   discordant — a coarse label that is secretly a mixture of two families
#
# Run:  Rscript analysis/02_edge_cases.R

root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), ".."))
for (f in list.files(file.path(root, "R"), full.names = TRUE)) source(f)

sim <- tn_simulate_pair(batch_sd = 0.5, seed = 42)

## --- novel: remove family F4 from atlas B entirely --------------------------
keep_b <- sim$B$family != "F4"
counts_b <- sim$B$counts[, keep_b]
leaf_b <- sim$B$leaf[keep_b]

meas <- tn_measure(sim$A$counts, sim$A$family, counts_b, leaf_b, n_hvg = 1000)
tree_b <- tn_tree_from_levels(data.frame(
  family = sub("\\..*", "", unique(leaf_b)), leaf = unique(leaf_b)))

set.seed(7)
map_a <- tn_baseline_map(meas$cache_a, meas$labels_a, tree_b, meas$costs$S)
print(map_a, row.names = FALSE)
f4 <- map_a[map_a$query == "F4", ]
others_ok <- all(map_a$selected[map_a$query != "F4"] ==
                 paste0("family:", c("F1", "F2", "F3")))
novel_ok <- f4$relation %in% c("unmatched", "discordant")
cat(sprintf("novel case: F4 -> %s (auroc %.3f) [%s]; other families correct: %s\n",
            f4$relation, f4$auroc, ifelse(novel_ok, "PASS", "FAIL"), others_ok))

## --- discordant: relabel a mixture of F1.s1 + F2.s1 cells as one type -------
labels_mix <- sim$A$family
mix_cells <- sim$A$leaf %in% c("F1.s1", "F2.s1")
labels_mix[mix_cells] <- "MIX"

meas2 <- tn_measure(sim$A$counts, labels_mix, sim$B$counts, sim$B$leaf, n_hvg = 1000)
tree_b_full <- tn_tree_from_levels(data.frame(
  family = sub("\\..*", "", sim$leaves), leaf = sim$leaves))

set.seed(7)
map_mix <- tn_baseline_map(meas2$cache_a, meas2$labels_a, tree_b_full, meas2$costs$S)
print(map_mix, row.names = FALSE)
mixrow <- map_mix[map_mix$query == "MIX", ]
mix_ok <- mixrow$relation %in% c("discordant", "unmatched")
cat(sprintf("mixture case: MIX -> %s (selected %s, compactness %.2f) [%s]\n",
            mixrow$relation, mixrow$selected, mixrow$compactness,
            ifelse(mix_ok, "PASS", "FAIL")))

stopifnot(novel_ok, others_ok, mix_ok)
cat("\nedge-case gate passed\n")
