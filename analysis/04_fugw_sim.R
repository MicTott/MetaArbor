# FUGW estimator validation on simulation (DESIGN.md §3c-3d).
#
# Coarse atlas A (4 families, flat tree) vs fine atlas B (12 subtypes,
# family-structured tree). The coupling should split each family's mass
# across its own three subtypes — mass splitting as the representation of
# resolution difference — and the rolled-up assignment should pick each
# family node. A novel-family case checks that unbalanced marginals leave
# unmatched mass on the table instead of forcing it elsewhere.
#
# Requires a python with POT: TN_PYTHON=<path> Rscript analysis/04_fugw_sim.R

root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), ".."))
for (f in list.files(file.path(root, "R"), full.names = TRUE)) source(f)
script <- file.path(root, "python", "fugw_run.py")

sim <- tn_simulate_pair(batch_sd = 0.5, seed = 42)
meas <- tn_measure(sim$A$counts, sim$A$family, sim$B$counts, sim$B$leaf, n_hvg = 1000)

tree_a <- tn_tree_from_levels(data.frame(leaf = sim$families))
tree_b <- tn_tree_from_levels(data.frame(family = sub("\\..*", "", sim$leaves),
                                         leaf = sim$leaves))
CA <- tn_leaf_path_dist(tree_a)
CB <- tn_leaf_path_dist(tree_b)
M <- meas$costs$M[rownames(CA), rownames(CB)]

cat("=== alpha sweep, batch_sd 0.5 ===\n")
for (alpha in c(0.3, 0.5, 0.8)) {
  fit <- tn_fugw(M, CA, CB, alpha = alpha, rho = 1, epsilon = 1e-3,
                 script = script)
  assign <- tn_fugw_assign(fit$pi, tree_b, q = 0.9)
  ok <- assign$node == paste0("family:", assign$query)
  # per-family share of mass inside own family subtree
  own_share <- vapply(sim$families, function(f) {
    row <- fit$pi[f, ]
    sum(row[grep(paste0("^", f, "\\."), names(row))]) / sum(row)
  }, numeric(1))
  cat(sprintf("alpha %.1f | assignments correct %d/4 | own-family mass %.3f-%.3f | P-Q gap %.1e\n",
              alpha, sum(ok), min(own_share), max(own_share), fit$pi_gap))
  if (any(!ok)) print(assign[!ok, ], row.names = FALSE)
}

cat("\n=== novel family: F4 absent from B ===\n")
keep <- !grepl("^F4\\.", colnames(M))
tree_b3 <- tn_tree_from_levels(data.frame(
  family = sub("\\..*", "", sim$leaves[!grepl("^F4", sim$leaves)]),
  leaf = sim$leaves[!grepl("^F4", sim$leaves)]))
CB3 <- tn_leaf_path_dist(tree_b3)
fit3 <- tn_fugw(M[, keep], CA, CB3, alpha = 0.5, rho = 1, epsilon = 1e-3,
                script = script)
sent <- rowSums(fit3$pi) / (1 / nrow(M))   # mass sent, relative to marginal
cat("relative mass transported per family (F4 should be depressed):\n")
print(round(sent, 3))
stopifnot(sent["F4"] < 0.8 * min(sent[c("F1", "F2", "F3")]))
cat("novel-family gate passed\n")
