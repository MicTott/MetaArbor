# Validation of the measurement kernel + baseline estimator on simulated
# hierarchies with known ground truth (DESIGN.md §4.3 stages 4-5, in vitro).
#
# Setup: atlas A is labeled coarse (families), atlas B fine (subtypes).
# Truth: every A family should select its family node in T_B (relation
# "family", never the root, never a lone leaf); every B subtype should select
# its family leaf in T_A (relation "leaf"). Swept over batch severity.
#
# Run:  Rscript analysis/01_simulation_validation.R

root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), ".."))
for (f in list.files(file.path(root, "R"), full.names = TRUE)) source(f)

check_direction <- function(map, truth, direction) {
  ok <- mapply(function(q, sel) identical(sel, truth[[q]]), map$query, map$selected)
  cat(sprintf("  %s: %d/%d correct\n", direction, sum(ok), nrow(map)))
  if (any(!ok)) print(map[!ok, ], row.names = FALSE)
  all(ok)
}

results <- list()
for (batch_sd in c(0, 0.5, 1.0)) {
  cat(sprintf("\n=== batch_sd = %.1f ===\n", batch_sd))
  sim <- tn_simulate_pair(batch_sd = batch_sd, seed = 42)

  meas <- tn_measure(sim$A$counts, sim$A$family, sim$B$counts, sim$B$leaf,
                     n_hvg = 1000)

  # trees: T_B has family internal nodes over subtype leaves; T_A is flat
  tree_b <- tn_tree_from_levels(data.frame(
    family = sub("\\..*", "", sim$leaves), leaf = sim$leaves))
  tree_a <- tn_tree_from_levels(data.frame(leaf = sim$families))

  set.seed(7)
  map_a <- tn_baseline_map(meas$cache_a, meas$labels_a, tree_b, meas$costs$S)
  map_b <- tn_baseline_map(meas$cache_b, meas$labels_b, tree_a, t(meas$costs$S))

  truth_a <- setNames(as.list(paste0("family:", sim$families)), sim$families)
  truth_b <- setNames(as.list(sub("\\..*", "", sim$leaves)), sim$leaves)

  ok_a <- check_direction(map_a, truth_a, "A families -> T_B nodes")
  ok_b <- check_direction(map_b, truth_b, "B subtypes -> T_A leaves")
  cat("  relations A->B:", paste(sort(table(map_a$relation)), collapse = " "),
      names(sort(table(map_a$relation))), "\n")
  cat(sprintf("  mean symmetrized AUROC of true pairs: %.3f\n",
      mean(mapply(function(i) mean(meas$costs$S[i, grep(paste0("^", i, "\\."),
        colnames(meas$costs$S))]), sim$families))))
  results[[as.character(batch_sd)]] <- list(ok_a = ok_a, ok_b = ok_b,
                                            map_a = map_a, map_b = map_b)
}

cat("\n=== summary ===\n")
for (b in names(results))
  cat(sprintf("batch_sd %s : A->B %s | B->A %s\n", b,
      ifelse(results[[b]]$ok_a, "PASS", "FAIL"),
      ifelse(results[[b]]$ok_b, "PASS", "FAIL")))

# hard gate at mild batch severity; stronger settings reported, not gated
stopifnot(results[["0"]]$ok_a, results[["0"]]$ok_b,
          results[["0.5"]]$ok_a, results[["0.5"]]$ok_b)
cat("\nvalidation gate passed\n")
