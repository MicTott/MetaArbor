# Package parity, stage 1 (R side).
#
# GATE: the packaged walk (MINSTD bootstrap, per-query seeds) must reproduce
# the saved frozen-benchmark selections exactly — the RNG re-plumbing is
# packaging, not estimation, and this proves the decisions did not move.
#
# Then export fixtures for the python package's parity tests: the real
# platform vote caches, trees, similarity, the packaged-R walk outputs
# (python must match them), simulation inputs/outputs, and a MINSTD stream
# check vector.
#
# Run: Rscript pkgs/parity/01_export_and_gate.R   (from the repo root)

repo <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "..", ".."))
pkgR <- file.path(repo, "pkgs", "r", "R")
for (f in list.files(pkgR, full.names = TRUE)) source(f)
fx <- file.path(repo, "pkgs", "fixtures")
dir.create(fx, showWarnings = FALSE)

meas <- readRDS(file.path(repo, "results", "wmb_meas.rds"))
cells_a <- read.csv(file.path(repo, "data", "wmb_plilaorb", "cells_10Xv2.csv"))
cells_b <- read.csv(file.path(repo, "data", "wmb_plilaorb", "cells_10Xv3.csv"))
sims <- readRDS(file.path(repo, "results", "wmb_similarity.rds"))
fwd_saved <- read.csv(file.path(repo, "results", "wmb_map_subclass_to_tree.csv"))
rev_saved <- read.csv(file.path(repo, "results", "wmb_map_cluster_to_subclass.csv"))

tax_b <- unique(cells_b[, c("class", "subclass", "supertype", "cluster")])
tax_b <- tax_b[order(tax_b$class, tax_b$subclass, tax_b$supertype, tax_b$cluster), ]
tree_b <- ma_tree_from_levels(tax_b)
subclasses <- sort(unique(cells_a$subclass))
tree_a <- ma_tree_from_levels(data.frame(leaf = subclasses))
sub_of <- setNames(cells_a$subclass, cells_a$cluster)[!duplicated(cells_a$cluster)]
cache_b_sub <- ma_aggregate_cache(meas$cache_b, sub_of)

## ---- identity gate --------------------------------------------------------
fwd <- ma_baseline_map(meas$cache_a, cells_a$subclass, tree_b, sims$S_sub)
rev <- ma_baseline_map(cache_b_sub, cells_b$cluster, tree_a, t(sims$S_sub))
same_fwd <- identical(fwd$selected, fwd_saved$selected) &&
            identical(fwd$relation, fwd_saved$relation)
same_rev <- identical(rev$selected, rev_saved$selected) &&
            identical(rev$relation, rev_saved$relation)
cat(sprintf("identity gate | forward: %s | reverse: %s\n",
            ifelse(same_fwd, "IDENTICAL", "DIFFERS"),
            ifelse(same_rev, "IDENTICAL", "DIFFERS")))
if (!same_fwd) {
  d <- which(fwd$selected != fwd_saved$selected | fwd$relation != fwd_saved$relation)
  print(cbind(fwd[d, c("query", "selected", "relation")],
              saved_sel = fwd_saved$selected[d], saved_rel = fwd_saved$relation[d]))
}
if (!same_rev) {
  d <- which(rev$selected != rev_saved$selected | rev$relation != rev_saved$relation |
             is.na(rev$selected) != is.na(rev_saved$selected))
  print(cbind(rev[d, c("query", "selected", "relation")],
              saved_sel = rev_saved$selected[d], saved_rel = rev_saved$relation[d]))
}

## ---- fixtures for python parity -------------------------------------------
wgz <- function(x, f) {
  con <- gzfile(file.path(fx, f), "w")
  write.csv(x, con, row.names = FALSE)
  close(con)
}
wgz(data.frame(meas$cache_a$V, check.names = FALSE), "cacheA_V.csv.gz")
wgz(data.frame(leaf = meas$cache_a$leaves, size = meas$cache_a$leaf_sizes),
    "cacheA_leaves.csv.gz")
wgz(data.frame(cache_b_sub$V, check.names = FALSE), "cacheBsub_V.csv.gz")
wgz(data.frame(leaf = cache_b_sub$leaves, size = cache_b_sub$leaf_sizes),
    "cacheBsub_leaves.csv.gz")
wgz(data.frame(label = cells_a$subclass), "labelsA_subclass.csv.gz")
wgz(data.frame(label = cells_b$cluster), "labelsB_cluster.csv.gz")
wgz(tax_b, "tree_levels_b.csv.gz")
wgz(data.frame(subclass = subclasses), "subclasses.csv.gz")
Ssub <- data.frame(sims$S_sub, check.names = FALSE)
Ssub <- cbind(query = rownames(sims$S_sub), Ssub)
wgz(Ssub, "S_sub.csv.gz")
wgz(fwd, "walkR_forward.csv.gz")
wgz(rev, "walkR_reverse.csv.gz")

# MINSTD stream check (seed 42, first 10 raw states; and indices mod 7)
rng <- ma_minstd_new(42)
states <- vapply(1:10, function(i) { ma_minstd_index(rng, 7); rng$state }, 0)
rng2 <- ma_minstd_new(42)
idx7 <- vapply(1:10, function(i) ma_minstd_index(rng2, 7), 0)
wgz(data.frame(state = states, idx7 = idx7), "minstd_check.csv.gz")

# simulation fixtures: inputs + package-R measurement + walk outputs
sim <- ma_simulate_pair(batch_sd = 0.5, seed = 42)
m2 <- ma_measure(sim$A$counts, sim$A$family, sim$B$counts, sim$B$leaf, n_hvg = 1000)
tree_bs <- ma_tree_from_levels(data.frame(family = sub("\\..*", "", sim$leaves),
                                          leaf = sim$leaves))
map_sim <- ma_baseline_map(m2$cache_a, m2$labels_a, tree_bs, m2$costs$S)
wgz(data.frame(gene = rownames(sim$A$counts), sim$A$counts, check.names = FALSE),
    "sim_countsA.csv.gz")
wgz(data.frame(gene = rownames(sim$B$counts), sim$B$counts, check.names = FALSE),
    "sim_countsB.csv.gz")
wgz(data.frame(label = sim$A$family), "sim_labelsA.csv.gz")
wgz(data.frame(label = sim$B$leaf), "sim_labelsB.csv.gz")
wgz(data.frame(hvg = m2$hvg), "sim_hvg.csv.gz")
Ss <- cbind(query = rownames(m2$costs$S),
            data.frame(m2$costs$S, check.names = FALSE))
wgz(Ss, "sim_S.csv.gz")
wgz(map_sim, "sim_walkR.csv.gz")
cat("fixtures written to", fx, "\n")
stopifnot(same_fwd, same_rev)
cat("GATE PASSED\n")
