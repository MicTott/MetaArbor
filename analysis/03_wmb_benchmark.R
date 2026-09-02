# Allen WMB PL-ILA-ORB benchmark, stages 2/4/5 of DESIGN.md §4.3.
#
# Pseudo-atlases from the real platform split:
#   atlas A = 10Xv2 cells, queried at SUBCLASS (coarse, ~25)
#   atlas B = 10Xv3 cells, tree over kept CLUSTERS (fine, ~100)
# Ground truth = the curated subclass -> cluster nesting.
#
# Stage 2: cluster-vs-cluster reciprocal best hits across platforms
# Stage 4: compact-support check per subclass
# Stage 5: baseline hierarchical selection, both directions, scored exactly
#
# Inputs: data/wmb_plilaorb/ (built by scratchpad export_wmb_subset.py)
# Run:    Rscript analysis/03_wmb_benchmark.R

suppressPackageStartupMessages(library(Matrix))
root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), ".."))
for (f in list.files(file.path(root, "R"), full.names = TRUE)) source(f)
dir_in <- file.path(root, "data", "wmb_plilaorb")
dir_out <- file.path(root, "results")
dir.create(dir_out, showWarnings = FALSE)

load_platform <- function(plat) {
  counts <- as.matrix(readMM(file.path(dir_in, sprintf("counts_%s.mtx", plat))))
  rownames(counts) <- readLines(file.path(dir_in, "genes.txt"))
  cells <- read.csv(file.path(dir_in, sprintf("cells_%s.csv", plat)))
  colnames(counts) <- cells$cell_label
  lib <- as.numeric(readLines(file.path(dir_in, sprintf("lib_%s.txt", plat))))
  list(counts = counts, cells = cells, lib = lib)
}

t0 <- Sys.time()
a <- load_platform("10Xv2")
b <- load_platform("10Xv3")
cat(sprintf("A (10Xv2): %d cells | B (10Xv3): %d cells | %d genes\n",
            ncol(a$counts), ncol(b$counts), nrow(a$counts)))

## measurement at cluster level (finest labels both sides) ------------------
## cached: the vote caches depend only on the data and finest labels, so
## estimator iterations reuse them (that is the point of the §3a design)
meas_file <- file.path(dir_out, "wmb_meas.rds")
if (file.exists(meas_file)) {
  meas <- readRDS(meas_file)
  cat("measurement loaded from cache\n")
} else {
  meas <- tn_measure(a$counts, a$cells$cluster, b$counts, b$cells$cluster,
                     n_hvg = 1000, lib_a = a$lib, lib_b = b$lib)
  saveRDS(meas, meas_file)
  cat(sprintf("joint HVGs: %d | measurement done %.1f min\n",
              length(meas$hvg), as.numeric(Sys.time() - t0, units = "mins")))
}

## stage 2: reciprocal best hits at matched resolution ----------------------
S <- meas$costs$S  # rows = A clusters, cols = B clusters (same taxonomy names)
best_ab <- colnames(S)[max.col(S, ties.method = "first")]
best_ba <- rownames(S)[max.col(t(S), ties.method = "first")]
rbh <- sum(vapply(seq_len(nrow(S)), function(i)
  best_ab[i] == rownames(S)[i] &&
  best_ba[match(rownames(S)[i], colnames(S))] == rownames(S)[i], logical(1)))
diag_auc <- diag(S[, rownames(S)])
cat(sprintf("stage 2 | clusters: %d | self-RBH: %d (%.0f%%) | median self AUROC %.3f | self AUROC > 0.9: %d\n",
            nrow(S), rbh, 100 * rbh / nrow(S), median(diag_auc), sum(diag_auc > 0.9)))

## subclass-level similarity (rows = A subclasses, cols = B clusters) -------
sub_of <- setNames(a$cells$subclass, a$cells$cluster)[!duplicated(a$cells$cluster)]
subclasses <- sort(unique(a$cells$subclass))
cache_b_sub <- tn_aggregate_cache(meas$cache_b, sub_of)
S_sub <- matrix(NA_real_, length(subclasses), ncol(S),
                dimnames = list(subclasses, colnames(S)))
for (i in subclasses) for (j in colnames(S)) {
  auc1 <- tn_node_auroc(meas$cache_a, a$cells$subclass, i, j)
  auc2 <- tn_node_auroc(cache_b_sub, b$cells$cluster, j, i)
  S_sub[i, j] <- (auc1 + auc2) / 2
}

## stage 4: compact support per subclass (cell-vote form: fraction of the
## subclass's cells whose argmax B cluster is one of its own clusters) ------
ms_a <- sweep(meas$cache_a$V, 2, meas$cache_a$leaf_sizes, "/")
top_leaf_a <- meas$cache_a$leaves[max.col(ms_a, ties.method = "first")]
compact <- vapply(subclasses, function(s) {
  own <- unique(a$cells$cluster[a$cells$subclass == s])
  mean(top_leaf_a[a$cells$subclass == s] %in% own)
}, numeric(1))
cat(sprintf("stage 4 | compact support: median %.2f | >= 0.7: %d/%d | min %.2f (%s)\n",
            median(compact, na.rm = TRUE), sum(compact >= 0.7, na.rm = TRUE),
            length(compact), min(compact, na.rm = TRUE),
            names(which.min(compact))))

## stage 5: baseline, A subclasses -> tree over B clusters ------------------
lev_b <- unique(b$cells[, c("class", "subclass", "supertype", "cluster")])
tree_b <- tn_tree_from_levels(lev_b)
set.seed(7)
map_ab <- tn_baseline_map(meas$cache_a, a$cells$subclass, tree_b, S_sub)
truth_leaves <- lapply(subclasses, function(s)
  sort(unique(b$cells$cluster[b$cells$subclass == s])))
names(truth_leaves) <- subclasses
sel_leaves <- lapply(map_ab$selected, function(id)
  if (is.na(id)) NA else sort(tn_leaves_under(tree_b, id)))
map_ab$correct <- mapply(function(sel, q) identical(sel, truth_leaves[[q]]),
                         sel_leaves, map_ab$query)
cat(sprintf("stage 5 | A subclass -> B tree: %d/%d exact leaf-set matches\n",
            sum(map_ab$correct), nrow(map_ab)))
print(table(map_ab$relation))
if (any(!map_ab$correct)) print(map_ab[!map_ab$correct, ], row.names = FALSE)

## stage 5 reverse: B clusters -> flat subclass tree of A -------------------
tree_a <- tn_tree_from_levels(data.frame(leaf = subclasses))
set.seed(7)
map_ba <- tn_baseline_map(cache_b_sub, b$cells$cluster, tree_a, t(S_sub))
truth_parent <- setNames(b$cells$subclass, b$cells$cluster)[!duplicated(b$cells$cluster)]
map_ba$correct <- !is.na(map_ba$selected) &
                  map_ba$selected == truth_parent[map_ba$query]
cat(sprintf("stage 5 | B cluster -> A subclasses: %d/%d correct parents\n",
            sum(map_ba$correct), nrow(map_ba)))
print(table(map_ba$relation))

write.csv(map_ab, file.path(dir_out, "wmb_map_subclass_to_tree.csv"), row.names = FALSE)
write.csv(map_ba, file.path(dir_out, "wmb_map_cluster_to_subclass.csv"), row.names = FALSE)
write.csv(data.frame(subclass = names(compact), compact_support = compact),
          file.path(dir_out, "wmb_compact_support.csv"), row.names = FALSE)
saveRDS(list(S = S, S_sub = S_sub), file.path(dir_out, "wmb_similarity.rds"))
cat(sprintf("total %.1f min; results in results/\n",
            as.numeric(Sys.time() - t0, units = "mins")))
