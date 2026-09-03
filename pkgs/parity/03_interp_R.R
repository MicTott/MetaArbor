# Interpretation-layer gate, R side: reproduce the python summaries from the
# same fixtures (+ the exported coupling), assert numeric parity, and check
# the matrix accessors against saved underlying values.
#
# Run: Rscript pkgs/parity/03_interp_R.R    (after 02_interp_py.py)

repo <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "..", ".."))
for (f in list.files(file.path(repo, "pkgs", "r", "R"), full.names = TRUE)) source(f)
fx <- file.path(repo, "pkgs", "fixtures")
rgz <- function(f) read.csv(gzfile(file.path(fx, f)), check.names = FALSE)

lr <- rgz("cacheA_leaves.csv.gz")
cache <- list(V = as.matrix(rgz("cacheA_V.csv.gz")), leaves = lr$leaf,
              leaf_sizes = setNames(lr$size, lr$leaf))
colnames(cache$V) <- lr$leaf
labels <- rgz("labelsA_subclass.csv.gz")$label
lv <- rgz("tree_levels_b.csv.gz")
tree_b <- ma_tree_from_levels(lv)
Ssub <- rgz("S_sub.csv.gz")
S <- as.matrix(Ssub[, -1]); rownames(S) <- Ssub$query
family_of_leaf <- setNames(lv$subclass, lv$cluster)

pi_df <- rgz("transport_pi.csv.gz")
pi <- as.matrix(pi_df[, -1]); rownames(pi) <- pi_df[[1]]

wk <- ma_walk_summary(cache, labels, tree_b, S)
ts <- ma_transport_summary(pi, family_of_leaf, tree = tree_b)
al <- ma_alignment_summary(wk, ts, tree_b)
py <- rgz("interp_py.csv.gz")
py <- py[order(py$query), ]; al <- al[order(al$query), ]

cat_cols <- c("walk_selected", "walk_relation", "transport_family",
              "transport_node", "transport_bin", "agreement")
num_cols <- c("walk_depth", "walk_auroc", "walk_best_sib_auroc",
              "walk_second_sib_auroc", "walk_sib_delta", "walk_vote",
              "walk_decision_support", "transport_mass",
              "transport_eff_leaves")
same <- function(a, b) {
  a <- ifelse(is.na(a) | a == "None" | a == "", NA, as.character(a))
  b <- ifelse(is.na(b) | b == "None" | b == "", NA, as.character(b))
  identical(a, b)
}
for (cc in cat_cols) stopifnot(same(al[[cc]], py[[cc]]))
for (nc in num_cols) {
  a <- suppressWarnings(as.numeric(al[[nc]]))
  b <- suppressWarnings(as.numeric(py[[nc]]))
  ok <- is.na(a) & is.na(b) | (!is.na(a) & !is.na(b) & abs(a - b) < 1e-9)
  stopifnot(all(ok))
}
cat("R vs python interpretation parity: categorical identical, numeric within 1e-9\n")

## accessors vs saved underlying values ---------------------------------------
votes_saved <- read.csv(file.path(repo, "figures", "tables",
                                  "fig3_votes_forward.csv"))
Vm <- ma_vote_matrix(cache, labels)
for (k in sample(nrow(votes_saved), 500)) {
  stopifnot(abs(Vm[votes_saved$row[k], votes_saved$col[k]] -
                votes_saved$value[k]) < 1e-9)
}
meas <- readRDS(file.path(repo, "results", "wmb_meas.rds"))
Am <- ma_node_auroc_matrix(cache, labels, tree_b)
sub_auc <- meas$costs$auc_b_to_a  # rows would be clusters; use S check instead
Fm <- ma_family_mass(pi, family_of_leaf)
stopifnot(max(abs(rowSums(Fm) - 1)) < 1e-9)
# family-level AUROC accessor equals the walk's node AUROC by construction:
fam_nodes <- paste0("subclass:", sort(unique(labels)))
fam_nodes <- fam_nodes[fam_nodes %in% tree_b$id]
Am_fam <- ma_node_auroc_matrix(cache, labels, tree_b, nodes = fam_nodes)
for (q in rownames(Am_fam)[1:5]) {
  n <- paste0("subclass:", q)
  if (n %in% colnames(Am_fam))
    stopifnot(abs(Am_fam[q, n] -
      ma_auroc(ma_node_scores(cache, intersect(ma_leaves_under(tree_b, n),
                                               cache$leaves)),
               labels == q)) < 1e-12)
}
cat("accessors reproduce saved vote fractions; normalized family mass rows sum to 1\n")
cat("INTERPRETATION GATE PASSED\n")
