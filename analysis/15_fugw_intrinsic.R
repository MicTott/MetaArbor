# Intrinsic recursive marginals for FUGW (NOTES.md item 14): the honest,
# prespecifiable version of hierarchy balancing. analysis/14's "hierarchy"
# scheme was ORACLE LEVEL BALANCING - it equalized mass across B's subclass
# level, a level chosen by the analyst with knowledge that the source labels
# are subclasses. Here weights are a pure function of each tree alone
# (tn_tree_weights): mass 1 at the root, split equally at every internal
# node, on BOTH sides independently. No level selection, no reference to the
# paired atlas.
#
# Unit tests (prespecified): source independence, refinement invariance,
# name invariance, mass conservation. Then the identical fig6 grid, no
# tuning, same read-outs as analysis/14.
#
# Run: TN_PYTHON=<python-with-POT> Rscript analysis/15_fugw_intrinsic.R

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "fig_common.R"))
source(file.path(fig_root, "analysis", "fugw_diag.R"))
qn <- rownames(sims$S_sub); ln <- colnames(sims$S_sub)

## ---- unit tests for tn_tree_weights ---------------------------------------
toy <- tn_tree_from_levels(data.frame(
  fam = c("A", "A", "A", "A", "B", "C", "C"),
  leaf = c("a1", "a2", "a3", "a4", "b1", "c1", "c2")))
w <- tn_tree_weights(toy)
stopifnot(abs(sum(w) - 1) < 1e-12,
          abs(w[["b1"]] - 1/3) < 1e-12,          # singleton keeps its branch
          abs(w[["a1"]] - 1/12) < 1e-12,         # 4-way family splits
          abs(w[["c1"]] - 1/6) < 1e-12)
# mass conservation at every internal node
wa <- tn_tree_weights(toy, all_nodes = TRUE)
for (v in toy$id[!toy$is_leaf]) {
  kids <- tn_children(toy, v)
  if (length(kids)) stopifnot(abs(wa[v] - sum(wa[kids])) < 1e-12)
}
# refinement invariance: split b1 into 10 children; branch B total unchanged
toy_ref <- tn_tree_from_levels(data.frame(
  fam = c(rep("A", 4), rep("B", 10), "C", "C"),
  leaf = c("a1", "a2", "a3", "a4", paste0("b1_", 1:10), "c1", "c2")))
w_ref <- tn_tree_weights(toy_ref)
stopifnot(abs(sum(w_ref[paste0("b1_", 1:10)]) - 1/3) < 1e-12)
# name invariance: permute labels, weights follow the topology
toy_perm <- tn_tree_from_levels(data.frame(
  fam = c("X", "X", "X", "X", "Y", "Z", "Z"),
  leaf = c("p1", "p2", "p3", "p4", "q1", "r1", "r2")))
stopifnot(identical(unname(sort(tn_tree_weights(toy_perm))), unname(sort(w))))
# source independence is structural: the function takes only the tree
cat("unit tests passed: conservation, refinement, name invariance, singleton branch\n")

## ---- intrinsic weights for both real trees --------------------------------
tree_a2 <- tn_tree_from_levels(unique(cells_a[, c("class", "subclass")]))
wA <- tn_tree_weights(tree_a2)[qn]
wB <- tn_tree_weights(tree_b)[ln]
stopifnot(abs(sum(wA) - 1) < 1e-9, abs(sum(wB) - 1) < 1e-9)

# diagnostic (not an input): how well do branch capacities align here?
famB <- tapply(wB, unname(truth_parent[ln]), sum)[qn]
cat(sprintf("branch capacity alignment |wA(subclass) - wB(family)|: max %.2e (shared taxonomy => expected ~0)\n",
            max(abs(wA - famB))))

## ---- identical grid, intrinsic weights ------------------------------------
CA <- tn_leaf_path_dist(tree_a2)[qn, qn]
CB <- tn_leaf_path_dist(tree_b)[ln, ln]
work <- file.path(dir_tab, "fig7_marg_intrinsic")
dir.create(work, showWarnings = FALSE)
wr <- function(x, f) write.table(x, file.path(work, f), sep = ",",
                                 row.names = FALSE, col.names = FALSE)
wr(1 - sims$S_sub, "M_raw.csv")
wr(pmax(0.95 - sims$S_sub, 0) / 0.95, "M_clip.csv")
wr(CA / max(CA), "CA.csv"); wr(CB / max(CB), "CB.csv")
wr(wA, "wA.csv"); wr(wB, "wB.csv")
python <- Sys.getenv("TN_PYTHON", "python3")
st <- system2(python, c(file.path(fig_root, "python", "fugw_sweep.py"), work),
              stdout = TRUE, stderr = TRUE)
cat("sweep:", tail(st, 1), "\n")

d <- fugw_load_sweep(work, qn, ln)
d$scheme <- "intrinsic"
write.csv(d, file.path(dir_tab, "fig7_intrinsic_perquery.csv"), row.names = FALSE)

singletons <- qn[vapply(qn, function(q) length(truth_leaves[[q]]) == 1, logical(1))]
summarize <- function(d) do.call(rbind, lapply(split(d, d$setting), function(x)
  data.frame(calibration = x$calibration[1], rho = x$rho[1], alpha = x$alpha[1],
             model = x$model[1], argmax = sum(x$argmax_correct),
             confident = sum(x$category == "confident_correct"),
             cross_family = sum(x$category == "cross_family_failure"),
             singleton_argmax = sum(x$argmax_correct[x$query %in% singletons]),
             singleton_med_ratio = round(median(x$eff_ratio[x$query %in% singletons],
                                                na.rm = TRUE), 2))))
si <- summarize(d)
write.csv(si, file.path(dir_tab, "fig7_intrinsic_summary.csv"), row.names = FALSE)

f <- si[si$model == "fugw", ]
cat(sprintf("\nintrinsic (both sides from tree alone), across FUGW settings:\n  argmax %d [%d-%d] | confident %d [%d-%d] | singleton argmax (of %d) %d [%d-%d] | singleton eff-ratio %.2f\n",
            median(f$argmax), min(f$argmax), max(f$argmax),
            median(f$confident), min(f$confident), max(f$confident),
            length(singletons), median(f$singleton_argmax),
            min(f$singleton_argmax), max(f$singleton_argmax),
            median(f$singleton_med_ratio)))
cat("\nat the fig6 reference setting (raw, rho=0.3, alpha=0.9):\n")
print(si[si$calibration == "raw" & si$rho == 0.3 & si$alpha == 0.9, ],
      row.names = FALSE)
cat("\nfor comparison (analysis/14): uniform median argmax 14 [10-21]; oracle-level 23 [21-23]\n")
