# Final prespecified FUGW comparison before closing (NOTES.md item 13):
# does the singleton-family smearing trace to the uniform-per-population
# marginals rather than to FUGW itself?
#
# Capacity arithmetic: under uniform marginals each source subclass carries
# 1/23 of source mass while a singleton target family may hold only 1/103
# without violating its marginal — a 103/23 = 4.48x mismatch, matching the
# observed singleton effective-leaf ratios (4.2x / 3.6x / 4.8x).
#
# Three marginal schemes, identical sweep grid, NO additional tuning:
#   uniform    — 1/23 per source subclass, 1/103 per target leaf (fig6)
#   abundance  — donor-balanced cell-abundance proportions on both sides
#   hierarchy  — 1/23 per source subclass; each target FAMILY gets equal
#                total mass, split evenly among its own leaves (uses only
#                B's own taxonomy, not the mapping truth)
#
# Prespecified read-out: singleton-family categories and effective-leaf
# ratios, plus overall argmax/thresholded accuracy — reported at every
# setting and at the fig6 reference setting (raw, rho=0.3, alpha=0.9, which
# was chosen under UNIFORM marginals, biasing this comparison against the
# new schemes, not for them).
#
# Run: TN_PYTHON=<python-with-POT> Rscript analysis/14_fugw_marginals.R

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "fig_common.R"))
source(file.path(fig_root, "analysis", "fugw_diag.R"))

qn <- rownames(sims$S_sub); ln <- colnames(sims$S_sub)

## marginal schemes ----------------------------------------------------------
donor_balanced <- function(cells, label_col, universe) {
  props <- sapply(split(cells, cells$donor_label), function(d) {
    tab <- table(factor(d[[label_col]], levels = universe))
    as.numeric(tab) / sum(tab)
  })
  w <- rowMeans(props)
  setNames(w / sum(w), universe)
}
w_uniform_A <- setNames(rep(1 / length(qn), length(qn)), qn)
w_uniform_B <- setNames(rep(1 / length(ln), length(ln)), ln)
w_abund_A <- donor_balanced(cells_a, "subclass", qn)
w_abund_B <- donor_balanced(cells_b, "cluster", ln)
fam_sizes <- table(unname(truth_parent[ln]))            # B's own subclass level
w_hier_B <- setNames(1 / (length(fam_sizes) * as.numeric(fam_sizes[truth_parent[ln]])), ln)
schemes <- list(
  uniform   = list(wA = w_uniform_A, wB = w_uniform_B),
  abundance = list(wA = w_abund_A, wB = w_abund_B),
  hierarchy = list(wA = w_uniform_A, wB = w_hier_B))

## solve the identical grid under each scheme --------------------------------
tree_a2 <- tn_tree_from_levels(unique(cells_a[, c("class", "subclass")]))
CA <- tn_leaf_path_dist(tree_a2)[qn, qn]
CB <- tn_leaf_path_dist(tree_b)[ln, ln]
M_raw <- 1 - sims$S_sub
M_clip <- pmax(0.95 - sims$S_sub, 0) / 0.95
python <- Sys.getenv("TN_PYTHON", "python3")

per_q <- list()
for (nm in names(schemes)) {
  work <- file.path(dir_tab, paste0("fig7_marg_", nm))
  dir.create(work, showWarnings = FALSE)
  wr <- function(x, f) write.table(x, file.path(work, f), sep = ",",
                                   row.names = FALSE, col.names = FALSE)
  wr(M_raw, "M_raw.csv"); wr(M_clip, "M_clip.csv")
  wr(CA / max(CA), "CA.csv"); wr(CB / max(CB), "CB.csv")
  wr(schemes[[nm]]$wA[qn], "wA.csv"); wr(schemes[[nm]]$wB[ln], "wB.csv")
  st <- system2(python, c(file.path(fig_root, "python", "fugw_sweep.py"), work),
                stdout = TRUE, stderr = TRUE)
  cat(nm, ":", tail(st, 1), "\n")
  d <- fugw_load_sweep(work, qn, ln)
  d$scheme <- nm
  per_q[[nm]] <- d
}
all_q <- do.call(rbind, per_q)
write.csv(all_q, file.path(dir_tab, "fig7_marginals_perquery.csv"), row.names = FALSE)

## read-outs ------------------------------------------------------------------
singletons <- qn[vapply(qn, function(q) length(truth_leaves[[q]]) == 1, logical(1))]
cat(sprintf("\nsingleton-target families (n=%d): %s\n", length(singletons),
            paste(sub(" (Glut|Gaba|NN)$", "", singletons), collapse = ", ")))

summ <- do.call(rbind, lapply(split(all_q, list(all_q$scheme, all_q$setting), drop = TRUE),
  function(d) data.frame(
    scheme = d$scheme[1], calibration = d$calibration[1], rho = d$rho[1],
    alpha = d$alpha[1], model = d$model[1],
    argmax = sum(d$argmax_correct),
    confident = sum(d$category == "confident_correct"),
    cross_family = sum(d$category == "cross_family_failure"),
    singleton_argmax = sum(d$argmax_correct[d$query %in% singletons]),
    singleton_med_ratio = round(median(d$eff_ratio[d$query %in% singletons],
                                       na.rm = TRUE), 2))))
write.csv(summ, file.path(dir_tab, "fig7_marginals_summary.csv"), row.names = FALSE)

fugw_only <- summ[summ$model == "fugw", ]
cat("\n=== across all FUGW settings (median [range]) ===\n")
for (nm in names(schemes)) {
  s <- fugw_only[fugw_only$scheme == nm, ]
  cat(sprintf("%-10s argmax %4.1f [%d-%d] | confident %4.1f [%d-%d] | singleton argmax (of %d) %3.1f [%d-%d] | singleton eff-ratio %.2f\n",
      nm, median(s$argmax), min(s$argmax), max(s$argmax),
      median(s$confident), min(s$confident), max(s$confident),
      length(singletons), median(s$singleton_argmax),
      min(s$singleton_argmax), max(s$singleton_argmax),
      median(s$singleton_med_ratio)))
}

cat("\n=== at the fig6 reference setting (raw, rho=0.3, alpha=0.9; chosen under uniform) ===\n")
ref <- summ[summ$calibration == "raw" & summ$rho == 0.3 & summ$alpha == 0.9, ]
print(ref[, c("scheme", "argmax", "confident", "cross_family",
              "singleton_argmax", "singleton_med_ratio")], row.names = FALSE)

cat("\n=== singleton detail at the reference setting ===\n")
refq <- all_q[all_q$calibration == "raw" & all_q$rho == 0.3 & all_q$alpha == 0.9 &
              all_q$query %in% singletons,
              c("scheme", "query", "category", "true_mass", "eff_ratio")]
refq$true_mass <- round(refq$true_mass, 3); refq$eff_ratio <- round(refq$eff_ratio, 2)
print(refq[order(refq$query, refq$scheme), ], row.names = FALSE)
