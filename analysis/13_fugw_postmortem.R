# FUGW post-mortem before closing it (per-query decomposition of the fig6
# sweep couplings): is the 12-14/23 ceiling underconfidence (correct family,
# mass < 0.90), expected coarse-to-fine splitting (diffuse WITHIN the right
# family), genuine cross-family transport failure, or solver regularization?
#
# Note: the fig6 sweep ran at epsilon = 0 (mm solver, no entropic term), so
# an explicit epsilon probe is added to test the regularization hypothesis.
#
# Per query and setting:
#   argmax family (threshold-free) correctness; mass inside the true family;
#   first-vs-second family margin; entropy across families; entropy within
#   the true family; effective leaves / true descendant leaves.
#
# Run: TN_PYTHON=<python-with-POT> Rscript analysis/13_fugw_postmortem.R

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "fig_common.R"))

settings <- read.csv(file.path(dir_tab, "fig6_fugw_sweep.csv"))
pis <- read.csv(file.path(dir_tab, "fig6_problem", "pi_long.csv"))
qn <- rownames(sims$S_sub); ln <- colnames(sims$S_sub)

fam_of_leaf <- unname(truth_parent[ln])   # each kept cluster's subclass
ent <- function(p) { p <- p[p > 0]; -sum(p * log(p)) }

decompose <- function(pi) {
  do.call(rbind, lapply(qn, function(q) {
    row <- pi[q, ]; tot <- sum(row)
    if (tot <= 0) return(data.frame(query = q, argmax_correct = FALSE,
      true_mass = 0, fam_margin = 0, H_family = NA, H_within = NA,
      eff_leaves = 0, n_true = length(truth_leaves[[q]]), eff_ratio = NA))
    p <- row / tot
    fam <- tapply(p, fam_of_leaf, sum)
    fam <- fam[order(-fam)]
    inb <- names(row) %in% truth_leaves[[q]]
    pw <- p[inb] / sum(p[inb])
    data.frame(query = q,
               argmax_correct = names(fam)[1] == q,
               true_mass = sum(p[inb]),
               fam_margin = fam[1] - ifelse(length(fam) > 1, fam[2], 0),
               H_family = ent(fam),
               H_within = if (sum(p[inb]) > 0) ent(pw) else NA,
               eff_leaves = exp(ent(p)),
               n_true = length(truth_leaves[[q]]),
               eff_ratio = exp(ent(p)) / length(truth_leaves[[q]]))
  }))
}
classify_q <- function(d)
  ifelse(!d$argmax_correct, "cross_family_failure",
  ifelse(d$true_mass >= 0.9, "confident_correct",
  ifelse(d$true_mass >= 0.5, "underconfident_correct", "diffuse_correct")))

per_q <- do.call(rbind, lapply(settings$setting, function(s) {
  sub <- pis[pis$setting == s, ]
  pi <- matrix(0, length(qn), length(ln), dimnames = list(qn, ln))
  pi[cbind(sub$i + 1, sub$j + 1)] <- sub$value
  d <- decompose(pi)
  d$category <- classify_q(d)
  cbind(settings[settings$setting == s,
                 c("setting", "calibration", "rho", "alpha", "model", "exact")], d,
        row.names = NULL)
}))
write.csv(per_q, file.path(dir_tab, "fig7_fugw_postmortem.csv"), row.names = FALSE)

cat("=== per-setting summary: threshold-free vs 0.90-threshold ===\n")
summ <- do.call(rbind, lapply(split(per_q, per_q$setting), function(d) {
  data.frame(calibration = d$calibration[1], rho = d$rho[1], alpha = d$alpha[1],
             model = d$model[1], exact_thresh090 = d$exact[1],
             argmax_correct = sum(d$argmax_correct),
             confident = sum(d$category == "confident_correct"),
             underconf = sum(d$category == "underconfident_correct"),
             diffuse = sum(d$category == "diffuse_correct"),
             cross_family = sum(d$category == "cross_family_failure"),
             med_true_mass = round(median(d$true_mass[d$argmax_correct]), 3),
             med_eff_ratio = round(median(d$eff_ratio, na.rm = TRUE), 2))
}))
summ <- summ[order(summ$calibration, summ$rho, summ$alpha), ]
print(summ, row.names = FALSE)
write.csv(summ, file.path(dir_tab, "fig7_postmortem_summary.csv"), row.names = FALSE)

## epsilon probe: does entropic regularization explain diffuseness? ----------
## representative setting raw / rho=1 / alpha=0.5; sweep baseline was eps=0
cat("\n=== epsilon probe (raw cost, rho = 1, alpha = 0.5) ===\n")
tree_a2 <- tn_tree_from_levels(unique(cells_a[, c("class", "subclass")]))
CA <- tn_leaf_path_dist(tree_a2)[qn, qn]
CB <- tn_leaf_path_dist(tree_b)[ln, ln]
M <- 1 - sims$S_sub
eps_rows <- lapply(c(0, 1e-3, 1e-2), function(eps) {
  fit <- tn_fugw(M, CA, CB, alpha = 0.5, rho = 1, epsilon = eps,
                 script = file.path(fig_root, "python", "fugw_run.py"))
  d <- decompose(fit$pi)
  d$category <- classify_q(d)
  data.frame(epsilon = eps, argmax_correct = sum(d$argmax_correct),
             confident = sum(d$category == "confident_correct"),
             med_true_mass = round(median(d$true_mass[d$argmax_correct]), 3),
             med_eff_leaves = round(median(d$eff_leaves), 1),
             med_eff_ratio = round(median(d$eff_ratio, na.rm = TRUE), 2))
})
eps_tab <- do.call(rbind, eps_rows)
print(eps_tab, row.names = FALSE)
write.csv(eps_tab, file.path(dir_tab, "fig7_epsilon_probe.csv"), row.names = FALSE)

## representative per-query detail at the sweep's best setting ---------------
best <- settings$setting[which.max(settings$exact)]
cat(sprintf("\n=== per-query detail at best sweep setting (%s, rho=%s, alpha=%s) ===\n",
            settings$calibration[settings$setting == best],
            settings$rho[settings$setting == best],
            settings$alpha[settings$setting == best]))
d <- per_q[per_q$setting == best,
           c("query", "category", "true_mass", "fam_margin", "H_family",
             "H_within", "eff_leaves", "n_true", "eff_ratio")]
d[3:9] <- round(d[3:9], 3)
print(d, row.names = FALSE)
