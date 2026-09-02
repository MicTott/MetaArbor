# Shared FUGW coupling diagnostics (used by analysis/13 and /14).
# Requires fig_common.R already sourced (truth_parent, truth_leaves, sims).

fugw_ent <- function(p) { p <- p[p > 0]; -sum(p * log(p)) }

#' Per-query decomposition of a coupling matrix (rows = source subclasses,
#' cols = target kept clusters): family masses, margins, entropies,
#' effective-leaf ratios. `tp`/`tl` default to the platform-condition truth
#' from fig_common; pass condition-specific ones for other splits.
fugw_decompose <- function(pi, qn = rownames(pi), ln = colnames(pi),
                           tp = truth_parent, tl = truth_leaves) {
  truth_parent <- tp; truth_leaves <- tl
  fam_of_leaf <- unname(truth_parent[ln])
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
               H_family = fugw_ent(fam),
               H_within = if (sum(p[inb]) > 0) fugw_ent(pw) else NA,
               eff_leaves = exp(fugw_ent(p)),
               n_true = length(truth_leaves[[q]]),
               eff_ratio = exp(fugw_ent(p)) / length(truth_leaves[[q]]))
  }))
}

fugw_classify <- function(d)
  ifelse(!d$argmax_correct, "cross_family_failure",
  ifelse(d$true_mass >= 0.9, "confident_correct",
  ifelse(d$true_mass >= 0.5, "underconfident_correct", "diffuse_correct")))

#' Load a sweep output dir into a per-(setting, query) decomposition table.
fugw_load_sweep <- function(dir, qn, ln) {
  settings <- read.csv(file.path(dir, "settings.csv"))
  pis <- read.csv(file.path(dir, "pi_long.csv"))
  do.call(rbind, lapply(settings$setting, function(s) {
    sub <- pis[pis$setting == s, ]
    pi <- matrix(0, length(qn), length(ln), dimnames = list(qn, ln))
    pi[cbind(sub$i + 1, sub$j + 1)] <- sub$value
    d <- fugw_decompose(pi, qn, ln)
    d$category <- fugw_classify(d)
    cbind(settings[settings$setting == s,
                   c("setting", "calibration", "rho", "alpha", "model")], d,
          row.names = NULL)
  }))
}
