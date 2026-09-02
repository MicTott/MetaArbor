# Figure 1: how TreeNeighbor walks a tree — six representative queries, every
# visited split shown with child vote fractions, sibling AUROCs, the margin
# test, the vote override, and the resulting decision.
#
# Also proves the trace instrumentation is recording-only: the full forward
# and reverse maps are re-run and asserted identical to the saved benchmark.
#
# Run: Rscript analysis/07_fig_walks.R

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "fig_common.R"))

## trace non-interference check --------------------------------------------
set.seed(7)
fwd_chk <- tn_baseline_map(meas$cache_a, cells_a$subclass, tree_b, sims$S_sub)
stopifnot(identical(fwd_chk$selected, fwd_map$selected),
          identical(fwd_chk$relation, fwd_map$relation))
set.seed(7)
rev_chk <- tn_baseline_map(cache_b_sub, cells_b$cluster, tree_a, t(sims$S_sub))
stopifnot(identical(rev_chk$selected, rev_map$selected),
          identical(rev_chk$relation, rev_map$relation))
cat("identity check passed: instrumented walk reproduces the saved benchmark\n")

## six representative walks -------------------------------------------------
first_of <- function(map, cond) map$query[which(cond)[1]]
picks <- list(
  list(q = first_of(fwd_map, fwd_map$relation == "leaf" & fwd_map$correct),
       dir = "fwd", label = "correct leaf selection"),
  list(q = {f <- fwd_map$relation == "family" & fwd_map$correct
            if ("004 L6 IT CTX Glut" %in% fwd_map$query[f]) "004 L6 IT CTX Glut"
            else first_of(fwd_map, f)},
       dir = "fwd", label = "correct family-level stop"),
  list(q = first_of(rev_map, !rev_map$correct & is.na(rev_map$selected) &
                    rev_map$relation == "discordant" &
                    grepl("L6 IT", rev_map$query)),
       dir = "rev", label = "deep-layer IT premature stop"),
  list(q = first_of(rev_map, !rev_map$correct & !is.na(rev_map$selected)),
       dir = "rev", label = "adjacent same-class near-miss"),
  list(q = "039 Lamp5 Gaba", dir = "fwd", label = "Lamp5 forward miss (one level deep)"),
  list(q = "301 Endo NN", dir = "fwd", label = "Endo forward miss (one level shallow)"))

walk_one <- function(q, dir) {
  set.seed(7)
  if (dir == "fwd")
    tn_select_node(meas$cache_a, cells_a$subclass, q, tree_b, trace = TRUE)
  else
    tn_select_node(cache_b_sub, cells_b$cluster, q, tree_a, trace = TRUE)
}
on_true_path <- function(child, q, dir, tree) {
  true_id <- if (dir == "fwd") sub_node(q) else unname(truth_parent[q])
  if (!true_id %in% tree$id) {  # chain-collapsed subclass: use its leaves
    return(any(tn_leaves_under(tree, child) %in% truth_leaves[[q]]))
  }
  child == true_id || child %in% ancestors(tree, true_id) ||
    true_id %in% tn_subtree_ids(tree, child)
}
tn_subtree_ids <- function(tree, id) {
  out <- id; stack <- tn_children(tree, id)
  while (length(stack)) {
    x <- stack[[1]]; stack <- stack[-1]
    out <- c(out, x); stack <- c(stack, tn_children(tree, x))
  }
  out
}

abbrev <- function(x) {
  x <- sub("^(class|subclass|supertype):", "", x)
  x <- gsub(" (Glut|Gaba|NN)$", "", x)
  ifelse(nchar(x) > 16, paste0(substr(x, 1, 15), "~"), x)
}

all_traces <- list()
pages <- list()
for (k in seq_along(picks)) {
  p <- picks[[k]]
  w <- walk_one(p$q, p$dir)
  tree_used <- if (p$dir == "fwd") tree_b else tree_a
  tr <- w$trace
  saved <- if (p$dir == "fwd") fwd_map else rev_map
  sel_saved <- saved$selected[saved$query == p$q]
  stopifnot(identical(w$selected, sel_saved))   # traced walk == benchmark
  tr$true_path <- vapply(seq_len(nrow(tr)), function(r)
    isTRUE(tryCatch(on_true_path(tr$child[r], p$q, p$dir, tree_used),
                    error = function(e) FALSE)), logical(1))
  tr$case <- p$label
  all_traces[[k]] <- tr
  splits <- unique(tr$split_at)
  panels <- lapply(seq_along(splits), function(si) {
    d <- tr[tr$split_at == splits[si], ]
    d$child_ab <- factor(abbrev(d$child), levels = abbrev(d$child)[order(d$child)])
    r1 <- d[1, ]
    verdict <- if (r1$override)
      sprintf("override: top vote %.2f >= 0.90 -> DESCEND", max(d$vote))
    else if (r1$decision == "descend")
      sprintf("sibling dAUROC lower bound %.3f > margin 0.01 -> DESCEND", r1$sib_lo)
    else if (!is.na(r1$par_lo) && r1$par_lo > 0)
      sprintf("parent significantly better (lower bound %.3f) -> STOP", r1$par_lo)
    else
      sprintf("sibling dAUROC lower bound %.3f <= margin 0.01 -> STOP", r1$sib_lo)
    ggplot(d, aes(child_ab, vote)) +
      geom_col(aes(fill = is_best), width = 0.7, show.legend = FALSE) +
      geom_col(data = d[d$true_path, ], fill = NA, color = "#1a9850",
               linewidth = 1.1, width = 0.7) +
      geom_text(aes(label = sprintf("%.3f", child_auroc)), angle = 90,
                hjust = -0.05, size = 2.6, color = "grey30") +
      scale_fill_manual(values = c(`TRUE` = "#2b8cbe", `FALSE` = "grey75")) +
      scale_y_continuous(limits = c(0, max(d$vote) * 1.45)) +
      labs(title = sprintf("split at %s — %s", abbrev(splits[si]), verdict),
           x = NULL, y = "vote fraction") +
      theme_bw(base_size = 9) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(size = 9))
  })
  head_txt <- sprintf(
    "%s\nquery: %s (%s)  |  true: %s  |  selected: %s  |  numbers over bars = child one-vs-all AUROC; green outline = true path; blue = entered child",
    p$label, p$q, if (p$dir == "fwd") "coarse to tree" else "cluster to subclasses",
    if (p$dir == "fwd") sub_node(p$q) else unname(truth_parent[p$q]),
    ifelse(is.na(w$selected), "none (stopped at root)", w$selected))
  pages[[k]] <- wrap_plots(panels, ncol = 1) +
    plot_annotation(title = head_txt,
                    theme = theme(plot.title = element_text(size = 9)))
}
write.csv(do.call(rbind, all_traces),
          file.path(dir_tab, "fig1_walks.csv"), row.names = FALSE)
save_fig(NULL, "fig1_walks", 9, 10, pages = pages)
for (k in seq_along(pages)) {
  png(file.path(dir_fig, sprintf("fig1_walk_%d.png", k)), 990, 1100, res = 110)
  print(pages[[k]])
  dev.off()
}
cat("fig1 done\n")
