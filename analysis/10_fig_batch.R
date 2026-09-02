# Figure 4: summary across the three batch conditions (random half-split,
# donor-held-out, 10Xv2<->10Xv3). Performance as points/lines with raw
# numerators/denominators printed; failure composition as stacked bars.
#
# Run: Rscript analysis/10_fig_batch.R

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "fig_common.R"))

summ <- read.csv(file.path(dir_res, "batch_conditions_summary.csv"))
summ$condition <- factor(summ$condition, levels = c("random", "donor", "platform"),
                         labels = c("random half-split", "donor-held-out",
                                    "10Xv2 vs 10Xv3"))

frac <- function(s) {
  p <- as.numeric(strsplit(s, "/")[[1]])
  c(p[1] / p[2], p[1], p[2])
}
perf <- do.call(rbind, lapply(seq_len(nrow(summ)), function(i) {
  fw <- frac(summ$fwd_exact[i]); rv <- frac(summ$rev_parent[i])
  data.frame(condition = summ$condition[i],
             metric = c("self-RBH recovery", "median self-AUROC",
                        "forward family accuracy", "reverse parent recovery",
                        "median sibling margin"),
             value = c(summ$self_rbh[i] / summ$n_clusters[i],
                       summ$med_self_auroc[i], fw[1], rv[1], summ$sib_margin[i]),
             label = c(sprintf("%d/%d", summ$self_rbh[i], summ$n_clusters[i]),
                       sprintf("%.3f", summ$med_self_auroc[i]),
                       summ$fwd_exact[i], summ$rev_parent[i],
                       sprintf("%.3f", summ$sib_margin[i])))
}))
write.csv(perf, file.path(dir_tab, "fig4_performance.csv"), row.names = FALSE)

## outcome composition, both directions, classified per condition ------------
comp_rows <- list()
for (cond in c("random", "donor", "platform")) {
  if (cond == "platform") {
    fw <- fwd_map; rv <- rev_map
  } else {
    fw <- read.csv(file.path(dir_res, sprintf("cond_%s_forward.csv", cond)))
    rv <- read.csv(file.path(dir_res, sprintf("cond_%s_reverse.csv", cond)))
  }
  # per-condition truth: reuse global maps (same taxonomy; clusters absent
  # from a condition simply don't appear as queries)
  fw$outcome <- mapply(classify_fwd, fw$selected, fw$query, fw$relation)
  rv$outcome <- mapply(classify_rev, rv$selected, rv$query, rv$relation)
  comp_rows[[length(comp_rows) + 1]] <- data.frame(
    condition = cond, direction = "forward",
    as.data.frame(table(outcome = fw$outcome)))
  comp_rows[[length(comp_rows) + 1]] <- data.frame(
    condition = cond, direction = "reverse",
    as.data.frame(table(outcome = rv$outcome)))
}
comp <- do.call(rbind, comp_rows)
comp$condition <- factor(comp$condition, levels = c("random", "donor", "platform"),
                         labels = levels(summ$condition))
comp$outcome <- factor(comp$outcome,
                       levels = c("exact", "adjacent_same_class",
                                  "premature_stop", "wrong_branch", "unmatched"))
write.csv(comp, file.path(dir_tab, "fig4_composition.csv"), row.names = FALSE)

p1 <- ggplot(perf, aes(condition, value, group = metric)) +
  geom_line(color = "grey40") + geom_point(size = 2, color = "#2b8cbe") +
  geom_text(aes(label = label), vjust = -0.9, size = 2.8) +
  facet_wrap(~metric, scales = "free_y", nrow = 1) +
  scale_y_continuous(expand = expansion(mult = c(0.08, 0.30))) +
  labs(x = NULL, y = NULL,
       title = "Figure 4 — three batch conditions, frozen estimator",
       subtitle = "labels show raw values (numerator/denominator where applicable)") +
  theme_bw(base_size = 9) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

p2 <- ggplot(comp, aes(condition, Freq, fill = outcome)) +
  geom_col(width = 0.65, color = "white", linewidth = 0.2) +
  geom_text(aes(label = ifelse(Freq > 0, Freq, "")),
            position = position_stack(vjust = 0.5), size = 2.6, color = "white") +
  facet_wrap(~direction, scales = "free_y") +
  scale_fill_manual(values = outcome_colors, name = "outcome") +
  labs(x = NULL, y = "queries",
       subtitle = "outcome composition (counts printed); wrong-branch and unmatched are zero everywhere") +
  theme_bw(base_size = 9) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1),
        legend.position = "bottom")

p <- p1 / p2 + plot_layout(heights = c(1, 1.4))
save_fig(p, "fig4_batch_conditions", 11, 7.5)
cat("fig4 done\n")
