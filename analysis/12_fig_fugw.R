# Figure 6: does the GW tree-structure term improve or degrade the same
# molecular-only transport model? Recovery vs molecular weight fraction, with
# tree-only GW, the fusion sweep, molecular-only UOT (identical cost /
# marginals / regularization), and TreeNeighbor as the reference line.
# Facets = bounded rho and cost-calibration sensitivity (no post-hoc best).
#
# Run: TN_PYTHON=<python-with-POT> Rscript analysis/12_fig_fugw.R

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "fig_common.R"))

work <- file.path(dir_tab, "fig6_problem")
dir.create(work, showWarnings = FALSE)
tree_a2 <- tn_tree_from_levels(unique(cells_a[, c("class", "subclass")]))
CA <- tn_leaf_path_dist(tree_a2)[rownames(sims$S_sub), rownames(sims$S_sub)]
CB <- tn_leaf_path_dist(tree_b)[colnames(sims$S_sub), colnames(sims$S_sub)]
M_raw <- 1 - sims$S_sub
M_clip <- pmax(0.95 - sims$S_sub, 0) / 0.95  # matrix first: pmax keeps its dims
wr <- function(x, f) write.table(x, file.path(work, f), sep = ",",
                                 row.names = FALSE, col.names = FALSE)
wr(M_raw, "M_raw.csv"); wr(M_clip, "M_clip.csv")
wr(CA / max(CA), "CA.csv"); wr(CB / max(CB), "CB.csv")
wr(rep(1 / nrow(M_raw), nrow(M_raw)), "wA.csv")
wr(rep(1 / ncol(M_raw), ncol(M_raw)), "wB.csv")

python <- Sys.getenv("TN_PYTHON", "python3")
status <- system2(python, c(file.path(fig_root, "python", "fugw_sweep.py"), work),
                  stdout = TRUE, stderr = TRUE)
cat(tail(status, 2), sep = "\n")

settings <- read.csv(file.path(work, "settings.csv"))
pis <- read.csv(file.path(work, "pi_long.csv"))

acc <- vapply(settings$setting, function(s) {
  sub <- pis[pis$setting == s, ]
  pi <- matrix(0, nrow(M_raw), ncol(M_raw),
               dimnames = dimnames(M_raw))
  pi[cbind(sub$i + 1, sub$j + 1)] <- sub$value
  assign <- tn_fugw_assign(pi, tree_b, q = 0.9)
  sum(vapply(seq_len(nrow(assign)), function(k) {
    if (is.na(assign$node[k]) || assign$node[k] == "root") return(FALSE)
    identical(sort(tn_leaves_under(tree_b, assign$node[k])),
              truth_leaves[[assign$query[k]]])
  }, logical(1)))
}, numeric(1))
settings$exact <- acc
tn_ref <- sum(fwd_map$correct)
write.csv(settings, file.path(dir_tab, "fig6_fugw_sweep.csv"), row.names = FALSE)
print(settings[, c("calibration", "rho", "alpha", "model", "exact",
                   "median_row_entropy")], row.names = FALSE)

settings$model_lbl <- c(gw_only = "tree-only GW", fugw = "FUGW (fusion sweep)",
                        uot_molecular = "molecular-only UOT")[settings$model]
p1 <- ggplot(settings, aes(alpha, exact, color = model_lbl, shape = model_lbl)) +
  geom_hline(yintercept = tn_ref, linetype = 2, color = "#d7191c") +
  annotate("text", x = 0.05, y = tn_ref + 0.6, hjust = 0, size = 3,
           color = "#d7191c",
           label = sprintf("TreeNeighbor direct selection: %d/23", tn_ref)) +
  geom_line(data = subset(settings, model != "uot_molecular"), linewidth = 0.4) +
  geom_point(size = 2.4) +
  facet_grid(calibration ~ rho,
             labeller = labeller(rho = function(x) paste0("rho = ", x),
                                 calibration = c(raw = "cost: 1 - S",
                                                 clip = "cost: clipped (0.95 - S)+"))) +
  scale_color_manual(values = c("tree-only GW" = "#984ea3",
                                "FUGW (fusion sweep)" = "#2b8cbe",
                                "molecular-only UOT" = "#1a9850")) +
  labs(title = "Figure 6 — what the GW tree term contributes",
       subtitle = "exact subclass recovery (of 23) vs molecular fraction of the fused cost (design weight alpha); no post-hoc setting selection",
       x = "molecular fraction of fused cost (design weight alpha; 0 = tree-only, 1 = molecular-only UOT)",
       y = "exact recoveries (of 23)", color = NULL, shape = NULL) +
  theme_bw(base_size = 10) + theme(legend.position = "bottom")

diag_long <- rbind(
  data.frame(settings[c("calibration", "rho", "alpha", "model_lbl")],
             metric = "median row entropy (nats)", value = settings$median_row_entropy),
  data.frame(settings[c("calibration", "rho", "alpha", "model_lbl")],
             metric = "molecular component", value = settings$lin_component),
  data.frame(settings[c("calibration", "rho", "alpha", "model_lbl")],
             metric = "structural (GW) component", value = settings$gw_component))
p2 <- ggplot(diag_long, aes(alpha, value, color = model_lbl, shape = model_lbl)) +
  geom_line(data = subset(diag_long, model_lbl != "molecular-only UOT"), linewidth = 0.4) +
  geom_point(size = 2) +
  facet_grid(metric ~ calibration + rho, scales = "free_y") +
  scale_color_manual(values = c("tree-only GW" = "#984ea3",
                                "FUGW (fusion sweep)" = "#2b8cbe",
                                "molecular-only UOT" = "#1a9850")) +
  labs(subtitle = "objective components and coupling entropy across the same sweep",
       x = "molecular fraction of fused cost", y = NULL,
       color = NULL, shape = NULL) +
  theme_bw(base_size = 9) + theme(legend.position = "bottom")

save_fig(p1, "fig6_fugw_contribution", 10, 6.5)
save_fig(p2, "fig6_fugw_diagnostics", 12, 7)
cat("fig6 done\n")
