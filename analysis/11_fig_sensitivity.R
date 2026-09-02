# Figure 5: threshold sensitivity around the FROZEN settings
# (margin = 0.01, vote_override = 0.90). Sensitivity analysis ONLY — the
# frozen values are outlined, no new thresholds are selected. Axis-aligned
# sweep (each threshold varied with the other held at its frozen value).
#
# Run: Rscript analysis/11_fig_sensitivity.R   (slow: ~9 full map re-runs)

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "fig_common.R"))
suppressPackageStartupMessages(library(parallel))

margins   <- c(0.0025, 0.005, 0.01, 0.02, 0.04)
overrides <- c(0.80, 0.85, 0.90, 0.95, 1.50)   # 1.50 = override disabled
grid <- unique(rbind(data.frame(margin = margins, override = 0.90),
                     data.frame(margin = 0.01, override = overrides)))

run_setting <- function(k) {
  m <- grid$margin[k]; v <- grid$override[k]
  set.seed(7)
  fwd <- tn_baseline_map(meas$cache_a, cells_a$subclass, tree_b, sims$S_sub,
                         margin = m, vote_override = v)
  fwd$outcome <- mapply(classify_fwd, fwd$selected, fwd$query, fwd$relation)
  set.seed(7)
  rev <- tn_baseline_map(cache_b_sub, cells_b$cluster, tree_a, t(sims$S_sub),
                         margin = m, vote_override = v)
  rev$outcome <- mapply(classify_rev, rev$selected, rev$query, rev$relation)
  data.frame(margin = m, override = v,
             fwd_exact = sum(fwd$outcome == "exact"), fwd_n = nrow(fwd),
             rev_exact = sum(rev$outcome == "exact"), rev_n = nrow(rev),
             fwd_wrong = sum(fwd$outcome == "wrong_branch"),
             rev_wrong = sum(rev$outcome == "wrong_branch"))
}

res <- do.call(rbind, mclapply(seq_len(nrow(grid)), run_setting, mc.cores = 5))
write.csv(res, file.path(dir_tab, "fig5_sensitivity.csv"), row.names = FALSE)
print(res, row.names = FALSE)

long <- rbind(
  transform(res, sweep = "sibling margin (override fixed 0.90)", x = margin)[res$override == 0.90, ],
  transform(res, sweep = "vote override (margin fixed 0.01)", x = override)[res$margin == 0.01, ])
long$override_lbl <- ifelse(long$x == 1.5 & grepl("override", long$sweep), "off", format(long$x))
frozen <- data.frame(sweep = c("sibling margin (override fixed 0.90)",
                               "vote override (margin fixed 0.01)"),
                     x = c(0.01, 0.90))
pl <- rbind(
  data.frame(long[c("sweep", "x")], metric = "forward exact (of 23)", value = long$fwd_exact),
  data.frame(long[c("sweep", "x")], metric = "reverse exact (of 103)", value = long$rev_exact),
  data.frame(long[c("sweep", "x")], metric = "wrong-branch (fwd+rev)",
             value = long$fwd_wrong + long$rev_wrong))

p <- ggplot(pl, aes(x, value)) +
  geom_vline(data = frozen, aes(xintercept = x), linetype = 2, color = "#d7191c") +
  geom_line(color = "grey40") + geom_point(size = 2) +
  geom_text(aes(label = value), vjust = -0.8, size = 3) +
  facet_grid(metric ~ sweep, scales = "free") +
  scale_x_continuous(trans = "log10") +
  labs(title = "Figure 5 — threshold sensitivity around the frozen settings (no retuning)",
       subtitle = "dashed red = frozen value (margin 0.01, override 0.90); override 1.5 = disabled; log-x",
       x = NULL, y = NULL) +
  theme_bw(base_size = 11)
save_fig(p, "fig5_sensitivity", 9, 7)
cat("fig5 done\n")
