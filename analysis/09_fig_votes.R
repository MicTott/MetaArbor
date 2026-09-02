# Figure 3: vote structure across all queries. Hierarchy-ordered heatmaps of
# per-cell vote fractions (each query cell votes for its argmax candidate
# leaf). Family blocks, saturated siblings, and cross-family scatter become
# visible; the true family and selected node are marked per row.
#
# Run: Rscript analysis/09_fig_votes.R

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "fig_common.R"))

vote_matrix <- function(cache, labels) {
  ms <- sweep(cache$V, 2, cache$leaf_sizes, "/")
  top <- cache$leaves[max.col(ms, ties.method = "first")]
  t(vapply(sort(unique(labels)), function(q)
    as.numeric(table(factor(top[labels == q], levels = cache$leaves))) /
      sum(labels == q), numeric(length(cache$leaves)))) |>
    (\(m) { colnames(m) <- cache$leaves; m })()
}

cl_ord <- tax_b$cluster
sub_ord <- unique(tax_b$subclass)
sub_ord <- sub_ord[sub_ord %in% subclasses]

## forward: 23 subclass queries voting over 103 B clusters -------------------
Vf <- vote_matrix(meas$cache_a, cells_a$subclass)[sub_ord, cl_ord]
fwd <- fwd_map
fwd$outcome <- mapply(classify_fwd, fwd$selected, fwd$query, fwd$relation)

## reverse: 103 cluster queries voting over 23 A subclass leaves -------------
Vr <- vote_matrix(cache_b_sub, cells_b$cluster)[cl_ord, sub_ord]
rev <- rev_map
rev$outcome <- mapply(classify_rev, rev$selected, rev$query, rev$relation)

melt <- function(m) data.frame(row = rep(rownames(m), ncol(m)),
                               col = rep(colnames(m), each = nrow(m)),
                               value = as.vector(m))
df_f <- melt(Vf); df_r <- melt(Vr)
write.csv(df_f, file.path(dir_tab, "fig3_votes_forward.csv"), row.names = FALSE)
write.csv(df_r, file.path(dir_tab, "fig3_votes_reverse.csv"), row.names = FALSE)

# per-row marks: true block and selected block as x-ranges in column order
rng <- function(cols, universe) {
  ix <- match(intersect(cols, universe), universe)
  if (!length(ix)) return(c(NA, NA))
  c(min(ix) - 0.5, max(ix) + 0.5)
}
marks_f <- do.call(rbind, lapply(seq_along(sub_ord), function(k) {
  q <- sub_ord[k]
  tr <- rng(truth_leaves[[q]], cl_ord)
  sel <- fwd$selected[fwd$query == q]
  sr <- if (is.na(sel)) c(NA, NA) else rng(tn_leaves_under(tree_b, sel), cl_ord)
  data.frame(row = k, true_lo = tr[1], true_hi = tr[2],
             sel_lo = sr[1], sel_hi = sr[2],
             outcome = fwd$outcome[fwd$query == q])
}))
marks_r <- do.call(rbind, lapply(seq_along(cl_ord), function(k) {
  q <- cl_ord[k]
  tr <- rng(unname(truth_parent[q]), sub_ord)
  sel <- rev$selected[rev$query == q]
  sr <- if (is.na(sel)) c(NA, NA) else rng(sel, sub_ord)
  data.frame(row = k, true_lo = tr[1], true_hi = tr[2],
             sel_lo = sr[1], sel_hi = sr[2],
             outcome = rev$outcome[rev$query == q])
}))

heat <- function(df, rows, cols, marks, title, xlab, ylab) {
  df$ri <- match(df$row, rows); df$ci <- match(df$col, cols)
  class_breaks <- cumsum(rle(tax_b$class[match(cols, tax_b$cluster)])$lengths) + 0.5
  if (all(is.na(class_breaks))) class_breaks <- NULL
  g <- ggplot(df, aes(ci, ri, fill = value)) +
    geom_tile() +
    geom_rect(data = marks, inherit.aes = FALSE, na.rm = TRUE,
              aes(xmin = true_lo, xmax = true_hi,
                  ymin = row - 0.5, ymax = row + 0.5),
              fill = NA, color = "#1a9850", linewidth = 0.35) +
    geom_rect(data = subset(marks, !is.na(sel_lo)), inherit.aes = FALSE,
              aes(xmin = sel_lo, xmax = sel_hi,
                  ymin = row - 0.5, ymax = row + 0.5),
              fill = NA, color = "#2b8cbe", linewidth = 0.35, linetype = 2) +
    geom_point(data = marks, inherit.aes = FALSE,
               aes(x = -1.6, y = row, color = outcome), size = 1.4) +
    scale_fill_gradient(low = "white", high = "grey10", limits = c(0, 1),
                        name = "vote fraction") +
    scale_color_manual(values = outcome_colors, name = "outcome") +
    scale_y_reverse(breaks = seq_along(rows),
                    labels = sub(" (Glut|Gaba|NN)$", "", rows), expand = c(0, 0)) +
    labs(title = title, x = xlab, y = ylab) +
    theme_bw(base_size = 7) +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
          panel.grid = element_blank(), legend.position = "right")
  if (!is.null(class_breaks))
    g <- g + geom_vline(xintercept = class_breaks, color = "grey60",
                        linewidth = 0.2)
  g
}
p1 <- heat(df_f, sub_ord, cl_ord, marks_f,
           "forward: subclass queries (rows) voting over 103 clusters (columns, taxonomy order)",
           "B clusters (class boundaries in grey)", NULL)
p2 <- heat(df_r, cl_ord, sub_ord, marks_r,
           "reverse: cluster queries (rows) voting over 23 subclasses (columns)",
           "A subclasses", NULL) +
  theme(axis.text.y = element_text(size = 3))
p <- p1 / p2 + plot_layout(heights = c(1, 2.6)) +
  plot_annotation(
    title = "Figure 3 — vote structure across all queries",
    subtitle = "green solid = true family block; blue dashed = selected node; left dots = outcome")
save_fig(p, "fig3_vote_structure", 11, 13)
cat("fig3 done\n")
