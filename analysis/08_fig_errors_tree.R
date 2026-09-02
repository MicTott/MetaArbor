# Figure 2: where errors occur in the true Allen hierarchy. Every query is
# placed at its true position in the curated tree and connected to its
# TreeNeighbor-selected position; internal-node selections stay at internal
# nodes. Forward and reverse panels; recurrent cases labeled.
#
# Run: Rscript analysis/08_fig_errors_tree.R

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "fig_common.R"))

leaves_ord <- tax_b$cluster
lx <- setNames(seq_along(leaves_ord), leaves_ord)
node_y <- function(id) ifelse(id == "root", 0,
                       ifelse(startsWith(id, "class:"), 1,
                       ifelse(startsWith(id, "subclass:"), 2,
                       ifelse(startsWith(id, "supertype:"), 3, 4))))
node_x <- vapply(tree_b$id, function(id)
  if (id == "root") mean(lx) else mean(lx[tn_leaves_under(tree_b, id)]),
  numeric(1))
coords <- data.frame(id = tree_b$id, x = node_x, y = node_y(tree_b$id))
seg <- merge(merge(tree_b[!is.na(tree_b$parent), c("id", "parent")],
                   coords, by = "id"),
             coords, by.x = "parent", by.y = "id", suffixes = c("", "_p"))

place <- function(id) {
  id[is.na(id)] <- "root"
  coords[match(id, coords$id), c("x", "y")]
}

fwd <- fwd_map
fwd$outcome <- mapply(classify_fwd, fwd$selected, fwd$query, fwd$relation)
fwd$true_id <- sub_node(fwd$query)
rev <- rev_map
rev$outcome <- mapply(classify_rev, rev$selected, rev$query, rev$relation)
rev$true_id <- rev$query                                   # cluster leaf
rev$sel_id <- ifelse(is.na(rev$selected), NA, sub_node(rev$selected))

build_panel_df <- function(map, sel_col, dir_label) {
  tp <- place(map$true_id); sp <- place(map[[sel_col]])
  data.frame(direction = dir_label, query = map$query, outcome = map$outcome,
             true_id = map$true_id, sel_id = map[[sel_col]],
             x = tp$x, y = tp$y, xend = sp$x, yend = sp$y)
}
pf <- build_panel_df(fwd, "selected", "forward: 23 subclasses -> cluster tree")
pr <- build_panel_df(rev, "sel_id", "reverse: 103 clusters -> subclasses")
pd <- rbind(pf, pr)
write.csv(pd, file.path(dir_tab, "fig2_error_topology.csv"), row.names = FALSE)

lab <- rbind(
  data.frame(direction = pr$direction[1],
             x = mean(lx[grep("L6 IT|L5 IT|L4/5 IT", names(lx))]), y = 4.45,
             text = "deep-layer IT continuum\n(all premature stops / near-misses)"),
  data.frame(direction = pr$direction[1],
             x = mean(lx[grep("COP|OPC", names(lx))]), y = 4.45,
             text = "COP -> OPC\n(lineage intermediate)"),
  data.frame(direction = pf$direction[1],
             x = mean(lx[truth_leaves[["039 Lamp5 Gaba"]]]), y = 4.45,
             text = "Lamp5: one\nsupertype deep"),
  data.frame(direction = pf$direction[1],
             x = mean(lx[truth_leaves[["301 Endo NN"]]]), y = 4.45,
             text = "Endo: stopped\nat class"))

p <- ggplot() +
  geom_segment(data = seg, aes(x = x, y = y, xend = x_p, yend = y_p),
               color = "grey82", linewidth = 0.25) +
  geom_curve(data = subset(pd, !(x == xend & y == yend)),
             aes(x = x, y = y, xend = xend, yend = yend, color = outcome),
             curvature = -0.15, linewidth = 0.5, alpha = 0.85,
             arrow = arrow(length = unit(4, "pt"))) +
  geom_point(data = pd, aes(x, y, color = outcome), size = 1.8) +
  geom_point(data = subset(pd, !is.na(sel_id) & sel_id != true_id),
             aes(xend, yend, color = outcome), shape = 5, size = 2.4,
             stroke = 0.8) +
  geom_text(data = lab, aes(x, y, label = text), size = 2.6, lineheight = 0.85,
            color = "grey25") +
  scale_color_manual(values = outcome_colors, name = "outcome") +
  scale_y_reverse(breaks = 0:4, labels = c("root", "class", "subclass",
                                           "supertype", "cluster"),
                  limits = c(4.8, -0.2)) +
  facet_wrap(~direction, ncol = 1) +
  labs(title = "Figure 2 — where errors occur in the true Allen hierarchy",
       subtitle = "dot = query at its true node; diamond + arrow = selected node when different; grey = curated tree (103 kept clusters)",
       x = "clusters in curated taxonomy order", y = NULL) +
  theme_bw(base_size = 10) +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        legend.position = "bottom")
save_fig(p, "fig2_error_topology", 12, 9)
cat("fig2 done\n")
