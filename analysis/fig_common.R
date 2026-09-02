# Shared loader for the benchmark figure set (analysis/07-12).
# Everything comes from cached measurements and saved results — the estimator
# is frozen and is not re-tuned anywhere in the figure scripts.
suppressPackageStartupMessages({
  library(Matrix); library(ggplot2); library(patchwork)
})

fig_root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), ".."))
for (f in list.files(file.path(fig_root, "R"), full.names = TRUE)) source(f)
dir_res <- file.path(fig_root, "results")
dir_fig <- file.path(fig_root, "figures")
dir_tab <- file.path(dir_fig, "tables")
dir.create(dir_tab, recursive = TRUE, showWarnings = FALSE)

meas <- readRDS(file.path(dir_res, "wmb_meas.rds"))
cells_a <- read.csv(file.path(fig_root, "data", "wmb_plilaorb", "cells_10Xv2.csv"))
cells_b <- read.csv(file.path(fig_root, "data", "wmb_plilaorb", "cells_10Xv3.csv"))
sims <- readRDS(file.path(dir_res, "wmb_similarity.rds"))
fwd_map <- read.csv(file.path(dir_res, "wmb_map_subclass_to_tree.csv"))
rev_map <- read.csv(file.path(dir_res, "wmb_map_cluster_to_subclass.csv"))

# taxonomy: Allen ids carry numeric prefixes, so lexical sort = curated order
tax_b <- unique(cells_b[, c("class", "subclass", "supertype", "cluster")])
tax_b <- tax_b[order(tax_b$class, tax_b$subclass, tax_b$supertype, tax_b$cluster), ]
tree_b <- tn_tree_from_levels(tax_b)
subclasses <- sort(unique(cells_a$subclass))
tree_a <- tn_tree_from_levels(data.frame(leaf = subclasses))

sub_of   <- setNames(cells_a$subclass, cells_a$cluster)[!duplicated(cells_a$cluster)]
class_of <- setNames(cells_a$class, cells_a$subclass)[!duplicated(cells_a$subclass)]
truth_parent <- setNames(cells_b$subclass, cells_b$cluster)[!duplicated(cells_b$cluster)]
truth_leaves <- lapply(subclasses, function(s)
  sort(unique(cells_b$cluster[cells_b$subclass == s])))
names(truth_leaves) <- subclasses
cache_b_sub <- tn_aggregate_cache(meas$cache_b, sub_of)

# node id helpers for tree_b ("class:x" / "subclass:x" / "supertype:x" / leaf)
sub_node <- function(s) paste0("subclass:", s)
node_class <- function(id) {
  # the class a tree_b node belongs to (root -> NA)
  if (id == "root") return(NA_character_)
  while (!startsWith(id, "class:")) {
    if (id %in% names(truth_parent))       # a cluster leaf
      return(unname(cells_b$class[match(truth_parent[id],
                                        cells_b$subclass)]))
    id <- tree_b$parent[match(id, tree_b$id)]
    if (is.na(id)) return(NA_character_)
  }
  sub("^class:", "", id)
}
ancestors <- function(tree, id) {
  out <- character(0)
  repeat {
    id <- tree$parent[match(id, tree$id)]
    if (is.na(id)) return(out)
    out <- c(out, id)
  }
}

# outcome classification, shared by figures 2/4/5
classify_fwd <- function(selected, query, relation) {
  true_id <- sub_node(query)
  # a subclass whose node chain-collapses (single cluster) is exact at the leaf
  true_set <- truth_leaves[[query]]
  if (is.na(selected)) return(if (relation == "unmatched") "unmatched" else "premature_stop")
  if (identical(sort(tn_leaves_under(tree_b, selected)), true_set)) return("exact")
  if (selected %in% ancestors(tree_b, true_id)) return("premature_stop")
  if (identical(node_class(selected), unname(class_of[query]))) return("adjacent_same_class")
  "wrong_branch"
}
classify_rev <- function(selected, query, relation) {
  truth <- unname(truth_parent[query])
  if (is.na(selected)) return(if (relation == "unmatched") "unmatched" else "premature_stop")
  if (selected == truth) return("exact")
  if (identical(unname(class_of[selected]), unname(class_of[truth])))
    return("adjacent_same_class")
  "wrong_branch"
}

outcome_colors <- c(exact = "#2b8cbe", correct_family = "#2b8cbe",
                    adjacent_same_class = "#fdae61", premature_stop = "#984ea3",
                    wrong_branch = "#d7191c", unmatched = "#666666")

save_fig <- function(plot, name, width, height, pages = NULL) {
  pdf(file.path(dir_fig, paste0(name, ".pdf")), width = width, height = height)
  if (is.null(pages)) print(plot) else for (p in pages) print(p)
  dev.off()
  if (is.null(pages)) {
    png(file.path(dir_fig, paste0(name, ".png")), width = width * 110,
        height = height * 110, res = 110)
    print(plot)
    dev.off()
  }
}
