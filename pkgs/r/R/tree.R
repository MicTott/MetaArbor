# Tree representation and utilities (DESIGN.md §2, §3).
#
# A tree is a data.frame of nodes: id (character), parent (character or NA for
# the root), is_leaf (logical). Leaf ids must equal the leaf labels used by the
# vote cache. Built from taxonomy level columns (coarse -> fine), which is how
# curated atlas hierarchies ship.

#' Build a tree from a data.frame of level columns, one row per leaf,
#' ordered coarse -> fine; the last column is the leaf label.
#' Internal node ids are "level:label" to keep labels unique across levels.
tn_tree_from_levels <- function(levels_df) {
  levels_df <- unique(levels_df)
  n_lev <- ncol(levels_df)
  stopifnot(n_lev >= 1, !anyDuplicated(levels_df[[n_lev]]))
  nodes <- data.frame(id = "root", parent = NA_character_, is_leaf = FALSE)
  for (k in seq_len(n_lev)) {
    is_last <- k == n_lev
    ids <- if (is_last) as.character(levels_df[[k]])
           else paste0(names(levels_df)[k], ":", levels_df[[k]])
    parents <- if (k == 1) rep("root", nrow(levels_df))
               else paste0(names(levels_df)[k - 1], ":", levels_df[[k - 1]])
    add <- unique(data.frame(id = ids, parent = parents, is_leaf = is_last))
    if (anyDuplicated(add$id))
      stop("level ", names(levels_df)[k], " labels map to multiple parents")
    nodes <- rbind(nodes, add)
  }
  # collapse single-child chains? kept as-is: curated levels are meaningful
  nodes
}

#' Children ids of a node.
tn_children <- function(tree, id) tree$id[!is.na(tree$parent) & tree$parent == id]

#' Leaf labels under a node (the node itself if a leaf).
tn_leaves_under <- function(tree, id) {
  if (tree$is_leaf[match(id, tree$id)]) return(id)
  out <- character(0)
  stack <- tn_children(tree, id)
  while (length(stack)) {
    x <- stack[[1]]; stack <- stack[-1]
    if (tree$is_leaf[match(x, tree$id)]) out <- c(out, x)
    else stack <- c(stack, tn_children(tree, x))
  }
  out
}

#' Intrinsic recursive marginals from a tree ALONE (NOTES.md item 14):
#' mass 1 at the root; every internal node splits its mass equally among its
#' children, so a leaf's weight is the product of 1/deg(v) along its path.
#' By construction: independent of any paired atlas, its label count, or any
#' known correspondence; invariant to label names; refining one leaf into k
#' children redistributes that branch's mass without changing its total —
#' annotation resolution never buys transport capacity. Single-child chains
#' pass mass through unchanged. Returns leaf weights (sum to 1); with
#' `all_nodes = TRUE`, every node's mass, for conservation checks.
tn_tree_weights <- function(tree, all_nodes = FALSE) {
  w <- setNames(numeric(nrow(tree)), tree$id)
  w["root"] <- 1
  queue <- "root"
  while (length(queue)) {
    v <- queue[[1]]; queue <- queue[-1]
    kids <- tn_children(tree, v)
    if (!length(kids)) next
    w[kids] <- w[v] / length(kids)
    queue <- c(queue, kids)
  }
  if (all_nodes) w else w[tree$id[tree$is_leaf]]
}

#' Path (hop) distance matrix between leaves — the structure input C^A / C^B
#' for FUGW (DESIGN.md §3c). Cophenetic alternative: depth-weighted variant
#' to be added when branch lengths exist.
tn_leaf_path_dist <- function(tree) {
  leaves <- tree$id[tree$is_leaf]
  anc <- function(id) {
    path <- id
    while (!is.na(tree$parent[match(id, tree$id)])) {
      id <- tree$parent[match(id, tree$id)]
      path <- c(path, id)
    }
    path
  }
  paths <- lapply(leaves, anc)
  names(paths) <- leaves
  d <- matrix(0, length(leaves), length(leaves), dimnames = list(leaves, leaves))
  for (a in seq_along(leaves)) for (b in seq_len(a - 1)) {
    pa <- paths[[a]]; pb <- paths[[b]]
    common <- intersect(pa, pb)[1]  # lowest common ancestor (paths are leaf->root)
    d[a, b] <- d[b, a] <- (match(common, pa) - 1) + (match(common, pb) - 1)
  }
  d
}
