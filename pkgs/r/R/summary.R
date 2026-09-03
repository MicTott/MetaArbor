# MetaArbor interpretation layer (numeric): one-row-per-query alignment
# summaries and the joint Walk/Transport agreement categories. Mirrors the
# python implementation exactly (parity-gated); exposes only quantities the
# frozen estimators already compute — no new thresholds, no altered
# decisions. Walk rows re-derive the walk with the same per-query seeds as
# ma_baseline_map, so summarized decisions are exactly the map's decisions.

ma_node_depth <- function(tree, node) length(ma_ancestors(tree, node))

ma_ancestors <- function(tree, id) {
  out <- character(0)
  repeat {
    id <- tree$parent[match(id, tree$id)]
    if (is.na(id)) return(out)
    out <- c(out, id)
  }
}

#' One row per query for MetaArbor-Walk.
ma_walk_summary <- function(cache, test_labels, tree, S_dir, base_seed = 7, ...) {
  queries <- sort(unique(as.character(test_labels)))
  map <- ma_baseline_map(cache, test_labels, tree, S_dir,
                         base_seed = base_seed, ...)
  res <- lapply(seq_along(queries), function(i) {
    q <- queries[i]
    sel <- ma_select_node(cache, test_labels, q, tree,
                          seed = base_seed + i - 1, trace = TRUE, ...)
    m <- map[map$query == q, ]
    last <- if (nrow(sel$path)) sel$path[nrow(sel$path), ] else NULL
    best_auc <- second_auc <- NA_real_
    if (!is.null(last) && !is.null(sel$trace)) {
      st <- sel$trace[sel$trace$split_at ==
                      sel$trace$split_at[nrow(sel$trace)], ]
      best_auc <- st$child_auroc[match(last$best, st$child)]
      second_auc <- st$child_auroc[match(last$second, st$child)]
    }
    support <- if (is.null(last)) NA_real_
      else if (last$override) NA_real_
      else if (last$stopped && !is.na(last$par_gt0) && last$par_lo > 0)
        last$par_gt0
      else if (last$stopped) 1 - last$sib_gt_margin
      else last$sib_gt_margin
    data.frame(
      query = q,
      walk_selected = m$selected,
      walk_depth = if (is.na(m$selected)) NA_real_
                   else ma_node_depth(tree, m$selected),
      walk_parent = if (is.na(m$selected)) NA_character_
                    else tree$parent[match(m$selected, tree$id)],
      walk_auroc = m$auroc,
      walk_best_sib_auroc = best_auc,
      walk_second_sib_auroc = second_auc,
      walk_sib_delta = if (is.null(last)) NA_real_ else last$sib_delta,
      walk_vote = if (is.null(last)) NA_real_ else last$vote,
      walk_override = if (is.null(last)) NA else last$override,
      walk_decision_support = support,
      walk_relation = m$relation)
  })
  out <- do.call(rbind, res)
  rownames(out) <- NULL
  out
}

#' One row per query for MetaArbor-Transport, from a coupling matrix
#' computed by the python package (`pi`: rows = queries, cols = target
#' leaves; plain matrix with dimnames). `family_of_leaf`: named vector
#' leaf -> grouping label at an explicit target-taxonomy level.
ma_transport_summary <- function(pi, family_of_leaf, tree = NULL) {
  fams <- sort(unique(unname(family_of_leaf[colnames(pi)])))
  res <- lapply(rownames(pi), function(q) {
    r <- pi[q, ]
    tot <- sum(r)
    if (tot <= 0) return(data.frame(
      query = q, transport_family = NA_character_,
      transport_node = NA_character_, transport_mass = 0,
      transport_eff_leaves = NA_real_, transport_bin = "unmatched"))
    p <- r / tot
    fam_mass <- vapply(fams, function(f)
      sum(p[unname(family_of_leaf[colnames(pi)]) == f]), numeric(1))
    best <- fams[which.max(fam_mass)]
    mass <- max(fam_mass)
    pn <- p[p > 0]
    eff <- exp(-sum(pn * log(pn)))
    node <- NA_character_
    if (!is.null(tree)) {
      fam_leaves <- sort(names(family_of_leaf)[family_of_leaf == best])
      cand <- tree$id[tree$id != "root"]
      cand <- cand[vapply(cand, function(n) identical(
        sort(intersect(ma_leaves_under(tree, n), colnames(pi))), fam_leaves),
        logical(1))]
      if (length(cand))
        node <- cand[which.min(vapply(cand, function(n)
          ma_node_depth(tree, n), numeric(1)))]
    }
    data.frame(query = q, transport_family = best, transport_node = node,
               transport_mass = mass, transport_eff_leaves = eff,
               transport_bin = if (mass >= 0.9) "confident"
                               else if (mass >= 0.5) "moderate" else "diffuse")
  })
  out <- do.call(rbind, res)
  rownames(out) <- NULL
  out
}

#' The documented six-way agreement category for one query.
ma_agreement <- function(walk_selected, transport_node, tree) {
  w <- walk_selected; t <- transport_node
  if (is.na(w) && is.na(t)) return("both_unmatched")
  if (is.na(w)) return("transport_only")
  if (is.na(t)) return("walk_only")
  if (w == t) return("agree")
  if (w %in% ma_ancestors(tree, t) || t %in% ma_ancestors(tree, w))
    return("same_branch_different_depth")
  "conflicting_branch"
}

#' Join the per-query summaries and add the agreement category.
ma_alignment_summary <- function(walk_df, transport_df, tree) {
  out <- merge(walk_df, transport_df, by = "query", all.x = TRUE, sort = TRUE)
  out$transport_bin[is.na(out$transport_bin)] <- "unmatched"
  out$agreement <- vapply(seq_len(nrow(out)), function(i)
    ma_agreement(out$walk_selected[i], out$transport_node[i], tree),
    character(1))
  out
}
