# FUGW estimator (DESIGN.md §3c): MetaNeighbor-informed fused unbalanced
# Gromov-Wasserstein over LEAVES ONLY, solved by POT via python/fugw_run.py.
# Internal nodes never enter the transport support; they receive mass only
# through post-hoc aggregation (§3d).

#' Solve the FUGW coupling between two atlases' terminal populations.
#'
#' @param M leaf cost matrix (rows = atlas A leaves, cols = atlas B leaves),
#'   e.g. costs$M from tn_measure — 1 minus symmetrized AUROC
#' @param CA,CB within-tree leaf distance matrices (tn_leaf_path_dist),
#'   rescaled internally to [0, 1] so the GW term is commensurate with M
#' @param wA,wB leaf weights; default uniform per population (DESIGN.md §3e —
#'   cell-count weights are the sensitivity analysis, not the default)
#' @param alpha design-space trade-off, cost = alpha*M + (1-alpha)*GW; mapped
#'   to POT's linear-term coefficient as alpha/(1-alpha)
#' @param rho marginal-relaxation strength (KL), epsilon entropic term
#' @param python path to a python with POT installed
#' @param script path to fugw_run.py (default: python/ under the working dir)
tn_fugw <- function(M, CA, CB, wA = NULL, wB = NULL,
                    alpha = 0.5, rho = 1, epsilon = 0.001,
                    python = Sys.getenv("TN_PYTHON", "python3"),
                    script = file.path("python", "fugw_run.py"),
                    work_dir = tempfile("fugw")) {
  stopifnot(alpha > 0, alpha < 1,
            identical(rownames(M), rownames(CA)),
            identical(colnames(M), rownames(CB)))
  if (is.null(wA)) wA <- rep(1 / nrow(M), nrow(M))
  if (is.null(wB)) wB <- rep(1 / ncol(M), ncol(M))
  dir.create(work_dir, recursive = TRUE)
  wr <- function(x, f) write.table(x, file.path(work_dir, f), sep = ",",
                                   row.names = FALSE, col.names = FALSE)
  wr(M, "M.csv"); wr(CA / max(CA), "CA.csv"); wr(CB / max(CB), "CB.csv")
  wr(wA, "wA.csv"); wr(wB, "wB.csv")
  writeLines(sprintf('{"alpha_pot": %.10g, "rho": %.10g, "epsilon": %.10g}',
                     alpha / (1 - alpha), rho, epsilon),
             file.path(work_dir, "params.json"))
  if (!file.exists(script)) stop("fugw_run.py not found at ", script,
                                 " — pass `script` explicitly")
  status <- system2(python, c(script, work_dir), stdout = TRUE, stderr = TRUE)
  pi_file <- file.path(work_dir, "pi.csv")
  if (!file.exists(pi_file)) stop("fugw_run.py failed: ",
                                  paste(status, collapse = "\n"))
  pi <- as.matrix(read.csv(pi_file, header = FALSE))
  dimnames(pi) <- dimnames(M)
  info <- jsonlite_min(file.path(work_dir, "info.json"))
  list(pi = pi, pi_gap = info$pi_gap, mass = info$mass, log = status)
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# minimal flat-JSON reader (avoids a jsonlite dependency)
jsonlite_min <- function(path) {
  txt <- paste(readLines(path, warn = FALSE), collapse = "")
  kv <- regmatches(txt, gregexpr('"[^"]+"\\s*:\\s*[-0-9.eE+]+', txt))[[1]]
  out <- lapply(kv, function(s) as.numeric(sub('.*:\\s*', "", s)))
  names(out) <- vapply(kv, function(s) sub('"([^"]+)".*', "\\1", s), "")
  out
}

#' Roll transported mass up both trees (DESIGN.md §3d):
#' Gamma(u, v) = sum over descendant leaves of u and v of pi.
#' Returns the mass share each (A node, B node) pair captures.
tn_rollup <- function(pi, tree_a, tree_b) {
  nodes_a <- tree_a$id[tree_a$id != "root"]
  nodes_b <- tree_b$id[tree_b$id != "root"]
  G <- matrix(0, length(nodes_a), length(nodes_b),
              dimnames = list(nodes_a, nodes_b))
  for (u in nodes_a) {
    lu <- intersect(tn_leaves_under(tree_a, u), rownames(pi))
    if (!length(lu)) next
    for (v in nodes_b) {
      lv <- intersect(tn_leaves_under(tree_b, v), colnames(pi))
      if (!length(lv)) next
      G[u, v] <- sum(pi[lu, lv, drop = FALSE])
    }
  }
  G
}

#' For each atlas-A leaf, the share of its transported mass captured by each
#' B node; the FUGW analogue of the baseline's selected node is the deepest
#' B node holding at least `q` of the leaf's mass.
tn_fugw_assign <- function(pi, tree_b, q = 0.9) {
  res <- lapply(rownames(pi), function(i) {
    row <- pi[i, ]
    tot <- sum(row)
    if (tot <= 0) return(data.frame(query = i, node = NA, share = NA))
    nodes <- tree_b$id[tree_b$id != "root"]
    share <- vapply(nodes, function(v)
      sum(row[intersect(tn_leaves_under(tree_b, v), names(row))]) / tot,
      numeric(1))
    ok <- nodes[share >= q]
    if (!length(ok)) return(data.frame(query = i, node = "root", share = max(share)))
    sizes <- vapply(ok, function(v) length(tn_leaves_under(tree_b, v)), numeric(1))
    pick <- ok[which.min(sizes)]
    data.frame(query = i, node = pick, share = share[pick])
  })
  do.call(rbind, res)
}
