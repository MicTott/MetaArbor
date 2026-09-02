# Hierarchical two-atlas simulation with dialable batch effects.
# Ground truth: `n_family` families x `n_sub` subtypes. Family expression
# signal is stronger than subtype signal (coarse types are real). The two
# datasets share biology but differ by gene-wise multiplicative batch factors
# of strength `batch_sd` plus library-size variation — a crude but honest
# stand-in for the real batch axes the Allen benchmark will provide.

ma_simulate_pair <- function(n_family = 4, n_sub = 3, cells_per_leaf = 120,
                             n_genes = 2000, fam_markers = 40, sub_markers = 15,
                             fam_lfc = 1.2, sub_lfc = 0.9,
                             batch_sd = 0.5, seed = 1) {
  set.seed(seed)
  fams <- paste0("F", seq_len(n_family))
  leaves <- as.vector(outer(fams, paste0("s", seq_len(n_sub)), paste, sep = "."))
  base_mu <- rlnorm(n_genes, meanlog = 0, sdlog = 1)          # shared baseline
  genes <- paste0("g", seq_len(n_genes))

  # disjoint marker blocks
  pool <- sample(n_genes)
  fam_idx <- split(pool[seq_len(n_family * fam_markers)],
                   rep(seq_len(n_family), each = fam_markers))
  off <- n_family * fam_markers
  sub_idx <- split(pool[off + seq_len(length(leaves) * sub_markers)],
                   rep(seq_along(leaves), each = sub_markers))

  leaf_mu <- sapply(seq_along(leaves), function(l) {
    mu <- base_mu
    f <- match(sub("\\..*", "", leaves[l]), fams)
    mu[fam_idx[[f]]] <- mu[fam_idx[[f]]] * exp(fam_lfc)
    mu[sub_idx[[l]]] <- mu[sub_idx[[l]]] * exp(sub_lfc)
    mu
  })
  colnames(leaf_mu) <- leaves

  make_dataset <- function(tag) {
    batch <- rlnorm(n_genes, 0, batch_sd)                     # gene-wise batch
    counts <- lapply(leaves, function(l) {
      lib <- rgamma(cells_per_leaf, shape = 10, rate = 10)    # cell depth
      lam <- outer(leaf_mu[, l] * batch, lib)
      matrix(rpois(length(lam), lam), n_genes,
             dimnames = list(genes, paste(tag, l, seq_len(cells_per_leaf), sep = "_")))
    })
    list(counts = do.call(cbind, counts),
         leaf = rep(leaves, each = cells_per_leaf),
         family = rep(sub("\\..*", "", leaves), each = cells_per_leaf))
  }

  list(A = make_dataset("A"), B = make_dataset("B"),
       families = fams, leaves = leaves)
}
