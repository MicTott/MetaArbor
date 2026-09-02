# Portable MINSTD (Park-Miller) generator for cross-language-identical
# bootstrap draws. Products stay below 2^53, so the recurrence is exact in
# doubles; the python package implements the same stream, and given the same
# seed both implementations consume identical index sequences.

ma_minstd_new <- function(seed) {
  e <- new.env(parent = emptyenv())
  e$state <- (as.numeric(seed) %% 2147483646) + 1  # in [1, M-1], never 0
  e
}

#' Next 0-based index in [0, n).
ma_minstd_index <- function(rng, n) {
  rng$state <- (16807 * rng$state) %% 2147483647
  rng$state %% n
}

#' Draw k 0-based indices in [0, n).
ma_minstd_indices <- function(rng, k, n) {
  out <- numeric(k)
  for (i in seq_len(k)) out[i] <- ma_minstd_index(rng, n)
  out
}
