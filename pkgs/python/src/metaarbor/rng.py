"""Portable MINSTD (Park-Miller) generator for cross-language-identical
bootstrap draws. The R companion package implements the same recurrence in
doubles (products stay below 2^53, so it is exact there too); given the same
seed, both languages consume identical index streams, making walk decisions
bit-comparable across implementations.
"""

_M = 2147483647  # 2^31 - 1
_A = 16807


class Minstd:
    def __init__(self, seed: int):
        self.state = (int(seed) % (_M - 1)) + 1  # in [1, M-1], never 0

    def next_raw(self) -> int:
        self.state = (_A * self.state) % _M
        return self.state

    def index(self, n: int) -> int:
        """0-based index in [0, n). Modulo bias is negligible for n << 2^31
        and identical across languages, which is what matters here."""
        return self.next_raw() % n

    def indices(self, k: int, n: int):
        """k 0-based indices in [0, n) — tight-loop batch version of
        index(), identical stream."""
        s = self.state
        out = [0] * k
        for i in range(k):
            s = (16807 * s) % 2147483647
            out[i] = s % n
        self.state = s
        return out
