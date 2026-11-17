"""Paired comparisons for the paper's tables (pure Python: no numpy, no scipy - identical on every machine).

The unit of a comparison is a PAIR: two heads that differ in one thing only (e.g. trained with and without pose
noise, same head type and same seed, hence the same initial weights and the same training batches), or the same two
models on one validation zone. The test is the exact Wilcoxon signed-rank test: the p-value is the share of the 2^n
sign assignments of the observed |differences| whose rank sum is at least as extreme as the observed one. With n
pairs that all point the same way the one-sided p-value is 2^-n (0.125 for 3 pairs, 0.016 for 6, 0.002 for 9), so
the number of pairs is always reported next to it.
"""
from itertools import product
from typing import Dict, List, Sequence


def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks (1-based) of the values; ties share the mean of their ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def signed_rank_exact(diffs: Sequence[float], alternative: str = "greater", max_n: int = 20) -> float:
    """Exact p-value of the Wilcoxon signed-rank test for paired differences. alternative: "greater" (differences
    tend to be positive), "less", or "two-sided". Zero differences are dropped; no pair left -> 1.0."""
    d = [float(x) for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    if n > max_n:
        raise ValueError(f"{n} pairs: the exact enumeration is meant for at most {max_n}")
    r = _ranks([abs(x) for x in d])
    w = sum(ri for ri, x in zip(r, d) if x > 0)
    ge = le = 0
    for signs in product((0, 1), repeat=n):
        s = sum(ri for ri, k in zip(r, signs) if k)
        ge += s >= w - 1e-9
        le += s <= w + 1e-9
    p_greater, p_less = ge / 2 ** n, le / 2 ** n
    if alternative == "greater":
        return p_greater
    if alternative == "less":
        return p_less
    if alternative == "two-sided":
        return min(1.0, 2 * min(p_greater, p_less))
    raise ValueError(f"unknown alternative {alternative!r}")


def paired_summary(diffs: Sequence[float], alternative: str = "greater") -> Dict:
    """{"n", "mean", "min", "max", "positive", "p", "alternative"} of paired differences."""
    d = [float(x) for x in diffs]
    if not d:
        return {"n": 0, "mean": None, "min": None, "max": None, "positive": 0, "p": None, "alternative": alternative}
    return {"n": len(d), "mean": sum(d) / len(d), "min": min(d), "max": max(d), "positive": sum(x > 0 for x in d),
            "p": signed_rank_exact(d, alternative), "alternative": alternative}
