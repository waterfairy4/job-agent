"""Is version B really better than version A? Paired significance tests for evaluation runs.

Both versions are scored on the SAME items, so the comparison is paired:
- McNemar's exact test looks only at the items where the two versions differ
  (b = A right and B wrong, c = A wrong and B right). If changes were random, b and c would be
  about equal; the p-value is the chance of a split at least this uneven.
- A paired bootstrap resamples the items many times and gives a 95% confidence interval for the
  difference (in accuracy, or in Cohen's kappa). If the interval contains 0, the data cannot rule
  out "no real improvement".
"""
import random
from math import comb


def mcnemar_exact(a_ok: list[bool], b_ok: list[bool]) -> dict:
    assert len(a_ok) == len(b_ok)
    b = sum(x and not y for x, y in zip(a_ok, b_ok))  # A right, B wrong
    c = sum(y and not x for x, y in zip(a_ok, b_ok))  # A wrong, B right
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return {"a_only": b, "b_only": c, "p": p}


def bootstrap_diff(items: list, metric, a_key: str, b_key: str, rounds: int = 5000, seed: int = 0) -> dict:
    """95% CI of metric(B) - metric(A). `metric(rows, key)` scores a list of items for one version."""
    rng = random.Random(seed)  # fixed seed: the same report every time
    diffs = []
    for _ in range(rounds):
        sample = [items[rng.randrange(len(items))] for _ in items]
        diffs.append(metric(sample, b_key) - metric(sample, a_key))
    diffs.sort()
    return {"diff": metric(items, b_key) - metric(items, a_key),
            "low": diffs[int(0.025 * rounds)], "high": diffs[int(0.975 * rounds) - 1]}


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test: are wins and losses (ties dropped) more uneven than chance?"""
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n)


def weighted_kappa(pairs: list[tuple[int, int]], low: int = 1, high: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal scores (a 4 vs 5 counts as a small disagreement)."""
    if not pairs:
        return 0.0
    cats = range(low, high + 1)
    w = lambda i, j: (i - j) ** 2 / (high - low) ** 2
    n = len(pairs)
    ra, rb = [sum(a == c for a, _ in pairs) / n for c in cats], [sum(b == c for _, b in pairs) / n for c in cats]
    observed = sum(w(a, b) for a, b in pairs) / n
    expected = sum(w(i, j) * ra[i - low] * rb[j - low] for i in cats for j in cats)
    return 1.0 if expected == 0 else 1 - observed / expected


def verdict_text(p: float, low: float, high: float) -> str:
    if p < 0.05 and (low > 0 or high < 0):
        return "The difference is statistically significant (p < 0.05, CI excludes 0)."
    return ("Not statistically significant: the difference could be chance on this many items "
            "(p ≥ 0.05 or the CI includes 0). More test items are needed to be sure.")
