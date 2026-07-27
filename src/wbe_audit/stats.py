"""Confidence intervals for the audit proportions.

Wilson for the headline rates (it stays sensible at small proportions, where the
normal approximation doesn't) and exact Clopper-Pearson for the validation gate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

try:  # scipy is present in this environment; degrade gracefully if not
    from scipy.stats import beta as _beta
except ImportError:  # pragma: no cover
    _beta = None


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    k: int
    n: int

    def pct(self) -> str:
        return (f"{100*self.point:.2f}% "
                f"[{100*self.lo:.2f}, {100*self.hi:.2f}]  ({self.k}/{self.n})")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0, 0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), k, n)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Interval:
    """Exact binomial interval; used for the frozen-classifier validation gate."""
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0, 0)
    if _beta is None:  # pragma: no cover
        return wilson(k, n)
    lo = 0.0 if k == 0 else _beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta.ppf(1 - alpha / 2, k + 1, n - k)
    return Interval(k / n, float(lo), float(hi), k, n)


def finite_population_correction(n: int, N: int) -> float:
    """sqrt((N-n)/(N-1)); negligible here but reported for completeness."""
    if N <= 1 or n >= N:
        return 0.0
    return math.sqrt((N - n) / (N - 1))
