"""Paired comparison of two conditions: paired t-interval and exact Wilcoxon signed-rank test.

The paper pairs each source run's continuation with and without SIM and reports a two-sided
paired t test with its 95% interval, a two-sided signed-rank check and the effect size d_z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        for numerator in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            delta = d * c
            h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    """P(|T| >= |t|) for Student's t with `df` degrees of freedom."""
    if math.isinf(t):
        return 0.0
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def t_quantile(p: float, df: int) -> float:
    """The t with P(T <= t) = p, for p in [0.5, 1)."""
    if not 0.5 <= p < 1.0:
        raise ValueError("p must be in [0.5, 1)")
    target = 2.0 * (1.0 - p)
    low, high = 0.0, 1.0
    while t_two_sided_p(high, df) > target:
        high *= 2.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if t_two_sided_p(mid, df) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def signed_rank(differences: list[float]) -> tuple[float, float, float]:
    """Exact two-sided Wilcoxon signed-rank test. Returns (W+, W-, p).

    Zero differences are dropped. Ties share the mean of their ranks, and the null distribution
    is computed exactly for those ranks by counting all sign assignments.
    """
    values = [d for d in differences if abs(d) > 1e-12]
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 1.0
    order = sorted(range(n), key=lambda i: abs(values[i]))
    ranks2 = [0] * n  # ranks times two, so mid-ranks stay integers
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(values[order[j + 1]]) - abs(values[order[i]])) <= 1e-12:
            j += 1
        for k in range(i, j + 1):
            ranks2[order[k]] = i + j + 2  # 2 * mean of ranks i+1 .. j+1
        i = j + 1
    w_plus2 = sum(r for r, v in zip(ranks2, values) if v > 0)
    total2 = sum(ranks2)
    counts = [1] + [0] * total2  # counts[s]: sign assignments with 2 * W+ == s
    for r in ranks2:
        for s in range(total2, r - 1, -1):
            counts[s] += counts[s - r]
    assignments = 2**n
    lower = sum(counts[: w_plus2 + 1]) / assignments
    upper = sum(counts[w_plus2:]) / assignments
    return w_plus2 / 2.0, (total2 - w_plus2) / 2.0, min(1.0, 2.0 * min(lower, upper))


@dataclass
class PairedResult:
    n: int
    mean: float
    sd: float
    t: float
    df: int
    p_t: float
    ci: tuple[float, float]
    level: float
    w_plus: float
    w_minus: float
    p_signed_rank: float
    d_z: float
    wins: int

    def describe(self) -> str:
        return "\n".join([
            f"pairs: {self.n} (treatment better in {self.wins})",
            f"mean difference: {self.mean:+.2f} points (treatment - control), sd {self.sd:.2f}",
            f"{self.level:.0%} CI: [{self.ci[0]:.2f}, {self.ci[1]:.2f}]",
            f"paired t({self.df}) = {self.t:.2f}, two-sided p = {self.p_t:.3f}",
            f"signed rank: W+ = {self.w_plus:g}, W- = {self.w_minus:g}, two-sided exact p = {self.p_signed_rank:.3f}",
            f"effect size d_z = {self.d_z:.2f}",
        ])


def paired(treatment: list[float], control: list[float], level: float = 0.95) -> PairedResult:
    """Compare paired scores: treatment[i] and control[i] come from the same source run."""
    if len(treatment) != len(control):
        raise ValueError("treatment and control must have the same number of scores")
    n = len(treatment)
    if n < 2:
        raise ValueError("at least two pairs are needed")
    diffs = [a - b for a, b in zip(treatment, control)]
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    df = n - 1
    if sd == 0.0:
        t = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
        half, d_z = 0.0, t
    else:
        standard_error = sd / math.sqrt(n)
        t = mean / standard_error
        half = t_quantile(1.0 - (1.0 - level) / 2.0, df) * standard_error
        d_z = mean / sd
    w_plus, w_minus, p_rank = signed_rank(diffs)
    return PairedResult(
        n=n, mean=mean, sd=sd, t=t, df=df, p_t=t_two_sided_p(t, df) if sd else (1.0 if mean == 0.0 else 0.0),
        ci=(mean - half, mean + half), level=level,
        w_plus=w_plus, w_minus=w_minus, p_signed_rank=p_rank, d_z=d_z,
        wins=sum(1 for d in diffs if d > 0),
    )
