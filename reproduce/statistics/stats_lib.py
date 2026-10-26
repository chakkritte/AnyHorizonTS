"""Statistics of the paper: per-cell tests on the five per-seed MSEs of two arms.

For two arms with five per-seed MSEs each (a = the tested arm, b = the reference arm):
- welch(a, b): two-sided Welch's t-test (scipy.stats.ttest_ind, equal_var=False).
- holm(p): Holm's step-down adjusted p values within one family of tests (monotone, capped at 1).
- tost(a, b, margin): equivalence by two one-sided Welch tests with margin delta = margin * mean(b), i.e. the hypotheses
  mean(a) - mean(b) <= -delta and mean(a) - mean(b) >= delta are both rejected at alpha = 0.05. Equivalently, the 90% Welch
  interval of mean(a) - mean(b) lies inside (-delta, +delta). The reference mean is treated as fixed when the margin is set.
- ci(a, b, level): Welch interval of mean(a) - mean(b), returned in % of mean(b).
- min_margin(a, b): the smallest margin (in % of mean(b)) at which tost() would declare equivalence = max(|lo|, |hi|) of the 90% interval.
- mde(a, b): the relative difference (% of mean(b)) that a two-sided Welch test at alpha = 0.05 detects with 80% power, given the
  observed standard deviations and the Welch degrees of freedom (noncentral t).
- binom_tail(k, n, alpha): P(X >= k) for X ~ Binomial(n, alpha), the chance of at least k significant cells if no cell differed and the
  tests were independent (they are not: cells of one dataset share data, so this is only a rough reference).
- cell(a, b, margin): all of the above for one cell, as a dict (d_pct = 100 * (mean(a) - mean(b)) / mean(b); positive = a is worse).
"""
import numpy as np
from scipy import stats

ALPHA = 0.05


def _se_df(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return se, df


def welch(a, b):
    t = stats.ttest_ind(np.asarray(a, float), np.asarray(b, float), equal_var=False)
    return float(t.statistic), float(t.pvalue)


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def ci(a, b, level=0.90):
    a, b = np.asarray(a, float), np.asarray(b, float)
    se, df = _se_df(a, b)
    d = a.mean() - b.mean()
    q = stats.t.ppf(0.5 + level / 2, df)
    return 100 * (d - q * se) / b.mean(), 100 * (d + q * se) / b.mean()


def tost(a, b, margin):
    a, b = np.asarray(a, float), np.asarray(b, float)
    se, df = _se_df(a, b)
    d, delta = a.mean() - b.mean(), margin * b.mean()
    p_low = stats.t.sf((d + delta) / se, df)      # H0: d <= -delta
    p_high = stats.t.cdf((d - delta) / se, df)    # H0: d >= +delta
    return float(max(p_low, p_high))


def min_margin(a, b):
    lo, hi = ci(a, b, 0.90)
    return max(abs(lo), abs(hi))


def mde(a, b, power=0.80, alpha=ALPHA):
    a, b = np.asarray(a, float), np.asarray(b, float)
    se, df = _se_df(a, b)
    q = stats.t.ppf(1 - alpha / 2, df)

    def pw(ncp):
        v = stats.nct.sf(q, df, ncp) + stats.nct.cdf(-q, df, ncp)
        return 1.0 if not np.isfinite(v) else v   # scipy's nct returns NaN far in the tail, where the power is ~1
    lo, hi = 0.0, 20.0
    for _ in range(80):                       # bisection: power rises monotonically with the noncentrality
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pw(mid) < power else (lo, mid)
    ncp = (lo + hi) / 2
    return 100 * ncp * se / b.mean()


def binom_tail(k, n, alpha=ALPHA):
    return float(stats.binom.sf(k - 1, n, alpha))


def cell(a, b, margin=0.02):
    """All per-cell quantities for one comparison (a against reference b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = welch(a, b)
    lo, hi = ci(a, b, 0.90)
    lo95, hi95 = ci(a, b, 0.95)
    return dict(mse_a=a.mean(), sd_a=a.std(ddof=1), mse_b=b.mean(), sd_b=b.std(ddof=1),
                d_pct=100 * (a.mean() - b.mean()) / b.mean(), welch_t=t, welch_p=p,
                ci90_lo=lo, ci90_hi=hi, ci95_lo=lo95, ci95_hi=hi95,
                tost_p=tost(a, b, margin), min_margin_pct=min_margin(a, b), mde80_pct=mde(a, b),
                lower_in=int(sum(x < y for x, y in zip(a, b))))
