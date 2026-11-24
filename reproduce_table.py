"""Reproduce the derived quantities in Table 1 and Sections 3-5 of
"Mean and Median Income: What Their Gap Reveals About Inequality" (v0.4).

Inputs are published point estimates from Census P60-282 (Guzman and Kollar 2024),
Table A-2 (mean, median; 2023 dollars) and Table A-4b (Gini), all households.
Standard library only.
"""
from math import exp, log, sqrt
from statistics import NormalDist, median

PHI = NormalDist()

# Income reference year: (mean, median, published Gini).
# Each CPS ASEC survey collects income for the preceding calendar year.
# 2017 uses the updated-processing row; 2020 onward uses 2020 Census controls.
DATA = {
    2017: (107_300, 74_810, 0.489),
    2018: (108_000, 75_790, 0.486),
    2019: (115_900, 81_210, 0.484),
    2020: (114_000, 79_560, 0.488),
    2021: (114_600, 79_260, 0.494),
    2022: (110_600, 77_540, 0.488),
    2023: (114_500, 80_610, 0.485),
}


def gini_lognormal(r: float) -> float:
    """Gini implied by a lognormal distribution with mean/median ratio r >= 1."""
    return 2 * PHI.cdf(sqrt(log(r))) - 1


def ratio_lognormal(g: float) -> float:
    """Mean/median ratio a lognormal distribution needs to have Gini g."""
    return exp(PHI.inv_cdf((1 + g) / 2) ** 2)


def gini_discrete(x):
    """Population Gini: sum_i sum_j |x_i - x_j| / (2 n^2 mean)."""
    n, a = len(x), sum(x) / len(x)
    return sum(abs(p - q) for p in x for q in x) / (2 * n * n * a)


def sign(x: float) -> int:
    """Classify a computed change as a decline (-1), no change (0), or rise (1).

    Two zero changes agree; a zero and a nonzero change do not.
    This exact arithmetic classification is not a significance test.
    """
    return (x > 0) - (x < 0)


def main() -> None:
    print(f"{'Income year':<12}{'R':>8}{'G':>8}{'G_LN':>8}{'G-G_LN':>9}")
    g_ln = {}
    for year, (a, m, g) in DATA.items():
        r = a / m
        g_ln[year] = gini_lognormal(r)
        print(f"{year:<12}{r:8.3f}{g:8.3f}{g_ln[year]:8.3f}{g - g_ln[year]:9.4f}")

    print("\nYear-to-year changes (published G vs G_LN):")
    years = sorted(DATA)
    same = 0
    for y0, y1 in zip(years, years[1:]):
        dg, dln = DATA[y1][2] - DATA[y0][2], g_ln[y1] - g_ln[y0]
        agree = sign(dg) == sign(dln)
        same += agree
        print(f"  {y0}-{y1}: {dg:+.3f} vs {dln:+.4f}  {'same' if agree else 'DIFFERENT'}")
    print(f"  Same direction in {same} of {len(years) - 1} changes")

    (a0, m0, _), (a1, m1, _) = DATA[2017], DATA[2023]
    print("\n2017 -> 2023:")
    print(f"  mean growth   {a1 / a0 - 1:+.2%}   dlog a = {log(a1 / a0):.4f}")
    print(f"  median growth {m1 / m0 - 1:+.2%}   dlog m = {log(m1 / m0):.4f}")
    print(f"  R change      {(a1 / m1) / (a0 / m0) - 1:+.2%}   dlog R = {log((a1 / m1) / (a0 / m0)):.4f}")
    print(f"  absolute gap  {a0 - m0:,} -> {a1 - m1:,}")
    print(f"  R needed for published 2023 Gini under lognormality: {ratio_lognormal(DATA[2023][2]):.3f}")

    print("\nCounterexamples (Section 3):")
    for name, x in {"A": [1, 2, 3, 4, 10], "B": [1, 2, 3, 6, 8],
                    "C": [1, 2, 4, 5, 8], "C'": [1.5, 2, 3.5, 5, 8]}.items():
        s = sorted(x)
        mean, med = sum(s) / len(s), median(s)
        print(f"  {name:<3} mean={mean:.2f} median={med:.2f} R={mean / med:.3f} G={gini_discrete(s):.3f}")


if __name__ == "__main__":
    main()
