"""
VBLS -- robustness of the a-posteriori sample restriction in Section 17.3.5.3
============================================================================

Section 17.3 reports two regressions of the somatic-mutation rate on maximum
longevity, rate ~ T_max^phi (VBLS predicts phi = -1):

    all 15 species          phi = -0.697, R^2 = 0.586   (primary result)
    11 species w/o anomalies phi = -0.814, R^2 = 0.800   (secondary)

Three of the four excluded species (horse, tiger, rabbit) are excluded a
posteriori; only the naked mole-rat is excluded on theoretical grounds
(E-channel). This script asks whether the reported improvement survives
scrutiny:

  1. leave-one-out on the restricted set -- does phi depend on one species?
  2. THE DECISIVE TEST: compare the achieved (phi, R^2) against the full
     distribution over all C(15,4) = 1365 ways of dropping four species.
     If the actual choice sits in the middle of that distribution, the
     restriction carries no information beyond "drop any four".
  3. sensitivity to the cat's T_max (17 yr as used vs 30 yr in AnAge).
  4. bootstrap confidence intervals, to judge the improvement against
     sampling error.

Run: python cagan_robustness.py
"""
import itertools

import numpy as np
from scipy import stats

# (species, mutation rate per year, T_max yr, flag: N normal / A anomalous)
# Cat T_max = 17 (AnAge reference value 30 is a captive record; see Section 17.4.2)
RECORDS = [
    ('Rat',                     607,   3.8, 'N'),
    ('Mouse',                   761,   4.0, 'N'),
    ('Ferret',                  485,  10.0, 'N'),
    ('Rabbit',                   98,  13.0, 'A'),
    ('Dog',                     222,  20.0, 'N'),
    ('Tiger',                   367,  26.0, 'A'),
    ('Ring-tailed lemur',       102,  27.0, 'N'),
    ('Lion',                    160,  29.0, 'N'),
    ('Cat',                     547,  17.0, 'N'),
    ('Cow',                     220,  30.0, 'N'),
    ('Black-and-white colobus',  79,  30.0, 'N'),
    ('Naked mole-rat',          103,  37.0, 'A'),
    ('Giraffe',                 100,  39.5, 'N'),
    ('Horse',                   201,  57.0, 'A'),
    ('Human',                    53, 122.5, 'N'),
]
EXCLUDED = {'Rabbit', 'Tiger', 'Naked mole-rat', 'Horse'}
RNG = np.random.default_rng(20260730)


def fit(rows):
    lt = np.log10([r[2] for r in rows])
    lr = np.log10([r[1] for r in rows])
    res = stats.linregress(lt, lr)
    return res.slope, res.rvalue ** 2


def show(tag, rows):
    phi, r2 = fit(rows)
    print(f"  {tag:34s} n={len(rows):>3}  phi={phi:+.4f}  R^2={r2:.4f}")
    return phi, r2


print("=" * 76)
print("0. REPRODUCTION OF THE PUBLISHED VALUES")
print("=" * 76)
all15 = RECORDS
keep11 = [r for r in RECORDS if r[0] not in EXCLUDED]
phi15, r215 = show("all species (Section 17.3.4)", all15)
phi11, r211 = show("without the four anomalies", keep11)
print(f"  published: -0.697 / 0.586  and  -0.814 / 0.800")

print()
print("=" * 76)
print("1. LEAVE-ONE-OUT ON THE RESTRICTED SET (n=11)")
print("=" * 76)
print(f"  {'excluded':26s} {'phi':>9} {'R^2':>8} {'d(phi)':>9}")
print("  " + "-" * 56)
loo = []
for i, r in enumerate(keep11):
    sub = keep11[:i] + keep11[i + 1:]
    p, q = fit(sub)
    loo.append(p)
    print(f"  {r[0]:26s} {p:+9.4f} {q:8.4f} {p - phi11:+9.4f}")
loo = np.array(loo)
print(f"\n  range [{loo.min():+.4f}; {loo.max():+.4f}], sd={loo.std():.4f}")
worst = keep11[int(np.argmax(np.abs(loo - phi11)))][0]
print(f"  most influential species: {worst}")

print()
print("=" * 76)
print("2. DECISIVE TEST: is dropping THESE four species informative?")
print("   comparison against all C(15,4) = 1365 possible drops of four")
print("=" * 76)
combos = list(itertools.combinations(range(15), 4))
phis, r2s = [], []
for c in combos:
    sub = [RECORDS[i] for i in range(15) if i not in c]
    p, q = fit(sub)
    phis.append(p)
    r2s.append(q)
phis, r2s = np.array(phis), np.array(r2s)

pct_phi = 100.0 * np.mean(phis <= phi11)      # phi closer to -1 = smaller = better
pct_r2 = 100.0 * np.mean(r2s <= r211)
print(f"  actual choice:            phi={phi11:+.4f}   R^2={r211:.4f}")
print(f"  distribution over 1365:   phi median {np.median(phis):+.4f} "
      f"[{phis.min():+.4f}; {phis.max():+.4f}]")
print(f"                            R^2 median {np.median(r2s):.4f} "
      f"[{r2s.min():.4f}; {r2s.max():.4f}]")
print(f"\n  percentile of the actual choice:  phi {pct_phi:5.1f}%   R^2 {pct_r2:5.1f}%")
print(f"  i.e. only {pct_phi:.1f}% of arbitrary 4-drops give phi as close to -1 or closer,")
print(f"       only {100 - pct_r2:.1f}% give R^2 as high or higher.")
print("  The actual choice is therefore near-OPTIMAL among the 1365 available.")
print("  Read correctly, this is evidence of selection, not of validity: a subset")
print("  chosen by inspecting which species deviate is expected to land near the")
print("  optimum of the space it was chosen from. The percentile quantifies the")
print("  multiple-comparison exposure (1365 subsets searched), it does not")
print("  distinguish 'these are genuine outliers' from 'this was optimised'.")

print()
print("=" * 76)
print("3. SENSITIVITY TO THE CAT'S T_max")
print("=" * 76)
for tm in (17.0, 30.0):
    rows = [(n, rt, tm if n == 'Cat' else t, f) for n, rt, t, f in RECORDS]
    show(f"T_max(Cat) = {tm:4.1f}, all 15", rows)
    show(f"T_max(Cat) = {tm:4.1f}, n=11",
         [r for r in rows if r[0] not in EXCLUDED])

print()
print("=" * 76)
print("4. BOOTSTRAP (10 000 resamples with replacement)")
print("=" * 76)
for tag, rows, point in (("all 15", all15, phi15), ("n=11", keep11, phi11)):
    b = []
    for _ in range(10000):
        idx = RNG.integers(0, len(rows), len(rows))
        sub = [rows[i] for i in idx]
        if len({r[2] for r in sub}) < 3:
            continue
        b.append(fit(sub)[0])
    b = np.array(b)
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"  {tag:8s} phi={point:+.4f}  95% CI [{lo:+.4f}; {hi:+.4f}]"
          f"   contains -1: {'yes' if lo <= -1 <= hi else 'NO'}")
    if tag == "all 15":
        print(f"           does the CI of the full sample contain the restricted "
              f"phi ({phi11:+.4f})? {'yes' if lo <= phi11 <= hi else 'no'}")
