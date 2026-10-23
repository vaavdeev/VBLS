"""
=================================================================
  VBLS — Viability Boundaries of Living Systems
  Author:     Vasiliy Avdeev (independent researcher)
  Monograph:  VBLS_THEORY
=================================================================

  Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0 -- see
  LICENSE in the repository root, or https://creativecommons.org/licenses/by-nc-nd/4.0/
  SPDX-License-Identifier: CC-BY-NC-ND-4.0
=================================================================
VBLS — Robustness of the Cagan estimate in Section 17.3
=================================================================

Section 17.3 reports the regression of the somatic-mutation rate on maximum
longevity, rate ~ T_max^phi (VBLS predicts phi = -1), over all 15 species, with
T_max from AnAge for every species and no species excluded after looking at the
data. This script checks how far that estimate depends on single species and on
the reference T_max of the cat:

  1. leave-one-out over the 15 species;
  2. bootstrap confidence interval of phi;
  3. sensitivity to the cat's T_max: 30 yr (AnAge, a captive record) against
     19 yr (the median of the literature maxima compiled by Cagan et al. 2022,
     Supplementary Table 6) and 17 yr.

Rates (mutations/genome/yr) are the per-species slopes used in
cagan_rateXtmax.py: OLS of the individual-mean burden on age (Cagan et al. 2022,
Supplementary Table 4); species represented by a single individual keep their
externally estimated slopes.

Run: python cagan_robustness.py
"""
import csv
import os

import numpy as np
from scipy import stats

ANAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'AnAge', 'anage_data.csv')

# species -> (mutation rate per year, AnAge genus, AnAge species)
SPECIES = {
    'Rat':                     (607.12, 'Rattus', 'norvegicus'),
    'Mouse':                   (761.50, 'Mus', 'musculus'),
    'Ferret':                  (485.10, 'Mustela', 'putorius'),
    'Rabbit':                  ( 97.54, 'Oryctolagus', 'cuniculus'),
    'Cow':                     (220.40, 'Bos', 'taurus'),
    'Tiger':                   (366.76, 'Panthera', 'tigris'),
    'Dog':                     (221.77, 'Canis', 'familiaris'),
    'Lion':                    (160.17, 'Panthera', 'leo'),
    'Cat':                     (546.63, 'Felis', 'catus'),
    'Naked mole-rat':          (102.64, 'Heterocephalus', 'glaber'),
    'Black-and-white colobus': ( 78.97, 'Colobus', 'guereza'),
    'Ring-tailed lemur':       (101.96, 'Lemur', 'catta'),
    'Giraffe':                 (100.25, 'Giraffa', 'camelopardalis'),
    'Horse':                   (200.92, 'Equus', 'caballus'),
    'Human':                   ( 53.24, 'Homo', 'sapiens'),
}
RNG = np.random.default_rng(20260730)


def anage_tmax(path=ANAGE_FILE):
    with open(path, encoding='utf-8-sig') as f:
        rows = list(csv.reader(f, delimiter=';'))
    h = rows[0]
    ig, isp, iml = h.index('Genus'), h.index('Species'), h.index('Maximum longevity (yrs)')
    out = {}
    for name, (_, genus, species) in SPECIES.items():
        hit = [r for r in rows[1:] if r[ig] == genus and r[isp] == species]
        if len(hit) != 1:
            raise SystemExit(f'AnAge: {genus} {species} found {len(hit)} times')
        out[name] = float(hit[0][iml])
    return out


TMAX = anage_tmax()
RECORDS = [(name, SPECIES[name][0], TMAX[name]) for name in SPECIES]


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
print("0. THE ESTIMATE OVER ALL 15 SPECIES (T_max from AnAge)")
print("=" * 76)
for name, rate, tm in RECORDS:
    print(f"  {name:26s} rate={rate:8.2f}  T_max={tm:6.1f}")
phi15, r215 = show("all 15 species", RECORDS)

print()
print("=" * 76)
print("1. LEAVE-ONE-OUT (n=15)")
print("=" * 76)
print(f"  {'excluded':26s} {'phi':>9} {'R^2':>8} {'d(phi)':>9}")
print("  " + "-" * 56)
loo = []
for i, r in enumerate(RECORDS):
    sub = RECORDS[:i] + RECORDS[i + 1:]
    p, q = fit(sub)
    loo.append(p)
    print(f"  {r[0]:26s} {p:+9.4f} {q:8.4f} {p - phi15:+9.4f}")
loo = np.array(loo)
print(f"\n  range [{loo.min():+.4f}; {loo.max():+.4f}], sd={loo.std():.4f}")
worst = RECORDS[int(np.argmax(np.abs(loo - phi15)))][0]
print(f"  most influential species: {worst}")

print()
print("=" * 76)
print("2. BOOTSTRAP (10 000 resamples with replacement)")
print("=" * 76)
b = []
for _ in range(10000):
    idx = RNG.integers(0, len(RECORDS), len(RECORDS))
    sub = [RECORDS[i] for i in idx]
    if len({r[2] for r in sub}) < 3:
        continue
    b.append(fit(sub)[0])
b = np.array(b)
lo, hi = np.percentile(b, [2.5, 97.5])
print(f"  phi={phi15:+.4f}  95% CI [{lo:+.4f}; {hi:+.4f}]"
      f"   contains -1: {'yes' if lo <= -1 <= hi else 'NO'}")

print()
print("=" * 76)
print("3. SENSITIVITY TO THE CAT'S T_max")
print("=" * 76)
for tm in (30.0, 19.0, 17.0):
    rows = [(n, rt, tm if n == 'Cat' else t) for n, rt, t in RECORDS]
    show(f"T_max(Cat) = {tm:4.1f}, all 15", rows)
