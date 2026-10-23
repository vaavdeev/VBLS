"""
VBLS -- Appendix D.5.8: cross-clade test of the structural law a = 1 - chi
=========================================================================

The structural law of VBLS (Section 14.2) states that the scaling exponent of
maximum longevity with body mass is the complement of the metabolic scaling
exponent to unity:

    T_max ~ M^a,  Phi ~ M^chi,  and the theory requires  a = 1 - chi,
    equivalently  a + chi - 1 = 0.

The prediction holds for ANY value of chi, so the documented variation of chi
between clades and activity levels (Dodds et al. 2001; White & Seymour 2003;
Savage et al. 2004; Glazier 2005, 2010) turns the dispute over the Kleiber
exponent into a test rather than an obstacle: a clade with lower chi must have
proportionally higher a, and the slope of a on chi across clades must be -1.

The test needs only M, T_max and Phi -- no species-specific nu_0 -- and is
therefore runnable on the whole AnAge database, not only on Mammalia.

Data: anage_data.csv (AnAge Build 14, semicolon-delimited)
Run:  python chi_a_crossclade.py
"""
import numpy as np
import pandas as pd
from scipy import stats

MIN_N_CLADE = 10      # below this a clade is reported but carries no weight
MIN_N_ORDER = 15


def load():
    d = pd.read_csv('anage_data.csv', sep=';', encoding='utf-8-sig')
    for col, src in (('M', 'Body mass (g)'),
                     ('Tmax', 'Maximum longevity (yrs)'),
                     ('Phi', 'Metabolic rate (W)')):
        d[col] = pd.to_numeric(d[src], errors='coerce')
    return d[(d['M'] > 0) & (d['Tmax'] > 0) & (d['Phi'] > 0) &
             (d['Data quality'] != 'low')].copy()


def exponents(g):
    """Return (chi, se_chi, a, se_a) by OLS in log10 coordinates."""
    lM = np.log10(g['M'].values)
    chi = stats.linregress(lM, np.log10(g['Phi'].values))
    a = stats.linregress(lM, np.log10(g['Tmax'].values))
    return chi.slope, chi.stderr, a.slope, a.stderr


def report(g, label):
    chi, se_chi, a, se_a = exponents(g)
    dev = a + chi - 1.0
    # upper bound on the error of the sum: the two slopes share the predictor,
    # so their errors are correlated; quadrature is conservative here.
    se = np.hypot(se_chi, se_a)
    decades = np.ptp(np.log10(g['M'].values))
    print(f"{label:26s} {len(g):>4}  {chi:+.4f}+-{se_chi:.4f}"
          f"  {a:+.4f}+-{se_a:.4f}   {1 - chi:+.4f}"
          f"   {dev:+.4f}+-{se:.4f}  {abs(dev) / se:>4.1f}s   {decades:>4.1f}")
    return chi, a


def main():
    base = load()

    print("Species with M, T_max and metabolic rate, quality != low:")
    print(base.groupby('Class').size().sort_values(ascending=False).to_string())

    print(f"\n{'clade':26s} {'n':>4}  {'chi (Phi~M)':>15}  {'a (Tmax~M)':>16}"
          f"   {'1-chi':>7}   {'a+chi-1':>15}  {'dev':>5}   {'dec':>4}")
    print("-" * 112)

    clade_pts = []
    for cl in ['Mammalia', 'Aves', 'Reptilia', 'Amphibia']:
        g = base[base['Class'] == cl]
        if len(g) >= MIN_N_CLADE:
            clade_pts.append(report(g, cl))

    mam = base[base['Class'] == 'Mammalia']
    report(mam[mam['Order'] != 'Chiroptera'], 'Mammalia w/o Chiroptera')

    print(f"\n{'Mammalian order':26s} {'n':>4}  {'chi':>9}  {'a':>9}  {'a+chi-1':>9}")
    print("-" * 66)
    order_pts = []
    for name, g in mam.groupby('Order'):
        if len(g) < MIN_N_ORDER:
            continue
        chi, _, a, _ = exponents(g)
        order_pts.append((chi, a))
        print(f"{name:26s} {len(g):>4}  {chi:+9.4f}  {a:+9.4f}  {a + chi - 1:+9.4f}")

    print("\n=== covariance of a with chi (VBLS requires slope -1) ===")
    for pts, what in ((clade_pts, 'classes'), (order_pts, 'Mammalian orders')):
        if len(pts) < 3:
            continue
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        r = stats.linregress(x, y)
        print(f"  {what:18s} n={len(pts)}:  a ~ {r.slope:+.3f}*chi"
              f"   r={r.rvalue:+.3f}  p={r.pvalue:.3f}   (VBLS: -1.000)")
    if len(clade_pts) < 5:
        print("  note: with four classes the slope is not determined; read the sign only.")

    print("\nNo phylogenetic correction is applied here: trees were built for "
          "Mammalia only (Section D.5.7).")


if __name__ == '__main__':
    main()
