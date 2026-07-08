"""
=================================================================
  VBLS — Whittemore (2019) telomere verification: SMALL-SAMPLE ROBUSTNESS
  Addresses reviewer point RV-3 (§17.2):
    R2_log = 0.976, MAPE = 12.7% reported on n=7 (cluster 1, R-channel)
    -> add bootstrap 95% CI, leave-one-out, and leverage diagnostic.

  Model (Variant B): t_B = (1 - R*_univ) / nu* ,
    R*_univ = mean residual telomere fraction over the cluster (1 fitted scalar),
    nu*     = fractional telomere shortening rate, taken directly from data.
  => effectively ONE fitted parameter against 7 points; nu* is not fitted.

  Author: Vasiliy Avdeev (independent researcher). Data: Whittemore et al. 2019 PNAS.
=================================================================
"""
import numpy as np
from scipy.stats import pearsonr

rng = np.random.default_rng(20260616)

# Whittemore 2019, Table S1+S2 (9 species):
#   name, L0_kb, rate_bp_yr, T_max(yr), T_avg(yr), residual_%_at_death
raw = [
    ("Mouse",              50.0, 7000,   4.0,  2.0, 44.0),
    ("Goat",               10.4,  363,  20.8, 16.5, 27.4),
    ("Audouin's gull",     35.0,  771,  25.0, 21.0, 44.9),
    ("Reindeer",           19.8,  531,  21.8, 15.0, 41.5),
    ("Griffon vulture",    19.8,  209,  41.4, 37.0, 56.3),
    ("Bottlenose dolphin", 90.7,  766,  51.6, 17.0, 56.4),
    ("American flamingo",  21.0,  105,  60.0, 40.0, 70.0),
    ("Sumatran elephant",  36.3,  109,  65.5, 60.0, 80.3),
    ("Human",              15.0,   71, 122.5, 79.0, 42.0),
]
name = np.array([r[0] for r in raw])
L0   = np.array([r[1] for r in raw], float)
rate = np.array([r[2] for r in raw], float)
Tmax = np.array([r[3] for r in raw], float)
res  = np.array([r[5] for r in raw], float) / 100.0
nu   = rate / (L0 * 1000.0)            # fractional loss per year

# Cluster 1 (R-channel): residual < 0.60  -> n = 7
m   = res < 0.60
nm, Tm, num, rm = name[m], Tmax[m], nu[m], res[m]
n   = int(m.sum())


def fit_eval(nu_sub, T_sub, res_sub):
    """R*_univ = mean residual; t_B = (1-R*)/nu*; return (R*, MAPE%, R2_log)."""
    Rstar = res_sub.mean()
    tpred = (1.0 - Rstar) / nu_sub
    mape  = np.mean(np.abs(tpred - T_sub) / T_sub) * 100.0
    if len(np.unique(T_sub)) < 2 or len(np.unique(tpred)) < 2:
        return Rstar, mape, np.nan
    r, _ = pearsonr(np.log(tpred), np.log(T_sub))
    return Rstar, mape, r * r


# ── Point estimate, n = 7 ─────────────────────────────────────────────────
R0, mape0, r2_0 = fit_eval(num, Tm, rm)
print("=" * 64)
print(f"POINT ESTIMATE  cluster 1, n={n}")
print(f"  R*_univ = {R0:.3f}    MAPE = {mape0:.1f}%    R2_log = {r2_0:.3f}")
print("=" * 64)

# ── Leave-one-out (recompute R* on retained 6, evaluate on retained 6) ─────
print("\nLEAVE-ONE-OUT  (drop one species, refit on remaining 6):")
loo_r2, loo_mape = [], []
for i in range(n):
    keep = np.arange(n) != i
    _, mp, r2 = fit_eval(num[keep], Tm[keep], rm[keep])
    loo_r2.append(r2); loo_mape.append(mp)
    print(f"  -{nm[i]:<20s} R2_log={r2:.3f}  MAPE={mp:.1f}%")
loo_r2, loo_mape = np.array(loo_r2), np.array(loo_mape)
print(f"  range: R2_log [{np.nanmin(loo_r2):.3f}, {np.nanmax(loo_r2):.3f}]  "
      f"MAPE [{np.nanmin(loo_mape):.1f}%, {np.nanmax(loo_mape):.1f}%]")

# ── Nonparametric bootstrap (resample 7 with replacement) ──────────────────
B = 20000
b_r2, b_mape = [], []
ndegen = 0
for _ in range(B):
    idx = rng.integers(0, n, n)
    _, mp, r2 = fit_eval(num[idx], Tm[idx], rm[idx])
    if np.isnan(r2):
        ndegen += 1
        continue
    b_r2.append(r2); b_mape.append(mp)
b_r2, b_mape = np.array(b_r2), np.array(b_mape)

def pctl(a):
    return np.percentile(a, [2.5, 50, 97.5])

print(f"\nBOOTSTRAP  (B={B}, {ndegen} degenerate resamples with <2 distinct pts skipped):")
lo, md, hi = pctl(b_r2)
print(f"  R2_log : median={md:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
lo, md, hi = pctl(b_mape)
print(f"  MAPE   : median={md:.1f}%   95% CI [{lo:.1f}%, {hi:.1f}%]")

# ── Leverage diagnostic: endpoints in log-log are Mouse (fast) & Human (slow)
order = np.argsort(num)
fast, slow = order[-1], order[0]
print("\nLEVERAGE  (log-log endpoints dominate the range):")
print(f"  fastest nu* : {nm[fast]:<20s} nu*={num[fast]:.4f}  T_max={Tm[fast]:.1f}")
print(f"  slowest nu* : {nm[slow]:<20s} nu*={num[slow]:.4f}  T_max={Tm[slow]:.1f}")
keep2 = np.ones(n, bool); keep2[fast] = False; keep2[slow] = False
_, mp2, r22 = fit_eval(num[keep2], Tm[keep2], rm[keep2])
print(f"  drop BOTH endpoints (n={int(keep2.sum())}, mid-range only): "
      f"R2_log={r22:.3f}  MAPE={mp2:.1f}%")
print("=" * 64)
