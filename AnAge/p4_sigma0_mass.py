# -*- coding: utf-8 -*-
"""
P4 test (РЕПАРАЦИЯ.md §10.8 / §11): is sigma0_eff independent of body mass M
after controlling for metabolism?

VBLS scaling:  dSigma/dt ~ sigma0 * (P/M)  =>  T_max ~ 1/(sigma0 * (P/M))
            => sigma0_eff ∝ 1/(T_max * (P/M)) = 1/Rubner.

Fixed-point theory (Eigen x Kirkwood, §10-11): sigma0* is N-independent, and for
unitary organisms N_struct ∝ M, so sigma0_eff should NOT depend on M.

Primary test — multiple OLS:  log T_max = b0 + b1*log(P/M) + b2*log(M).
  Derivation: log sigma0 = const - (b1+1) log(P/M) - b2 log M.
  => at fixed metabolism (P/M):  sigma0_eff ∝ M^(-b2).
  P4  <=>  b2 = 0    (joint VBLS+P4: also b1 = -1).

Cross-checks: simple slope(log Rubner ~ log M) = alpha+beta (= sigma0 ∝ M^-(a+b)
under gamma=-1); per-order breakdown; Primates-excluded robustness.

Reads anage_data.csv (semicolon). Does NOT overwrite anage.py outputs.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- load & filter
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, 'anage_data.csv'),
                 sep=';', encoding='utf-8-sig', on_bad_lines='skip')
mam = df[df['Class'] == 'Mammalia'].copy()
for c in ['Body mass (g)', 'Maximum longevity (yrs)', 'Metabolic rate (W)']:
    mam[c] = pd.to_numeric(mam[c], errors='coerce')
mam = mam.dropna(subset=['Body mass (g)', 'Maximum longevity (yrs)', 'Metabolic rate (W)'])
mam = mam[(mam['Body mass (g)'] > 0) & (mam['Maximum longevity (yrs)'] > 0)
          & (mam['Metabolic rate (W)'] > 0)]
mam['PM']     = mam['Metabolic rate (W)'] / mam['Body mass (g)'] * 1000.0   # W/kg
mam['Rubner'] = mam['Maximum longevity (yrs)'] * mam['PM']                  # W*yr/kg
mam['sigma0_eff'] = mam['Rubner'].median() / mam['Rubner']                  # ∝1/Rubner, median=1

base = mam[(mam['Data quality'] != 'low') & (mam['Order'] != 'Chiroptera')].copy()
filt = mam[(mam['Data quality'] == 'acceptable')
           & (mam['Sample size'].isin(['medium', 'large', 'huge']))
           & (mam['Order'] != 'Chiroptera')].copy()


# ------------------------------------------------------------------ OLS helpers
def ols(X, y):
    """Manual OLS. X already includes intercept column. Returns dict."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    dof = n - k
    s2 = (resid @ resid) / dof
    cov = s2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    tc = stats.t.ppf(0.975, dof)
    ci = np.array([beta - tc * se, beta + tc * se]).T
    sstot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - resid @ resid / sstot
    return dict(beta=beta, se=se, t=t, p=p, ci=ci, r2=r2, n=n, dof=dof)


def simple_slope(x, y):
    sl, ic, r, p, se = stats.linregress(x, y)
    return sl, se, r ** 2, p


def report(name, d):
    lo = np.log10
    print(f'\n  === {name}  (n={d["n"]}) ===')
    g = d['sample']
    LM, LQ, LT = lo(g['Body mass (g)'].values), lo(g['PM'].values), lo(g['Maximum longevity (yrs)'].values)

    a, a_se, a_r2, a_p = simple_slope(LM, LT)            # T_max ~ M^alpha
    b, b_se, b_r2, b_p = simple_slope(LM, LQ)            # P/M  ~ M^beta
    g3, g3_se, g3_r2, g3_p = simple_slope(LQ, LT)        # T_max ~ (P/M)^gamma
    rub_sl, rub_se, rub_r2, rub_p = simple_slope(LM, lo(g['Rubner'].values))  # = alpha+beta

    print(f'  alpha (T_max~M)   = {a:+.3f} ± {a_se:.3f}   (VBLS +0.250)')
    print(f'  beta  (P/M ~M)    = {b:+.3f} ± {b_se:.3f}   (Kleiber -0.250)')
    print(f'  gamma (T_max~P/M) = {g3:+.3f} ± {g3_se:.3f}  (VBLS -1.000)')
    print(f'  --- P4 simple: slope(log Rubner ~ log M) = alpha+beta = {rub_sl:+.4f} ± {rub_se:.4f}'
          f'  (p={rub_p:.2e})')
    print(f'      => sigma0_eff ∝ M^({-rub_sl:+.4f})   [P4: exponent 0]')

    # multiple regression: log T_max = b0 + b1 log(P/M) + b2 log(M)
    X = np.column_stack([np.ones_like(LQ), LQ, LM])
    m = ols(X, LT)
    b0, b1, b2 = m['beta']
    print(f'  --- P4 primary (multiple OLS: log T_max ~ log(P/M) + log M), R²={m["r2"]:.3f}')
    print(f'      b1 [log P/M] = {b1:+.3f} ± {m["se"][1]:.3f}  CI[{m["ci"][1,0]:+.3f},{m["ci"][1,1]:+.3f}]'
          f'  p={m["p"][1]:.2e}   (VBLS -1.000)')
    print(f'      b2 [log M]   = {b2:+.3f} ± {m["se"][2]:.3f}  CI[{m["ci"][2,0]:+.3f},{m["ci"][2,1]:+.3f}]'
          f'  p={m["p"][2]:.2e}   (P4: 0)')
    print(f'      => sigma0_eff ∝ M^({-b2:+.3f}) at fixed metabolism   [P4: exponent 0]')
    rqm = np.corrcoef(LQ, LM)[0, 1]
    print(f'      collinearity r(log P/M, log M) = {rqm:+.3f}')
    return dict(a=a, b=b, ab=rub_sl, ab_se=rub_se, b2=b2, b2_se=m['se'][2], b2_p=m['p'][2])


print('=' * 74)
print('P4 TEST — does sigma0_eff depend on body mass M? (AnAge Build 14, Mammalia)')
print('=' * 74)
print(f'Total Mammalia with M,T_max,P: {len(mam)}   base(excl low,Chiroptera)={len(base)}'
      f'   improved={len(filt)}')

rb = report('BASE sample', dict(sample=base, n=len(base)))
rf = report('IMPROVED sample (acceptable, medium+)', dict(sample=filt, n=len(filt)))

# robustness: drop Primates (known Rubner outlier) from base
nop = base[base['Order'] != 'Primates'].copy()
rn = report('BASE minus Primates', dict(sample=nop, n=len(nop)))

# per-order P4 slope (Rubner ~ M) for orders with n>=8
print('\n  === Per-order: slope(log Rubner ~ log M) [= sigma0 ∝ M^-slope] ===')
for order, grp in sorted(base.groupby('Order'), key=lambda x: -len(x[1])):
    if len(grp) >= 8:
        LM = np.log10(grp['Body mass (g)'].values)
        LR = np.log10(grp['Rubner'].values)
        sl, se, r2, p = simple_slope(LM, LR)
        print(f'  {order:16s} n={len(grp):3d}: slope={sl:+.3f} ± {se:.3f}  p={p:.2e}  R²={r2:.2f}')

print('\n' + '=' * 74)
print('VERDICT')
print('=' * 74)
for nm, r in [('base', rb), ('improved', rf), ('base-noPrimates', rn)]:
    sig = 'NOT sig (P4 ✓)' if r['b2_p'] > 0.05 else 'SIGNIFICANT (P4 ✗)'
    print(f'  {nm:18s}: residual M-coeff b2 = {r["b2"]:+.3f} ± {r["b2_se"]:.3f}'
          f'  (p={r["b2_p"]:.2e}) -> {sig};  sigma0 ∝ M^({-r["b2"]:+.3f})')
print('\nP4 predicts b2 = 0 (sigma0_eff independent of M after metabolism control).')
