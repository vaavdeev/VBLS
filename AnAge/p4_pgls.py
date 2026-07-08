# -*- coding: utf-8 -*-
"""
P4 with phylogenetic correction (taxonomic PGLS + Pagel's lambda).

No molecular tree / R available -> use the Linnaean taxonomy
(Order/Family/Genus/Species) as a proxy tree, Brownian-motion covariance,
and ML-estimate Pagel's lambda (lambda=0 == OLS, lambda=1 == full BM).
This is the standard fallback when a dated phylogeny is unavailable; it is
'phylogenetically informed', not a definitive molecular PGLS (branch lengths
are equal-per-rank). Reports OLS vs PGLS side by side so the honest p is visible.

Two models (as in p4_sigma0_mass.py):
  M1 (VBLS-native): log Rubner ~ log M     -> slope = alpha+beta ; sigma0 ∝ M^-slope
  M2 (multiple):    log T_max ~ log(P/M) + log M -> b2 ; sigma0 ∝ M^-b2
P4 predicts slope=0 (M1) and b2=0 (M2).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, 'anage_data.csv'),
                 sep=';', encoding='utf-8-sig', on_bad_lines='skip')
mam = df[df['Class'] == 'Mammalia'].copy()
for c in ['Body mass (g)', 'Maximum longevity (yrs)', 'Metabolic rate (W)']:
    mam[c] = pd.to_numeric(mam[c], errors='coerce')
mam = mam.dropna(subset=['Body mass (g)', 'Maximum longevity (yrs)', 'Metabolic rate (W)'])
mam = mam[(mam['Body mass (g)'] > 0) & (mam['Maximum longevity (yrs)'] > 0)
          & (mam['Metabolic rate (W)'] > 0)]
mam['PM']     = mam['Metabolic rate (W)'] / mam['Body mass (g)'] * 1000.0
mam['Rubner'] = mam['Maximum longevity (yrs)'] * mam['PM']

base = mam[(mam['Data quality'] != 'low') & (mam['Order'] != 'Chiroptera')].copy()
filt = mam[(mam['Data quality'] == 'acceptable')
           & (mam['Sample size'].isin(['medium', 'large', 'huge']))
           & (mam['Order'] != 'Chiroptera')].copy()


def tax_corr(g):
    """Brownian correlation matrix from taxonomy (Order/Family/Genus).
    Shared depth s_ij in {0,1,2,3} (hierarchical), diag depth=4 -> corr = C/4."""
    def codes(col):
        s = g[col].astype('object').copy()
        # make missing values unique so they never match
        miss = s.isna().values
        s = s.astype(str).values.copy()
        s[miss] = ['__UNK%d__' % i for i in np.where(miss)[0]]
        return pd.factorize(s)[0]
    o, f, ge = codes('Order'), codes('Family'), codes('Genus')
    Mo = (o[:, None] == o[None, :])
    Mf = Mo & (f[:, None] == f[None, :])
    Mg = Mf & (ge[:, None] == ge[None, :])
    s = Mo.astype(float) + Mf + Mg            # 0..3 (diagonal becomes 3)
    C = s.copy()
    np.fill_diagonal(C, 4.0)                   # root-to-tip depth
    return C / 4.0                             # correlation form, diag=1


def pgls(X, y, R):
    """GLS with V(lambda)=lambda*R_offdiag + I_diag, lambda ML-estimated."""
    n, k = X.shape
    I = np.eye(n)
    offdiag = R - np.diag(np.diag(R))          # R has diag=1

    def fit(lam):
        V = lam * offdiag + I                  # diag stays 1
        Vi = np.linalg.inv(V)
        XtVi = X.T @ Vi
        beta = np.linalg.solve(XtVi @ X, XtVi @ y)
        r = y - X @ beta
        rVir = float(r @ Vi @ r)
        sign, logdet = np.linalg.slogdet(V)
        return beta, Vi, rVir, logdet

    def negll(lam):
        lam = min(max(lam, 0.0), 1.0)
        beta, Vi, rVir, logdet = fit(lam)
        s2 = rVir / n
        return 0.5 * (n * np.log(2 * np.pi) + logdet + n * np.log(s2) + n)

    res = optimize.minimize_scalar(negll, bounds=(0.0, 1.0), method='bounded')
    lam = float(res.x)
    beta, Vi, rVir, logdet = fit(lam)
    dof = n - k
    s2 = rVir / dof
    cov = s2 * np.linalg.inv(X.T @ Vi @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return dict(lam=lam, beta=beta, se=se, t=t, p=p, dof=dof, n=n)


def ols(X, y):
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    r = y - X @ beta
    s2 = (r @ r) / (n - k)
    se = np.sqrt(np.diag(s2 * XtXi))
    p = 2 * stats.t.sf(np.abs(beta / se), n - k)
    return beta, se, p


def run(name, g):
    print('\n' + '=' * 74)
    print(f'{name}   n={len(g)}')
    print('=' * 74)
    R = tax_corr(g)
    LM = np.log10(g['Body mass (g)'].values)
    LQ = np.log10(g['PM'].values)
    LT = np.log10(g['Maximum longevity (yrs)'].values)
    LR = np.log10(g['Rubner'].values)

    # M1: log Rubner ~ log M  (slope = alpha+beta ; sigma0 ∝ M^-slope)
    X1 = np.column_stack([np.ones_like(LM), LM])
    b_ols, se_ols, p_ols = ols(X1, LR)
    g1 = pgls(X1, LR, R)
    print('\n M1  log Rubner ~ log M    (slope = alpha+beta ; P4: 0)')
    print(f'   OLS : slope={b_ols[1]:+.4f} ± {se_ols[1]:.4f}  p={p_ols[1]:.2e}'
          f'   => sigma0 ∝ M^{-b_ols[1]:+.4f}')
    print(f'   PGLS: slope={g1["beta"][1]:+.4f} ± {g1["se"][1]:.4f}  p={g1["p"][1]:.2e}'
          f'   lambda={g1["lam"]:.3f}  => sigma0 ∝ M^{-g1["beta"][1]:+.4f}')

    # M2: log T_max ~ log(P/M) + log M  (b2 ; sigma0 ∝ M^-b2)
    X2 = np.column_stack([np.ones_like(LM), LQ, LM])
    b2_ols, se2_ols, p2_ols = ols(X2, LT)
    g2 = pgls(X2, LT, R)
    print('\n M2  log T_max ~ log(P/M) + log M    (b2 [log M] ; P4: 0)')
    print(f'   OLS : b1={b2_ols[1]:+.3f}±{se2_ols[1]:.3f} (p={p2_ols[1]:.1e})  '
          f'b2={b2_ols[2]:+.3f}±{se2_ols[2]:.3f}  p={p2_ols[2]:.2e}  => sigma0 ∝ M^{-b2_ols[2]:+.3f}')
    print(f'   PGLS: b1={g2["beta"][1]:+.3f}±{g2["se"][1]:.3f} (p={g2["p"][1]:.1e})  '
          f'b2={g2["beta"][2]:+.3f}±{g2["se"][2]:.3f}  p={g2["p"][2]:.2e}  '
          f'lambda={g2["lam"]:.3f}  => sigma0 ∝ M^{-g2["beta"][2]:+.3f}')
    return g1, g2


print('P4 with taxonomic PGLS (Pagel lambda). Tree = Order/Family/Genus proxy.')
gb1, gb2 = run('BASE (quality!=low, excl Chiroptera)', base)
gf1, gf2 = run('IMPROVED (acceptable, medium+)', filt)

print('\n' + '=' * 74)
print('VERDICT (phylogenetically informed)')
print('=' * 74)
for nm, g1, g2 in [('base', gb1, gb2), ('improved', gf1, gf2)]:
    s1 = 'P4 holds' if g1['p'][1] > 0.05 else 'P4 rejected'
    s2 = 'P4 holds' if g2['p'][2] > 0.05 else 'P4 rejected'
    print(f'  {nm:9s}  M1(VBLS-native): slope={g1["beta"][1]:+.3f} p={g1["p"][1]:.2e} '
          f'(lam={g1["lam"]:.2f}) -> {s1}')
    print(f'  {nm:9s}  M2(multiple)   : b2={g2["beta"][2]:+.3f}    p={g2["p"][2]:.2e} '
          f'(lam={g2["lam"]:.2f}) -> {s2}')
