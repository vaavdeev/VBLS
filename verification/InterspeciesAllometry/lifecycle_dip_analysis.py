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
VBLS — Analysis of the initial dip in E(t): physics and two Phi_in variants

Both variants are shown on the same axes for each species:
  Red:  Phi_in(0)=0    — perinatal resource deficit
  Blue: Phi_in(0)=0.45 — live-bearing species with a maternal resource

A separate, larger panel for Human gives a detailed dip analysis.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

plt.rcParams.update({
    'font.family':       'DejaVu Serif',
    'font.size':         9,
    'axes.titlesize':    9.5,
    'axes.labelsize':    8.5,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.15,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
})

T_REF     = 76.9
PHI_BIRTH = 0.45

BASE = dict(mu=1.50, lam=0.30, sigma0=0.0162, alpha=0.030,
            nu0=0.006, nu=0.060, rho=0.002, R_star=0.15, E_star=0.04)
Y0 = [0.10, 0.01, 0.96, 0.99]

SPECIES = [
    ("Mouse",       4.0,   2.0),
    ("Rat",         3.8,   2.5),
    ("Rabbit",     13.0,   8.0),
    ("Dog",        24.0,  12.0),
    ("Cat",        30.0,  15.0),
    ("NMR",        31.0,  20.0),
    ("Pig",        27.0,  15.0),
    ("Horse",      57.0,  30.0),
    ("Chimpanzee", 59.0,  40.0),
    ("Elephant",   86.0,  60.0),
    ("Human",     122.5,  76.9),
]

def Phi_in_A(t_h):
    """Phi_in(0) = 0: canonical shape of Section 17.1.1 (argument in human years)."""
    return (1.0 - np.exp(-t_h / 10.0)) * np.exp(-max(0.0, t_h - 60.0) / 30.0)

def Phi_in_B(t_h):
    """Phi_in(0) = PHI_BIRTH: an initial maternal flow, otherwise the canonical shape."""
    g = PHI_BIRTH + (1.0 - PHI_BIRTH) * (1.0 - np.exp(-t_h / 10.0))
    return g * np.exp(-max(0.0, t_h - 60.0) / 30.0)

def scale(T_ls):
    """Consistent time rescaling (Appendix D.3.1): every rate (mu, lam, the amplitude of
    Phi_in, Phi_cap, alpha, nu0, nu, rho) times sc = T_REF/T; sigma0 (a fraction, not a
    rate), thresholds and initial conditions unchanged. Each species is the human
    calibration of Section 17.1.1 in its own time scale."""
    sc = T_REF / T_ls
    return dict(sc=sc, mu=BASE['mu']*sc, lam=BASE['lam']*sc,
                sigma0=BASE['sigma0'], alpha=BASE['alpha']*sc,
                nu0=BASE['nu0']*sc, nu=BASE['nu']*sc, rho=BASE['rho']*sc,
                Phi_cap=0.003*sc, R_star=BASE['R_star'], E_star=BASE['E_star'])

def system(t, y, p, T_ls, phi_f):
    E,S,eta,R = y
    E=max(E,0.); S=np.clip(S,0.,1.); eta=np.clip(eta,0.,1.); R=np.clip(R,0.,1.)
    d = p['mu']*E**2 + p['lam']*E
    g_E = max(0.0, min(1.0, (E - p['E_star'])/(Y0[0] - p['E_star'])))
    return [eta*p['sc']*phi_f(p['sc']*t) - d,
            max(0., p['sigma0']*d - R*p['Phi_cap']),
            -p['alpha']*S*eta,
            -p['nu0']*R - p['nu']*S*R + p['rho']*g_E*(1-S)*(1-R)]

def run(T_ls, phi_f):
    p = scale(T_ls)
    def ev(t,y,p,T,phi): return min(y[3]-p['R_star'], y[0]-p['E_star'])
    ev.terminal=True; ev.direction=-1
    sol = solve_ivp(system,[0,T_ls*1.15],Y0,args=(p,T_ls,phi_f),
                    method='RK45',t_eval=np.linspace(0,T_ls*1.15,5000),
                    events=ev,rtol=1e-8,atol=1e-10,max_step=T_ls/500)
    td = sol.t_events[0][0] if sol.t_events[0].size>0 else sol.t[-1]
    mask = sol.t <= td
    return sol.t[mask], sol.y[:,mask], td

# ── Computation ──────────────────────────────────────────────────────────────
print("Computing curves...")
res = {}
for name, Tmax, Tls in SPECIES:
    tA,yA,tdA = run(Tls, Phi_in_A)
    tB,yB,tdB = run(Tls, Phi_in_B)
    # Find the depth of the dip in E
    early_A = yA[0,:max(10, int(len(tA)*0.12))]
    dip_val = early_A.min()
    dip_depth = Y0[0] - dip_val
    dip_t_idx = np.argmin(early_A)
    dip_t = tA[dip_t_idx]
    # Depth as a fraction of the starting value: the form that is comparable
    # with the neonatal body-mass loss reported in the physiological literature.
    dip_frac = dip_depth / Y0[0]
    res[name] = dict(tA=tA,yA=yA,tdA=tdA,tB=tB,yB=yB,tdB=tdB,
                     Tls=Tls,Tmax=Tmax,dip_depth=dip_depth,dip_frac=dip_frac,
                     dip_t=dip_t)
    print(f"  {name:<14} T_ls={Tls:5.1f}y  "
          f"t_dip={dip_t:.3f}y  dE={dip_depth:.4f} ({dip_frac*100:.1f}% of E0)  "
          f"t_B(A)={tdA:.1f}y  t_B(B)={tdB:.1f}y  Δ={tdB-tdA:+.2f}y")

# ── FIGURE 1: All species, E(t) for both variants ────────────────────────────
BG  = '#f7f9fc'
CA  = '#e74c3c'   # variant A (dip) — red
CB  = '#2980b9'   # variant B (no dip) — blue
CRa = '#e74c3c'   # R(t) variant A — same red, dashed
CRb = '#2980b9'   # R(t) variant B — blue dashed

n=len(SPECIES); ncols=6; nrows=(n+ncols-1)//ncols

fig1, axes = plt.subplots(nrows, ncols, figsize=(22, 8), facecolor=BG)
axs = axes.flatten()

for idx,(name,Tmax,Tls) in enumerate(SPECIES):
    r  = res[name]
    ax = axs[idx]; ax.set_facecolor(BG)

    # E(t): both variants
    ax.plot(r['tA'], r['yA'][0], color=CA, lw=2.0, alpha=0.9)
    ax.plot(r['tB'], r['yB'][0], color=CB, lw=2.0, alpha=0.9)
    # R(t): both variants (thinner)
    ax.plot(r['tA'], r['yA'][3], color=CA, lw=1.2, ls='--', alpha=0.5)
    ax.plot(r['tB'], r['yB'][3], color=CB, lw=1.2, ls='--', alpha=0.5)
    # Death verticals
    ax.axvline(r['tdA'], color=CA, ls=':', lw=1.0, alpha=0.6)
    ax.axvline(r['tdB'], color=CB, ls=':', lw=1.0, alpha=0.6)
    # R* line
    ax.axhline(BASE['R_star'], color='#aaa', ls=':', lw=0.8)

    # Dip annotation (downward arrow)
    if r['dip_depth'] > 0.003:
        ax.annotate('',
            xy =(r['dip_t'], Y0[0]-r['dip_depth']),
            xytext=(r['dip_t'], Y0[0]),
            arrowprops=dict(arrowstyle='->', color=CA, lw=1.3))

    dt = r['tdB'] - r['tdA']
    ax.set_title(
        f'{name}  T_ls={Tls:.0f}y\n'
        f'Dip:{r["dip_frac"]*100:.1f}% of E0  t_dip={r["dip_t"]:.2f}y  '
        f'Δt_B={dt:+.1f}y',
        fontsize=7.5, pad=2
    )
    ax.set_xlim(0, max(r['tdA'],r['tdB'])*1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('yr', fontsize=7)
    if idx==0:
        from matplotlib.lines import Line2D
        leg = [Line2D([0],[0],color=CA,lw=2,label=f'$\\Phi(0)=0$ (dip)'),
               Line2D([0],[0],color=CB,lw=2,label=f'$\\Phi(0)={PHI_BIRTH}$'),
               Line2D([0],[0],color='gray',lw=1.2,ls='--',label='$R(t)$')]
        ax.legend(handles=leg,fontsize=6.5,framealpha=0.85,loc='center right')

for idx in range(n, nrows*ncols):
    axs[idx].set_visible(False)

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('lifecycle_dip_allspecies.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
print("\nFig. 1 saved: lifecycle_dip_allspecies.png")

# ── FIGURE 2: Detailed panel — Human ──────────────────────────────────────────
fig2 = plt.figure(figsize=(16, 10), facecolor=BG)
gs = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.38)

r = res['Human']
Tls = r['Tls']

# ── Panel A: Full E(t) curve ──────────────────────────────────────────────────
ax1 = fig2.add_subplot(gs[0, :2])
ax1.set_facecolor(BG)
ax1.plot(r['tA'], r['yA'][0], color=CA, lw=2.2, label=f'$\\Phi_{{in}}(0)=0$ — dip')
ax1.plot(r['tB'], r['yB'][0], color=CB, lw=2.2, label=f'$\\Phi_{{in}}(0)={PHI_BIRTH}$ — no dip')
ax1.axvline(r['tdA'], color=CA, ls='--', lw=1.3, alpha=0.7,
            label=f"Death A: {r['tdA']:.1f} yr")
ax1.axvline(r['tdB'], color=CB, ls='--', lw=1.3, alpha=0.7,
            label=f"Death B: {r['tdB']:.1f} yr")
ax1.axhline(BASE['E_star'], color='gray', ls=':', lw=1, alpha=0.6)

if r['dip_depth'] > 0.003:
    ax1.annotate(
        f'Dip: −{r["dip_frac"]*100:.1f}% of $E_0$\nt_dip = {r["dip_t"]:.1f} yr',
        xy=(r['dip_t'], Y0[0]-r['dip_depth']),
        xytext=(r['dip_t']+5, Y0[0]-r['dip_depth']-0.01),
        fontsize=9, color=CA,
        arrowprops=dict(arrowstyle='->', color=CA, lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', fc='#fff0ee', ec=CA, lw=0.8)
    )

# Phase shading
t_peak_A = r['tA'][np.argmax(r['yA'][0])]
ax1.axvspan(0, t_peak_A, alpha=0.05, color='green')
E_top = 1.08 * max(r['yA'][0].max(), r['yB'][0].max())
ax1.text(t_peak_A/2, 0.92*E_top, 'Phase I\n(growth)', ha='center',
         fontsize=8, color='green', alpha=0.7)

ax1.set_title('A.  Integrity potential $E(t)$ — full lifespan', fontsize=10)
ax1.set_xlabel('Age (yr)'); ax1.set_ylabel('$E(t)$')
ax1.legend(fontsize=8.5, framealpha=0.93)
ax1.set_ylim(0, E_top)

# ── Panel B: Zoom on the early period ──────────────────────────────────────────
ax2 = fig2.add_subplot(gs[0, 2])
ax2.set_facecolor(BG)
zoom_end = Tls * 0.18   # first 18% of life
mA = r['tA'] <= zoom_end
mB = r['tB'] <= zoom_end
ax2.plot(r['tA'][mA], r['yA'][0][mA], color=CA, lw=2.2)
ax2.plot(r['tB'][mB], r['yB'][0][mB], color=CB, lw=2.2)
ax2.axhline(Y0[0], color='gray', ls=':', lw=1, alpha=0.5, label='$E_0$')
ax2.axhline(BASE['E_star'], color='gray', ls='--', lw=0.8, alpha=0.4, label='$E^*$')

# Phi_in on the right-hand axis
ax2r = ax2.twinx()
t_zoom = np.linspace(0, zoom_end, 500)
phiA_vals = np.array([Phi_in_A(tt*T_REF/Tls) for tt in t_zoom])
phiB_vals = np.array([Phi_in_B(tt*T_REF/Tls) for tt in t_zoom])
ax2r.plot(t_zoom, phiA_vals, color=CA, lw=1.3, ls=':', alpha=0.6)
ax2r.plot(t_zoom, phiB_vals, color=CB, lw=1.3, ls=':', alpha=0.6)
ax2r.set_ylabel('$\\Phi_{in}(t)$', fontsize=8, color='gray')
ax2r.tick_params(axis='y', colors='gray', labelsize=7)

if r['dip_depth'] > 0.003:
    ax2.annotate(f'Minimum\nt={r["dip_t"]:.1f}y',
                xy=(r['dip_t'], Y0[0]-r['dip_depth']),
                xytext=(r['dip_t']+4, Y0[0]-r['dip_depth']-0.004),
                fontsize=8, color=CA,
                arrowprops=dict(arrowstyle='->', color=CA, lw=1))

ax2.set_title('B.  Zoom: early life', fontsize=10)
ax2.set_xlabel('Age (yr)'); ax2.set_ylabel('$E(t)$')
ax2.legend(fontsize=8, framealpha=0.9, loc='lower right')

# ── Panel C: R(t) for both variants ──────────────────────────────────────────
ax3 = fig2.add_subplot(gs[1, :2])
ax3.set_facecolor(BG)
ax3.plot(r['tA'], r['yA'][3], color=CA, lw=2.0, label='$R(t)$, $\\Phi(0)=0$')
ax3.plot(r['tB'], r['yB'][3], color=CB, lw=2.0, label=f'$R(t)$, $\\Phi(0)={PHI_BIRTH}$')
ax3.plot(r['tA'], r['yA'][2], color=CA, lw=1.5, ls='--', alpha=0.6,
         label='$\\eta_{{abs}}(t)$, $\\Phi(0)=0$')
ax3.plot(r['tB'], r['yB'][2], color=CB, lw=1.5, ls='--', alpha=0.6,
         label=f'$\\eta_{{abs}}(t)$, $\\Phi(0)={PHI_BIRTH}$')
ax3.axhline(BASE['R_star'], color='gray', ls=':', lw=1.2, alpha=0.7, label='$R^*$')
ax3.set_title('C.  Repair reserve $R(t)$ and efficiency $\\eta_{abs}(t)$', fontsize=10)
ax3.set_xlabel('Age (yr)'); ax3.set_ylabel('Value')
ax3.legend(fontsize=8.5, framealpha=0.93, ncol=2)
ax3.set_ylim(0, 1.05)

# ── Panel D: Phi_in(t) — visual comparison ───────────────────────────────────
ax4 = fig2.add_subplot(gs[1, 2])
ax4.set_facecolor(BG)
t_full = np.linspace(0, Tls*1.0, 1000)
phiA_full = np.array([Phi_in_A(tt*T_REF/Tls) for tt in t_full])
phiB_full = np.array([Phi_in_B(tt*T_REF/Tls) for tt in t_full])
ax4.plot(t_full, phiA_full, color=CA, lw=2.2,
         label=f'$\\Phi_{{in}}(0)=0$\n(oviparous, seeds)')
ax4.plot(t_full, phiB_full, color=CB, lw=2.2,
         label=f'$\\Phi_{{in}}(0)={PHI_BIRTH}$\n(viviparous)')
ax4.fill_between(t_full, phiA_full, phiB_full, alpha=0.12, color='purple',
                 label='Difference')
ax4.set_title('D.  Shape of $\\Phi_{in}(t)$, both variants', fontsize=10)
ax4.set_xlabel('Age (yr)'); ax4.set_ylabel('$\\Phi_{in}(t)$')
ax4.legend(fontsize=8, framealpha=0.93)

plt.savefig('lifecycle_dip_human.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
print("Fig. 2 saved: lifecycle_dip_human.png")
