#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0 -- see
LICENSE in the repository root, or https://creativecommons.org/licenses/by-nc-nd/4.0/
SPDX-License-Identifier: CC-BY-NC-ND-4.0

lifetime_near_Mmin.py — numerical check of how lifetime t_B depends on
repair capacity (rho) near the M_min boundary in the canonical VBLS system (S_can).

Context: monograph section 15.2.2.

What is tested:
  (A) The hypothesis t_B ~ C / D (C = R0 - R*, D = dR/dt|_tA) in the regime of
      the original illustrative table (R0 = 0.81, nu0 = 0.0247, rho = 0.12..0.50).
  (B) The true approach to M_min: the R-subsystem attractor R_bal(Sigma0) -> R*
      from above (starting right at the threshold, R0 = R* + eps; varying rho).

Method: integrating (S_can) with 4th-order Runge-Kutta until the first exit
from the viability region V = {E > E*, R > R*}.

Canonical system (formula 8.2):
  dE/dt   = eta * Phi_in - (mu E^2 + lam E)
  dSigma  = max(0, sigma0*(mu E^2 + lam E) - R * Phi_cap)
  deta/dt = -alpha * Sigma * eta
  dR/dt   = -nu0 R - nu Sigma R + rho * g(E) (1-Sigma)(1-R),   g(E)=(E-E*)/(E0-E*) in [0,1]

CONCLUSION (see the printed output below):
  - In part (A), t_B monotonically INCREASES with rho (84.6 at n_rep=12 -> 311.4
    at n_rep=50), matching the biologically correct direction: more repair
    capacity -> longer life. However, the range of t_B is only ~3.7x, while
    1/D varies by a factor of ~27; the product t_B*D is not constant.
    The hypothesis t_B ~ C/D is NOT confirmed. The claimed "tripling of lifetime
    upon loss of 3 repair genes" is refuted in both direction and magnitude:
    integration gives the opposite (loss of repair SHORTENS life).
  - In part (B), as R_bal -> R* (the true approach to M_min), t_B is FINITE
    and decreases to a small value (~7.5 at R_bal - R* ~ 0.007). There is no
    critical slowing down (t_B -> infinity) at the boundary.
  - Death in all runs is via the R-channel; t_B is set primarily by the
    Sigma dynamics (sigma0, Phi_cap, alpha) and by the position of the R_bal
    attractor above the threshold R*, not by the initial balance D = dR/dt|_tA.
"""

# ---------- common parameters (S_can) ----------
E0, Estar = 1.0, 0.04
Rstar     = 0.15
Sigma0, eta0 = 0.0, 1.0
mu, lam   = 1.5, 0.30           # F_diss(E0) = 1.8
nu        = 0.06
sigma0    = 0.0162
alpha     = 0.030
Phi_in, Phi_cap = 1.9, 0.02     # gamma0 = eta0*Phi_in/F_diss(E0) = 1.056 (starts near M_min)

def Fdiss(E):
    return mu*E*E + lam*E

def gfun(E):
    g = (E - Estar) / (E0 - Estar)
    return 0.0 if g < 0.0 else (1.0 if g > 1.0 else g)

def Rbal(rho, nu0, S=0.0):
    """Stable attractor of the R-subsystem with Sigma frozen and g=1."""
    return rho*(1-S) / (rho*(1-S) + nu0 + nu*S)

def D_balance(rho, nu0, R0):
    """Balance function D = dR/dt|_tA (at Sigma0=0, g(E0)=1)."""
    return rho*(1-Sigma0)*(1-R0) - (nu0 + nu*Sigma0)*R0

def integrate(rho, nu0, R0, dt=0.005, tmax=2.0e6):
    E, S, eta, R = E0, Sigma0, eta0, R0
    t = 0.0
    def f(E, S, eta, R):
        Fd = Fdiss(E)
        return (eta*Phi_in - Fd,
                max(0.0, sigma0*Fd - R*Phi_cap),
                -alpha*S*eta,
                -nu0*R - nu*S*R + rho*gfun(E)*(1-S)*(1-R))
    steps = int(tmax/dt)
    for _ in range(steps):
        if R <= Rstar: return t, "R"
        if E <= Estar: return t, "E"
        k1 = f(E, S, eta, R)
        k2 = f(E+0.5*dt*k1[0], S+0.5*dt*k1[1], eta+0.5*dt*k1[2], R+0.5*dt*k1[3])
        k3 = f(E+0.5*dt*k2[0], S+0.5*dt*k2[1], eta+0.5*dt*k2[2], R+0.5*dt*k2[3])
        k4 = f(E+dt*k3[0], S+dt*k3[1], eta+dt*k3[2], R+dt*k3[3])
        E   = max(E + dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6, 0.0)
        S   = min(max(S + dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6, 0.0), 1.0)
        eta = min(max(eta + dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6, 0.0), 1.0)
        R   = min(max(R + dt*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6, 0.0), 1.0)
        t  += dt
    return None, "none"

if __name__ == "__main__":
    print("="*72)
    print("(A) Regime of the original table, §15.2.2: R0=0.81, nu0=0.0247, rho=0.12..0.50")
    print("    Testing the hypothesis t_B ~ C/D.")
    print("="*72)
    nu0_A, R0_A = 0.0247, 0.81
    partA = []  # (1/D, t_B) — for the plot
    print(f"{'n_rep':>5} {'rho':>5} {'D':>9} {'1/D':>8} {'t_B':>9} {'t_B*D':>8} {'chan':>5}")
    for nrep in (50, 25, 18, 15, 14, 13, 12):
        rho = 0.01*nrep
        D   = D_balance(rho, nu0_A, R0_A)
        tB, ch = integrate(rho, nu0_A, R0_A)
        partA.append((1/D, tB))
        print(f"{nrep:>5} {rho:>5.2f} {D:>9.4f} {1/D:>8.1f} {tB:>9.1f} {tB*D:>8.3f} {ch:>5}")
    print("  => t_B grows with rho (84.6 -> 311.4, range x3.7), but NOT like 1/D")
    print("     (1/D range x27); t_B*D is not constant; the hypothesis t_B ~ C/D is refuted.")

    print()
    print("="*72)
    print("(B) True approach to M_min: R_bal(Sigma0) -> R* from above")
    print("    R0 = R* + 0.01 = 0.16; nu0=0.0247; varying rho.")
    print("="*72)
    nu0_B, R0_B = 0.0247, 0.16
    partB = []  # (R_bal - R*, t_B) — for the plot
    print(f"{'rho':>7} {'R_bal':>7} {'R_bal-R*':>9} {'t_B':>9} {'chan':>5}")
    for rho in (0.050, 0.020, 0.010, 0.007, 0.006, 0.005, 0.0046):
        Rb = Rbal(rho, nu0_B)
        tB, ch = integrate(rho, nu0_B, R0_B)
        partB.append((Rb - Rstar, tB))
        print(f"{rho:>7.4f} {Rb:>7.3f} {Rb-Rstar:>9.4f} {tB:>9.1f} {ch:>5}")
    print("  => as R_bal -> R*, lifetime is FINITE and decreases (rapid death near the boundary);")
    print("     t_B grows monotonically with rho (more repair -> longer life).")

    # ------------------ Figure: two-panel illustration (English labels) ------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))

        # Panel A: integrated t_B vs the proposed 1/D scaling.
        # If t_B ~ C/D held, points would lie on a straight line through the origin
        # (gray reference). Log Y axis lets both curves stay visible at their natural
        # ranges: integrated t_B spans ~85..311 (x3.7), reference spans ~85..8400 (x27).
        inv_D, tB_A = zip(*partA)
        span_data = max(tB_A) / min(tB_A)
        axA.plot(inv_D, tB_A, "o-", color="#1f77b4", lw=1.6, ms=7,
                 label=rf"Integrated $t_B$  (range $\times${span_data:.1f})")
        k_ref = tB_A[0] / inv_D[0]
        ref_vals = [k_ref*x for x in inv_D]
        span_ref = max(ref_vals) / min(ref_vals)
        axA.plot(inv_D, ref_vals, "--", color="0.55", lw=1.2,
                 label=rf"Reference $t_B \propto 1/\mathcal{{D}}$  (range $\times${span_ref:.1f})")
        axA.set_yscale("log")
        axA.set_xlabel(r"$1/\mathcal{D}$  (proposed scaling)")
        axA.set_ylabel(r"Lifetime  $t_B$")
        axA.set_title(r"(A)  $t_B$ vs $1/\mathcal{D}$ — proposed $t_B\sim C/\mathcal{D}$ refuted")
        axA.legend(loc="lower right", frameon=False, fontsize=9)
        axA.grid(alpha=0.3, which="both")

        # Panel B: integrated t_B vs distance to M_min (R_bal - R*).
        dist, tB_B = zip(*partB)
        axB.plot(dist, tB_B, "s-", color="#d62728", lw=1.6, ms=7,
                 label=r"Integrated $t_B$ (numerical)")
        # Mark the finite limit value as R_bal -> R*
        axB.annotate(f"$t_B \\approx {tB_B[-1]:.1f}$\nat $R_{{bal}}\\to R^*$",
                     xy=(dist[-1], tB_B[-1]),
                     xytext=(dist[-1]+0.05, tB_B[-1]+8),
                     arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
                     fontsize=9, color="0.25")
        axB.set_xlabel(r"Distance to $\mathcal{M}_{\min}$:  $R_{\mathrm{bal}} - R^*$")
        axB.set_ylabel(r"Lifetime  $t_B$  (integrated)")
        axB.set_title(r"(B)  Lifetime near $\mathcal{M}_{\min}$ — finite, no critical slowing")
        axB.set_xlim(left=0)
        axB.legend(loc="upper left", frameon=False, fontsize=9)
        axB.grid(alpha=0.3)

        fig.suptitle("VBLS: lifetime $t_B$ vs repair capacity near $\\mathcal{M}_{\\min}$  "
                     "(canonical system, R-channel)", y=1.02, fontsize=11)
        fig.tight_layout()
        out_path = "lifetime_near_Mmin.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved: {out_path}")
    except ImportError:
        print("\n(matplotlib not installed; figure not generated)")
