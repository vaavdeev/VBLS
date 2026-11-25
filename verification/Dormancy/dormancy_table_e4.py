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
VBLS — Table E.1 (Appendix E.4): maximal dormancy time in anabiosis

Prediction : t = ln(R_entry / R*) / nu0                (formula 12.11)
Temperature: nu0 = A exp(-Ea / (k_B T))                 (formula 12.9)
             Ea = 1.0 and 1.35 eV (Section 12.2), giving the 4-6x drop of nu0
             per 10 C cooling near room temperature quoted there.
Rows with a base nu0 refer to 20 C; the in-situ correction uses the same law
for every row, so all temperature corrections follow one rule.
"""

import math

K_B = 8.617333e-5      # Boltzmann constant, eV/K
YEAR = 3.15576e7       # seconds per year
EA = (1.0, 1.35)       # activation energies, eV (Section 12.2)


def dormancy_years(nu0, ratio):
    """Formula (12.11): maximal dormancy time in years."""
    return math.log(ratio) / nu0 / YEAR


def cooling_factor(t1_c, t2_c, ea):
    """Factor by which nu0 drops when cooling from t1_c to t2_c (deg C), formula (12.9)."""
    t1, t2 = t1_c + 273.15, t2_c + 273.15
    return math.exp(ea / K_B * (1.0 / t2 - 1.0 / t1))


def main():
    print('Check of Section 12.2: cooling 20 -> 10 C')
    for ea in EA:
        print('  Ea = %.2f eV: nu0 drops %.2f times' % (ea, cooling_factor(20, 10, ea)))
    print()

    # name, nu0 (1/s) at t_nu (C), R_entry/R*, in-situ temperature (C) or None, observed (yr)
    rows = [
        ('B. subtilis spores, 20 C', 3e-10, 20, 10, None, '~250'),
        ('H. vulgare, seed bank, 4 C', 2e-9, 4, 5, None, '~50'),
        ('Artemia salina cysts, 20 C', 1e-9, 20, 5, None, '~4-10'),
        ('Nelumbo nucifera seeds (base nu0)', 5e-12, 20, 3, None, '~1300'),
        ('Silene stenophylla (base nu0)', 8e-12, 20, 3, -7, '~32000 (age of the permafrost)'),
    ]
    print('Table E.1')
    for name, nu0, t_nu, ratio, t_situ, obs in rows:
        base = dormancy_years(nu0, ratio)
        line = '  %-34s t_pred = %7.0f yr' % (name, base)
        if t_situ is not None:
            f = [cooling_factor(t_nu, t_situ, ea) for ea in EA]
            line += ' | at %d C: nu0/%.0f..%.0f, t = %.1e..%.1e yr' % (t_situ, f[0], f[1], base * f[0], base * f[1])
        print(line + ' | t_obs ' + obs)
    print()

    # Text only (not a table row): spores at -20 C, no direct observation at this scale
    base = dormancy_years(3e-10, 10)
    f = [cooling_factor(20, -20, ea) for ea in EA]
    print('B. subtilis spores at -20 C (text): nu0/%.0f..%.0f, t = %.1e..%.1e yr; no observation'
          % (f[0], f[1], base * f[0], base * f[1]))


if __name__ == '__main__':
    main()
