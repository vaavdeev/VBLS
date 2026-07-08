# AnAge — Cross-species allometry on Mammalia

**Monograph section:** §17.4 (summary), Appendix D.6 (§D.6.1–D.6.6, full verification)
**VBLS predictions tested:** four scaling laws with zero free parameters on 318 species (base sample) and 111 species (quality-filtered sample) from AnAge Build 14 — T_max ~ M^0.218, Kleiber's law, T_max ~ (P/M)^(−1.0), Rubner invariant; allometric anomalies — identification of species deviating from the VBLS scaling law and diagnosis of the dominant system (S) parameter responsible for the deviation; Monte Carlo test of attenuation bias in the P/M measurement (§D.6.4, reviewer point).

**Data source:** Tacutu et al. (2018) AnAge Build 14, Nucleic Acids Research 46:D1083–D1090.

---

## Scripts

| File | What it does |
|---|---|
| `anage.py` | Main verification. Four scaling tests on two samples (base N=318 and quality-filtered N=111): T_max ~ M^α, P/M ~ M^β, T_max ~ (P/M)^γ, Rubner invariant. Computes empirical exponents and compares to VBLS predictions (0 free parameters). |
| `anomalies.py` | Allometric anomalies analysis (§D.6.6): for each species, computes deviation of observed T_max from the VBLS allometric prediction; identifies the dominant VBLS parameter explaining the anomaly. |
| `attenuation_bias.py` | Monte Carlo simulation (N=2000) addressing the reviewer concern that the observed γ = −0.656 vs. predicted −1.000 could be regression dilution from ~40% measurement noise in P/M (§D.6.4). |

## Data

| File | Description |
|---|---|
| `anage_data.csv` | AnAge Build 14 (semicolon-delimited): species, class, order, body mass (g), metabolic rate (W), T_max (yr), data quality flag, sample size. |
| `anomalies_table.csv` | Output of `anomalies.py`: full species table with predicted T_max, observed T_max, anomaly magnitude, dominant parameter. |
| `anomalies_focal.csv` | Subset: species with anomaly magnitude above threshold (focal cases discussed in the monograph). |

## Figures

| File | Figure in monograph |
|---|---|
| `anage_verification.png/.pdf` | Рис. D.6.1 — base sample (N=318), four scaling tests |
| `anage_improved.png/.pdf` | Рис. D.6.2 — quality-filtered sample (N=111) + breakdown by taxonomic order |
| `anomalies_fig.png/.pdf` | Рис. D.6.3 — allometric anomalies: observed vs. predicted T_max with anomaly markers |
| `attenuation_bias.png/.pdf` | Monte Carlo attenuation-bias simulation (§D.6.4, not separately numbered as a figure) |

## Run

```bash
python anage.py             # main scaling verification
python anomalies.py         # allometric anomalies
python attenuation_bias.py  # attenuation bias simulation
```
All three scripts read `anage_data.csv` from the same directory.

---

## Supplementary material: PGLS analysis of σ₀ vs. body mass ("P4")

The files `P4_RESULTS.md`, `_check_r.R`, and the `p4_*` scripts/outputs are a phylogenetically-controlled (PGLS) exploratory analysis of how σ₀ scales with body mass, using the `datelife`/`ape`/`nlme` R ecosystem. **This analysis is not part of the current monograph** — it is not cited or discussed anywhere in the published text, and `P4_RESULTS.md` itself references a working file ("РЕПАРАЦИЯ.md") that no longer exists in this project. It is kept in this repository as a basis for possible future work, not as a verified or citable result. Do not treat any number in `P4_RESULTS.md` as an endorsed VBLS prediction.

## Excluded from this repository

- `drugage.csv`, `observations.xlsx`, `release.html` — unrelated third-party datasets (DrugAge database, a different taxon table) not read by any script in this folder and not cited in the monograph.
