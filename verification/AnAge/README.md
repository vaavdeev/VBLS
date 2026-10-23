# AnAge — Cross-species allometry on Mammalia

**Monograph section:** §17.4 (summary), Appendix D.5 (§D.5.1–D.5.9)
**VBLS predictions tested:** four scaling laws on 318 species (base sample) and 111 species (quality-filtered sample) from AnAge Build 14, with the exponents fixed before the comparison (the Rubner constant is one fitted parameter) — T_max ~ M^0.25 (observed 0.218), Kleiber's law (an input assumption), T_max ~ (P/M)^(−1.0), Rubner invariant; allometric anomalies — departures from the empirical fit, checked for non-contradiction with system (S) (§D.5.6); Monte Carlo test of attenuation bias in the P/M measurement (§D.5.4, reviewer point).

**Data source:** Tacutu et al. (2018) AnAge Build 14, Nucleic Acids Research 46:D1083–D1090.

---

## Scripts

| File | What it does |
|---|---|
| `anage.py` | Main verification. Four scaling tests on two samples (base N=318 and quality-filtered N=111): T_max ~ M^a, P/M ~ M^β, T_max ~ (P/M)^p_scale, Rubner invariant. Computes the empirical exponents and compares them with the VBLS exponents, fixed before the comparison. |
| `anomalies.py` | Allometric anomalies (§D.5.6): for each species, the deviation of observed T_max from the empirical allometric fit; for the ten species examined, the parameter of (S) assigned to the deviation — an interpretation, checked only for compatibility with known features of the species. |
| `attenuation_bias.py` | Monte Carlo simulation (N=2000) addressing the reviewer concern that the observed p_scale = −0.656 vs. predicted −1.000 could be regression dilution from ~40% measurement noise in P/M (§D.5.4). |

## Data

| File | Description |
|---|---|
| `anage_data.csv` | AnAge Build 14 (semicolon-delimited): species, class, order, body mass (g), metabolic rate (W), T_max (yr), data quality flag, sample size. |
| `anomalies_table.csv` | Output of `anomalies.py`: anomalous species with predicted and observed T_max, anomaly magnitude and, for the genera examined, the assigned parameter (interpretation). |
| `anomalies_focal.csv` | Subset: species with anomaly magnitude above threshold (focal cases discussed in the monograph). |

## Figures

| File | Figure in monograph |
|---|---|
| `anage_verification.png/.pdf` | Fig. D.5.1 — base sample (N=318), four scaling tests |
| `anage_improved.png/.pdf` | Fig. D.5.2 — quality-filtered sample (N=111) + breakdown by taxonomic order |
| `anomalies_fig.png/.pdf` | Fig. D.5.3 — allometric anomalies and the parameters of (S) assigned to them |
| `attenuation_bias.png/.pdf` | Monte Carlo attenuation-bias simulation (§D.5.4, not separately numbered as a figure) |

## Run

```bash
python anage.py             # main scaling verification
python anomalies.py         # allometric anomalies
python attenuation_bias.py  # attenuation bias simulation
```
All three scripts read `anage_data.csv` from the same directory.

## Excluded from this repository

- `drugage.csv`, `observations.xlsx`, `release.html` — unrelated third-party datasets (DrugAge database, a different taxon table) not read by any script in this folder and not cited in the monograph.
- A phylogenetically-controlled (PGLS) exploratory test of whether σ₀ scales with body mass, using the `datelife`/`ape`/`nlme` R ecosystem: the prediction did not hold up and this line of work has been discontinued by the author. Not part of this repository.
