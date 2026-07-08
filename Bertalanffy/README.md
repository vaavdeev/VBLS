# Bertalanffy — Fish growth dynamics: VBLS vs. von Bertalanffy growth function

**Monograph section:** Appendix D.4 (§D.4.1–D.4.8)
**VBLS prediction tested:** mass-based growth trajectory M(t) with a memory-degraded assimilation term η_abs(t)=exp(−αt), reducing to the classical von Bertalanffy growth function (VBGF) in the limit α→0; three-phase dynamics (growth → peak mass → decline) absent from VBGF.

**Data source:** ICES DATRAS North Sea International Bottom Trawl Survey (NS-IBTS), 1989–2026, Atlantic cod (*Gadus morhua*), n=21794 individual records. [DATRAS data portal](https://www.ices.dk/data/data-portals/Pages/DATRAS.aspx).

---

## Scripts

| File | What it does |
|---|---|
| `mass_cod_EN.py` | Main, currently-cited script. Fits the VBLS mass equation (§18.2, eq. 18.2) and the classical VBGF to individual mass-at-age records; produces the three-panel comparison figure. Memory parameter fixed at α = 0.042 yr⁻¹ (independent length-based fit, see below — not fitted to the mass data itself). |

## Data

| File | Description |
|---|---|
| `Unaggregated trawl and biological information_2026-04-27 11_14_11.csv` | Raw DATRAS NS-IBTS records (age, length, weight per individual). Read directly by `mass_cod_EN.py`. |
| `Disclaimer_Unaggregated trawl and biological information_2026-04-27 11_14_11.txt` | ICES data-use disclaimer accompanying the CSV download (citation requirements, national submission list). |
| `iphc-2025-tsd-031.xlsx` | IPHC Pacific halibut survey data (§D.4.8). Kept for reference even though the halibut analysis itself is not carried out in this repository (see below) — the monograph describes this dataset's structure directly. |

## Figures

| File | Figure in monograph |
|---|---|
| `mass_cod_EN.png` | Рис. D.4.1 — VBLS vs. VBGF mass growth, Atlantic cod |

## Run

```bash
python mass_cod_EN.py
```

---

## Supplementary scripts (not embedded in the current monograph text)

Kept in the repository as supporting material, not required to reproduce Рис. D.4.1:

| File | What it does |
|---|---|
| `vbls_datras_analysis.py` | Length-based VBGF/VBLS fit on the same DATRAS cod data; source of α = 0.042 yr⁻¹, the value `mass_cod_EN.py` fixes its memory parameter to. **Not runnable as-is in this repository**: it reads from a path (`/home/claude/cod_datras_clean.csv`) outside this repo, an intermediate cleaned file not included here. Its output figure `vbls_datras_real.png` is retained for reference. |
| `vbls_longevity_4species.py` / `.png` | Comparison of long-lived, indeterminate-growth fish (Greenland shark, Atlantic cod, rockfish, kaluga, orange roughy) using literature values (Nielsen et al. 2016 *Science* for the Greenland shark, among others). Real published data, not synthetic; not cited in the current monograph text. |
| `vbls_maturation_3species.py` / `.png` | Growth deceleration at maturity in three species (Atlantic cod, European plaice, walleye), using published survey data (ICES AFWG 2023; Beverton & Holt 1957; Guy et al. 2002). Real published data, not synthetic; not cited in the current monograph text. |

## Excluded from this repository

- `DATRAS_dataproducts_units.pdf` — ICES technical documentation, distributed under separate terms that prohibit reproduction without permission (unlike the "Data" itself, which is CC BY 4.0). Not read by any script here; see the [DATRAS data portal](https://www.ices.dk/data/data-portals/Pages/DATRAS.aspx) instead.
- A synthetic-data proof-of-concept script (`vbls_fish_comparison.py`, `np.random.seed`-generated, not real measurements) and its figures, superseded by `mass_cod_EN.py`.
- `vbls_halibut_iphc.py`/`.png` — the halibut analysis the monograph explicitly describes as not carried out (§D.4.8, missing individual identifiers in the IPHC data); `iphc-2025-tsd-031.xlsx` itself is kept (see Data table above).
- `vbls_two_species_EN.py`/`.png`, `vbls_vs_vbgf_cod.png`, `vbls_repro_analytical.png` — outputs of the excluded synthetic/halibut scripts above.
