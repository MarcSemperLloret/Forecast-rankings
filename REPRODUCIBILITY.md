# Reproducibility guide

## Analysis roles

The package preserves four roles used in the study:

1. confirmatory or frozen national interventions;
2. prospective global tests;
3. descriptive scorecards and physical associations;
4. post-gate diagnostics retained after a prospective criterion failed.

Do not reinterpret a descriptive or post-gate result as confirmatory.

## Environment

The final manuscript figures were regenerated under Python 3.13.2 with NumPy
2.5.2, pandas 3.0.5, SciPy 1.18.1, xarray 2026.7.0, DuckDB 1.5.5, Matplotlib
3.11.1, pyproj 3.7.2, Shapely 2.1.2, PyYAML 6.0.3 and requests 2.34.2. The
declared compatible environment is in `pyproject.toml`.

## Fast audit from frozen results

- Figure 1: `scripts/69_plot_main_figure1_reference_choice.py`.
- Figure 2: `scripts/72_plot_main_figure2_controlled_smoothing.py`.
- Figure 3: `scripts/70_plot_main_figure3_support_and_rankings.py`.
- Extended Data Figure 1: `scripts/68_plot_model_family_scale_boundary.py`.
- Extended Data Table 1: `scripts/71_audit_regional_robustness_inventory.py`.
- Supplementary Figure S2 (domain-free control): run
  `scripts/76_simulate_support_ranking_identifiability.py` then
  `scripts/77_plot_support_ranking_identifiability.py`. This control is fully
  self-contained; it generates its own ground truth and needs no external data.

Each figure directory contains source data and `outputs_manifest.json`. Output
hashes should match after regeneration in the stated environment; PDF metadata
may vary across Matplotlib or operating-system versions even when plotted data
are identical.

## Full analysis route

The numbered scripts document the original acquisition and analysis sequence.
The central national extension is configured in
`config/national_causal_extension_2020.yaml` and executed by scripts 59–62.
The physical and model-family extensions are configured by the corresponding
2020 YAML files and executed by scripts 63–68. Scripts 69–72 assemble the final
display and audit layer without modifying scientific results.

Provider-controlled inputs are not present. After retrieving them, preserve the
relative `data/raw/` and `data/interim/` layout expected by the scripts. Never
place access tokens in YAML or logs; the MIDAS downloader reads `CEDA_TOKEN`
from the process environment.

## Frozen headline checks

- National smoothing: all 12 added model–network–metric contrasts select an
  ERA5 optimum of 12.5 or 25 km and a station optimum of 0 km.
- Multi-support scorecard: 22 of 24 positive-support rankings change at least
  one pair and 12 of 24 change winner.
- Physical analysis: all six pre-specified Spearman intervals lie below zero.
- Family test: the prospective common-grid gate fails and native slopes are
  negative at 25, 50 and 100 km.
- Regional sensitivity inventory: 344/344 estimates retain the sign and
  343/344 intervals exclude zero.
- Domain-free control: the ranking reverses in 73 of 81 (skill-gap,
  fine-content) cells, and the analytic boundary d^2 < g(2-g)*var_fine agrees
  with the simulated reversals at 100%.
