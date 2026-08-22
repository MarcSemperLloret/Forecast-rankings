# Verification scale reshapes weather-model rankings

This is the journal-safe reproducibility companion for the Paper3 analysis. It
contains analysis code, frozen derived tables, publication figures and their
source data. It intentionally excludes manuscript files, cover letters,
editorial audits, raw provider-controlled observations and forecast archives.

## Scientific scope

The package evaluates +24-hour 2-m temperature forecasts against point
observations and gridded analyses. The central tests alter either the spatial
structure of an unchanged forecast or the support of an unchanged observation
target. Global physical associations and the negative cross-family roughness
test are retained as bounded extensions.

## Layout

- `src/`: reusable extraction, interpolation, tessellation and spherical-kernel code.
- `scripts/`: numbered acquisition, validation, analysis and figure scripts.
- `config/`: frozen YAML analysis configurations.
- `results/`: selected non-confidential derived tables, summaries, figures, alt text and manifests.
- `REPRODUCIBILITY.md`: execution routes and expected outputs.
- `DATA_AVAILABILITY.md`: provider and redistribution constraints.
- `CHECKSUMS.sha256`: generated only after the staging audit is final.

## Rebuild the final displays

Create an environment from `pyproject.toml`, then run from the package root:

```powershell
python scripts/69_plot_main_figure1_reference_choice.py
python scripts/72_plot_main_figure2_controlled_smoothing.py
python scripts/70_plot_main_figure3_support_and_rankings.py
python scripts/68_plot_model_family_scale_boundary.py
python scripts/71_audit_regional_robustness_inventory.py
```

Figure 4 is included with its frozen source tables and manifest. A full rebuild
from provider data requires the inputs and layout described in
`DATA_AVAILABILITY.md`.

The domain-free support-ranking control (Supplementary Figure S2) is fully
self-contained and requires no external data. Regenerate it from scratch with:

```powershell
python scripts/76_simulate_support_ranking_identifiability.py
python scripts/77_plot_support_ranking_identifiability.py
```

## Release status

This directory is a local staging package, not yet a public release. Author
metadata, an explicit code licence and the versioned archive DOI must be added
before publication.
