# Release audit · 22 August 2026

## Status

`STAGING PASSED — PUBLICATION BLOCKED ON METADATA AND PERMISSION`

## Checks completed

- Main Figures 1–3, Extended Data Figure 1 and the robustness inventory were
  regenerated from the staged package.
- All staged Python scripts compile.
- JSON and YAML metadata are parseable.
- No absolute Windows paths remain.
- No bearer-token values or other supplied secret values were found.
- No manuscript, cover letter, editorial simulation, claim matrix or review
  material is present.
- No NetCDF, GRIB, Parquet, ZIP or provider-controlled raw archive is present.
- Generated Python bytecode is excluded by `.gitignore` and from checksums.
- `CHECKSUMS.sha256` covers every authorized staged file except itself.

## Deliberate exclusion

`scripts/57_synthesise_observational_replications.py` is excluded because it
depends on a local visualization-skill installation and does not generate an
Article result. Its underlying scientific outputs remain documented by the
portable national and global scripts included here.

## Blocking items before public release

All four items were closed for the v1.0.0 release of 7 September 2026.

1. `CITATION.cff` carries the final author list, affiliations, version 1.0.0,
   the archive DOI and the repository URL. RESOLVED.
2. `LICENSE` releases `src/` and `scripts/` under MIT and `results/`, `config/`
   and the documentation under CC BY 4.0, with the third-party data carve-out
   retained. RESOLVED.
3. Written AVAMET permission was obtained and is recorded in the manuscript
   Acknowledgements; the documented non-redistribution route is preserved, so
   no raw or processed AVAMET observations are included here. RESOLVED.
4. The clean repository is at <https://github.com/MarcSemperLloret/Forecast-rankings>
   and the versioned archive at https://doi.org/10.5281/zenodo.22638176; the DOI
   is inserted in the manuscript Code availability statement. RESOLVED.
