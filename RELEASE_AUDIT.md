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

1. Replace placeholder authors and repository URL in `CITATION.cff`.
2. Select an explicit code licence; the staging `LICENSE` currently reserves all rights.
3. Obtain written AVAMET redistribution permission or preserve the documented non-redistribution route.
4. Create the clean GitHub repository and versioned Zenodo archive, then insert the DOI in the manuscript.
