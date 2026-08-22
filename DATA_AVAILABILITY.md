# Data availability and redistribution

## AVAMET

AVAMET observations are publicly viewable through MeteoXarxaOnline. The AVAMET
website states that its contents are subject to attribution,
non-commercial-use and no-derivatives terms. AVAMET granted written permission
to use these observations in the associated study. Consistent with those terms,
raw and processed AVAMET observations are not redistributed in this package;
station identifiers, retrieval metadata and checksums are retained where those do
not disclose restricted data.

Source: <https://www.avamet.org/index.php>

## MIDAS Open

United Kingdom observations are obtained from MIDAS Open through CEDA. Users
must create their own account, accept applicable terms and retrieve the data
through CEDA. `scripts/54_download_midas_open_hourly_2020.py` reads a bearer
credential from the `CEDA_TOKEN` environment variable and does not write the
credential to manifests.

Source: <https://catalogue.ceda.ac.uk/>

## INMET

Brazilian hourly observations are retrieved from the Instituto Nacional de
Meteorologia. Provider files are not redistributed here; the acquisition and
normalization scripts record the requested product and checksums.

Source: <https://portal.inmet.gov.br/dadoshistoricos>

## WEATHER-5K and NOAA ISD

The global station archive follows WEATHER-5K, whose underlying source is NOAA
Integrated Surface Database. Users must follow the dataset card and source
terms. This package includes only selected aggregate cell-level results, not
the original archive.

Sources: <https://arxiv.org/abs/2406.14399> and
<https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database>

## ERA5 and forecast products

ERA5 and most forecast fields were retrieved from the public WeatherBench 2
archive and cited model providers. FuXi- and GenCast-derived fields are not
redistributed here. The package provides configurations and hashes for audit.

Source: <https://weatherbench2.readthedocs.io/>

## Domain-free simulation

The domain-free support-ranking control (`scripts/76_*` and `scripts/77_*`,
`config/synthetic_support_identifiability.yaml`,
`results/synthetic_support_identifiability/`) uses no external data. It generates
its own ground-truth field from fixed seeds, so it is fully redistributable and
reproducible without any provider access.
