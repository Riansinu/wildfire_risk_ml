# Dataset provenance

Original source: https://archive.ics.uci.edu/static/public/162/forest+fires.zip

Downloaded from UCI for this project; `forestfires.csv` is preserved without modification. The archive's original `forestfires.names` is also included. Record the checksum from `artifacts/metadata.json` when comparing runs.

Cortez, P. & Morais, A. (2007). Forest Fires [Dataset]. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5D88D. License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/.

517 records, 12 predictors, area in hectares. The source is a regression dataset. This project derives the classification target `area > 0`; zero area is not a confirmed non-fire day. No synthetic data or manufactured negative records are added.
