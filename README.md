# Ember — AI-Based Wildfire Risk Assessment

An application project for Unit 5 **Bayesian Model Averaging**, with an executable Jupyter notebook and Streamlit interface. Includes actual measured comparisons with Logistic Regression, Decision Tree, Random Forest and a prevalence baseline. The report and presentation are intentionally excluded.

## Start here

Open `notebooks/01_wildfire_risk.ipynb`. An executed copy with results is included. The notebook explains the target, spatial split, preprocessing, Bayesian evidence, alternatives and calibration checks, and regenerates the same artifacts used by the app.

### Install (PowerShell, from this project folder)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt -c requirements-lock.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Use Python 3.14 for the supplied exact version constraints and model artifact. `requirements-lock.txt` records the tested direct-package versions; it is not a full transitive environment lock. On a different Python/package environment, install `-e ".[dev]"` without constraints and **retrain** before loading models. Pickle/joblib artifacts are version-sensitive: load only trusted artifacts.

On macOS/Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.

### Regenerate and verify

```powershell
.venv\Scripts\python.exe -m wildfire.train
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m jupyterlab notebooks/01_wildfire_risk.ipynb
```

Running all notebook cells also retrains models. Bundled data makes training work offline after dependencies are installed. Hyperparameters and split seeds are specified in source. Timings vary by hardware.

## Target and scope

**The response is `1[recorded burned area > 0]`, conditional on a fire record.** UCI Forest Fires is a burned-area dataset, not a representative sample of fire and non-fire days. Area zero is not a confirmed absence of fire. The displayed probability is a positive recorded-area proxy; it is not daily ignition probability. Operational occurrence estimation requires representative non-fire controls and observation windows.

## Application features

* Interactive weather, month and FWI-component input with probabilities from each algorithm.
* BMA probability, validation-selected class and candidate-model disagreement.
* CSV template, validated batch scoring and downloadable probabilities.
* Measured comparison table, ROC/reliability plots and calibration-bin counts.
* Sixteen candidate model weights and feature-block inclusion probabilities.
* Input validation, training-range checks and explicit target labels.

## Experiment design

Group by `(X,Y)` cell: hold out 20% of cells for test (seed 42); hold out 25% of remaining cells for validation (seed 43). Row fractions differ because cell sizes vary. Month/day labels are not complete timestamps, so no artificial chronological ordering is used. Predictors omit coordinates, day and area.

Training-only four-fold GroupKFold selects hyperparameters using negative Brier score. Each Pipeline fits its scaler inside the fold. Rain uses log1p and month uses sine/cosine. Validation selects a per-model F1 threshold, frozen before test evaluation. Validation F1 is optimistic because of this selection; F1 at 0.5 is also included. Models stay fit to training only, and all algorithms use the identical validation and test rows. No calibration mapping is fitted on test.

BMA enumerates all subsets of weather, moisture, fire-index and season blocks. Each candidate is Bayesian logistic regression with independent normalized Gaussian coefficient priors, including intercept; equal model priors. Training CV chooses prior standard deviation 1 or 2.5. Marginal likelihood is approximated by Laplace integration, and coefficient uncertainty in prediction is integrated with Gauss-Hermite quadrature. This is approximate Bayesian model averaging. Full details and formulas are in the notebook and source.

ROC-AUC, F1, Brier and log loss are measured. Calibration is checked using reliability plots with five fixed bins and counts. Brier is not a pure calibration measure. No claimed accuracy or superiority is assumed. Training timings include search; selected-refit timing is separate. Inference timing is warmed batch milliseconds per row, not live request latency.

## Repository contents

```text
app.py                         Streamlit application
notebooks/01_wildfire_risk.ipynb Executed end-to-end notebook
src/wildfire/data.py            Schema, features and grouped splits
src/wildfire/bma.py             Bayesian logistic subset averaging
src/wildfire/train.py           Tuning, evaluation, persistence and plots
data/forestfires.csv            Original UCI source data
data/README.md                 Attribution and provenance
artifacts/                     Actual measured outputs and trained models
tests/                         Model/data checks and application smoke test
.github/workflows/tests.yml     Retrain and test on GitHub Actions
```

`artifacts/model_comparison.csv` is the requested comparison table. It includes validation/test metrics, training time, inference time and interpretability. `split_assignments.csv` and `test_predictions.csv` permit auditing the shared test set. `metadata.json` records versions, data checksum, target, split summary and hyperparameters. CV results are persisted per algorithm.

## GitHub and deployment

Repository: [Riansinu/wildfire_risk_ml](https://github.com/Riansinu/wildfire_risk_ml). Clone the repository and run the installation steps above from its root. CI retrains models before running tests. For Streamlit Community Cloud, use `app.py` as the entry point and Python 3.14 with matching dependencies; verify package versions or retrain artifacts in the deployment environment. The included `.streamlit/config.toml` sets the theme and headless behavior.

## Data and limitations

Cortez, P. & Morais, A. (2007). *Forest Fires* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5D88D. Dataset: CC BY 4.0. Dataset credit is separate from this project's MIT-licensed code.

517 records from one park provide limited external validity. Grouping cells cannot remove all cross-cell weather/event dependence. Small test and calibration-bin counts limit conclusions. Weights depend on the candidate family and Laplace approximation. Classification thresholds are educational decisions, not firefighting alert levels. The project contains no report or PPT.
