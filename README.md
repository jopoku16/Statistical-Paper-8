# Conformalized Prediction of Acute Kidney Injury in the ICU

Code for the manuscript *"Conformalized Prediction of Acute Kidney
Injury in the ICU: Distribution-Free Coverage Guarantees with
Class-Conditional Validity and External Validation."*

## Data

The study uses the PhysioNet/Computing in Cardiology Challenge 2019
dataset (ICU patients from two hospital systems). The raw data are
governed by the PhysioNet Credentialed Health Data Use Agreement and
cannot be redistributed here. Download them directly from PhysioNet:

- https://physionet.org/content/challenge-2019/

After downloading, place the combined table at
`data/physionet/Dataset.csv` before running the scripts.

## Scripts

Run from the repository root. Each script writes tables to `results/`
and figures to `figures/`.

- `code/02_real_data_analysis.py` — load and split the two hospitals,
  fit the base models, run split and class-conditional conformal
  prediction.
- `code/03_enhanced_analysis.py` — add XGBoost and LightGBM, SHAP
  explanations, external validation, and risk stratification.
- `code/04_deep_learning_bootstrap.py` — add the MLP baseline and
  bootstrap confidence intervals for AUC and coverage.

## Results

The `results/` folder holds the summary tables reported in the paper
(`summary.json`, `model_comparison.csv`, `bootstrap_results.json`,
`mondrian_results.csv`, `risk_stratification.csv`,
`feature_importance.csv`, `cohort_stats.csv`).
