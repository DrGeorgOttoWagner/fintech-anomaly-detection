# FinTech Anomaly Detection (FTAD) v1.0.0

First public release of FTAD, a local-first research platform for fraud and anomaly detection on
imbalanced tabular data.

## Highlights

- **Local-first:** data, models and reports stay on the analyst's machine (`127.0.0.1`, no outbound calls).
- **Reproducible EDA:** descriptive statistics, class imbalance, value and time analysis, feature separation and correlations, computed on the full dataset.
- **20 algorithms:** 17 supervised classifiers and 3 unsupervised anomaly detectors (Isolation Forest, LOF, One-Class SVM).
- **Leakage-safe evaluation:** a train/validation/test workflow with preprocessing and resampling fitted on training data only.
- **Validation-frozen thresholds:** threshold analysis with Threshold Studio and confusion matrices; each test set is scored once.
- **Reproducibility metadata:** dataset fingerprint, schema hash, split indices, seeds, configuration and library versions; Markdown/HTML/CSV/figure report exports.
- **Synthetic demo dataset:** `demo/synthetic/`, so you can try FTAD without downloading anything.
- **Public research showcase:** a reproducible study on the Credit Card Fraud Detection dataset.

## Research showcase

[`examples/credit-card-fraud/`](https://github.com/DrGeorgOttoWagner/fintech-anomaly-detection/tree/v1.0.0/examples/credit-card-fraud) applies FTAD to the public Kaggle
Credit Card Fraud Detection dataset:

- 284,807 transactions with 492 fraud cases (prevalence 0.1727 %), an extreme class imbalance;
- all 20 algorithms on three train/validation/test allocation profiles (70/15/15, 80/10/10, 60/20/20);
- duplicate-sensitivity and chronological robustness analyses.

CatBoost, XGBoost, Extra Trees and Random Forest consistently formed the leading supervised group.
**No universal winner was established:** the model selected by validation average precision changed
with the allocation profile.

## Platform

macOS only. Tested on macOS / Apple Silicon.

## Licence

- **FTAD:** MIT.
- **Dataset:** Kaggle's dataset metadata lists the licence as "Database: Open Database, Contents: Database
  Contents" ([ODbL v1.0](https://opendatacommons.org/licenses/odbl/1-0/) for the database,
  [DbCL v1.0](https://opendatacommons.org/licenses/dbcl/1-0/) for its contents). The dataset is not
  redistributed with FTAD; obtain it directly from
  [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
