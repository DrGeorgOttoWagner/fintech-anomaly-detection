# Credit Card Fraud Detection — Reproducible FTAD Study

A reproducible research example demonstrating FTAD on a highly imbalanced, real-world fraud-detection dataset.

> **The dataset is not distributed with this repository.** Download `creditcard.csv` directly from
> [Kaggle — Credit Card Fraud Detection (Machine Learning Group – ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
> and load it into FTAD locally. According to its Kaggle page, the dataset is published under the Open
> Database License (ODbL).

## At a glance

| | |
|---|---|
| Transactions | 284,807 (31 columns: `Time`, `V1`–`V28`, `Amount`, `Class`) |
| Frauds | 492 (prevalence **0.1727 %**, ≈ 578 genuine per fraud) |
| Period | ≈ 48 hours of elapsed time |
| Algorithms | 20 (17 supervised, 3 unsupervised anomaly detectors) |
| Protocols | 3 stratified train/validation/test profiles (70/15/15, 80/10/10, 60/20/20), duplicate-safe group splits, a chronological split |
| Selection | Validation average precision (AP); thresholds frozen on validation; each test set scored once |

## Key findings

1. **Accuracy is useless here.** Always predicting "genuine" scores 99.8273 % accuracy and catches no fraud.
2. **`Amount` alone is weak evidence.** Its relation to fraud is non-monotonic: the lowest and highest
   amount deciles have higher fraud rates than the middle range.
3. **The anonymised components carry the signal.** V14, V10, V12, V4 and V11 separate the classes most strongly.
4. **A leading group, not a single winner.** CatBoost, XGBoost, Extra Trees and Random Forest form the
   strongest supervised group in every profile. The model chosen by validation AP changes with the
   allocation profile.
5. **Supervised beats unsupervised on this labelled dataset.** In profile A, the best tree ensembles
   reach a test AP of about 0.83–0.84. The best unsupervised detector, Isolation Forest, reaches 0.15.
6. **Robustness checks lower the numbers.**
   - Removing test records that have an exact copy in training lowers AP slightly.
   - Duplicate-safe group splits and a chronological split lower it more.

| Profile | Validation-selected supervised model | Validation AP | Test AP | Test frauds |
|---|---|---:|---:|---:|
| A — 70/15/15 | CatBoost | 0.8349 | 0.8415 | 74 |
| B — 80/10/10 | Extra Trees | 0.8658 | 0.8901 | 49 |
| C — 60/20/20 | XGBoost | 0.8267 | 0.8773 | 99 |

![Precision vs recall at validation-frozen thresholds (profile A)](figures/06_precision_vs_recall_profile_A.png)

## Contents

| Document | What it covers |
|---|---|
| [workflow.md](workflow.md) | Step-by-step reproduction in FTAD, from download to export |
| [methodology.md](methodology.md) | Protocols, splits, selection rules, robustness design, verification |
| [results.md](results.md) | Full results: EDA, 20-algorithm benchmark, three profiles, robustness, cost, limitations |
| [tables/](tables/) | Aggregate CSV tables behind the results |
| [figures/](figures/) | Aggregate figures |

Only aggregate results are published here: no transaction rows, score vectors, record-level
predictions or identifiers.

## Environment of the reference run

FTAD 1.0.0 on macOS 27 / Apple M3 Pro (18 GB RAM). Library versions: Python 3.12.4, scikit-learn 1.9.1,
XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10, pandas 3.1.0, NumPy 2.5.4, SciPy 1.18.1, imbalanced-learn 0.14.2.
