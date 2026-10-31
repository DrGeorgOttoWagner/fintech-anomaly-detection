# Methodology

## Data

The public *Credit Card Fraud Detection* dataset (Machine Learning Group – ULB, Kaggle) contains
284,807 card transactions made by European cardholders over about two days in September 2013.

| Column | Content |
|---|---|
| `Time` | Seconds elapsed since the first transaction in the file. Not a calendar timestamp. |
| `V1`–`V28` | Principal components released in anonymised form. Their original meaning is not published. |
| `Amount` | Transaction amount. No currency is documented. |
| `Class` | 1 = fraud, 0 = genuine |

The analysed file had SHA-256 `76274b69…98551a89`. All descriptive statistics use every record.

## Protocol principles

These applied to every model run:

1. **Protocol frozen before training.** Split, seed, algorithms, hyperparameters (FTAD defaults),
   preprocessing and the threshold rule were fixed before any model was fitted.
2. **Three partitions.**
   - **Training data** fit preprocessing and models.
   - **Validation data** rank models (by AP) and set each model's alert threshold, which maximises F1
     on validation and is then frozen.
   - **Test data** are scored once at the frozen threshold.
3. **No resampling** in the reported runs. The models train on the observed class distribution.
4. **Unsupervised detectors** (Isolation Forest, LOF in novelty mode, One-Class SVM) are fitted without
   labels on training rows. Labels are used only to set their threshold on validation and to evaluate them.
5. **Resource caps** (FTAD *safe* profile) limit training to stratified subsamples:
   - Gradient Boosting: 120,000 rows;
   - k-NN: 50,000;
   - SVM (RBF): 20,000;
   - LOF: 30,000;
   - One-Class SVM: 10,000.

   These models are therefore not directly comparable with full-data models.

## Studies

| Study | Split | Purpose |
|---|---|---|
| **Confirmatory benchmark** | stratified 70/15/15, seed 42; Dummy, Logistic Regression, Random Forest, XGBoost, LightGBM | Pre-registered selection by validation AP (XGBoost was selected) |
| **20-algorithm comparison** | same partitions as the confirmatory benchmark | Exploratory extension to every registered algorithm |
| **Three allocation profiles** | stratified, seed 42: **A** 70/15/15, **B** 80/10/10, **C** 60/20/20 | Stability of rankings and selection under different allocations |
| **Duplicate sensitivity** | (a) re-scoring each stratified test set without records that have an exact copy in training; (b) group-aware splits keeping identical rows together, profiles A–C | Effect of exact-duplicate leakage |
| **Temporal robustness** | chronological 70/15/15 vs. stratified 70/15/15, both using 29 features (`Time` used only for ordering) | Performance on later data |

Partitions of the stratified profiles:

| Profile | Train (frauds) | Validation (frauds) | Test (frauds) |
|---|---|---|---|
| A 70/15/15 | 199,364 (344) | 42,721 (74) | 42,722 (74) |
| B 80/10/10 | 227,845 (394) | 28,481 (49) | 28,481 (49) |
| C 60/20/20 | 170,884 (295) | 56,961 (98) | 56,962 (99) |

The three profiles are not independent replications:

- their training sets are nested (C ⊂ A ⊂ B);
- their validation and test sets differ;
- any two test sets share only about half of their records.

## Metrics

The main metric is **average precision (AP)**, the step-wise area under the precision–recall curve.
Its random baseline equals the prevalence (0.0017).

At the frozen threshold, the study also reports precision, recall, F1 and the confusion counts. ROC-AUC is
reported for completeness. See [docs/metrics.md](../../docs/metrics.md) for why accuracy and ROC-AUC
are not used for selection.

## Independent verification

FTAD computed all results. Its stored outputs (scores, split indices, confusion counts) were then
checked outside FTAD with pandas, SciPy and scikit-learn:

| Check | Result |
|---|---|
| EDA values | 52 of 52 matched |
| Model metric rows: confirmatory and robustness experiments | 40 of 40 matched |
| Model metric rows: 20-algorithm comparison | 40 of 40 matched |
| Model metric rows: three-profile and duplicate-safe runs | 240 of 240 matched |

The verification compared AP, ROC-AUC, precision, recall, F1 and confusion matrices.

Recomputation confirms that the numbers follow from the stored outputs. It does not show that
the results generalise; see the limitations in [results.md](results.md#limitations).

## Environment

| | |
|---|---|
| Hardware | Apple M3 Pro, 12 logical CPUs, 18 GB RAM |
| OS | macOS 27 |
| Software | FTAD 1.0.0, Python 3.12.4, scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10, imbalanced-learn 0.14.2, pandas 3.1.0, NumPy 2.5.4, SciPy 1.18.1 |
| Training-time measurements | single measurements, at most 10 threads per model |
