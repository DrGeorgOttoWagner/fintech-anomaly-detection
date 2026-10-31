# User guide

FTAD follows one research workflow: **Upload → Schema → Explore → Experiment → Evaluate → Report.**
The workflow indicator in the header shows which steps are complete for the active dataset.

## 1. Starting FTAD

```bash
./scripts/start.sh
```

The app runs at **http://127.0.0.1:8790**. Installation and configuration are covered in
[getting-started.md](getting-started.md).

## 2. Uploading a dataset

On *Overview* or *Datasets*, click **Browse Files** to open the system file picker, or drag a file onto
the drop zone.

- **Formats:** CSV (comma, semicolon, tab or pipe separated) and Parquet.
- **Validation:** upload progress is shown. Problems such as an empty file, binary content, duplicate
  column names, invalid Parquet or a file over the size limit are explained.
- **Fingerprint:** each file gets a SHA-256 fingerprint. Uploading an identical file reopens the existing dataset.
- **Storage:** the raw upload is kept read-only, and analyses run on a parsed Parquet copy.

## 3. Schema mapping

*Schema Mapping* lists every column with its type, cardinality, missing values and detection hints.
Assign exactly one role per column:

| Role | Use for |
|---|---|
| **Target** | The binary label to analyse and predict. Choose the **positive label** explicitly, e.g. `1` = fraud. |
| **Numeric feature** | Numeric model inputs, including value/amount columns |
| **Categorical feature** | Text or code columns, one-hot (or ordinal) encoded |
| **Existing PCA feature** | Already-transformed, anonymised components. Modelled as numeric, never interpreted. |
| **Time / timestamp** | Elapsed seconds, a calendar timestamp or an ordinal order. Used for time analysis and chronological splits. |
| **Transaction ID** | Record identifier, shown in the Anomaly Explorer and never used as a feature |
| **Group / entity ID** | Customer, card or account key for group-aware splitting |
| **Ignore** | Excluded from all analyses |

In the summary card you also pick:

- the **amount/value column**, which gets the dedicated value analysis;
- the **domain terminology**: *fraud detection* or *generic anomaly / classification*.

Suggestions are only suggestions. The schema is validated live and must be confirmed before any
statistics are computed. A changed schema gets a new schema hash, and results are keyed by it.

## 4. Data quality

*Exploratory Analysis → Descriptive statistics* shows:

- the number of records and features, the prevalence and the imbalance ratio;
- **missing values** per column;
- **exact duplicate rows**: count and groups;
- warnings, for example heavy skew or extreme imbalance;
- a sortable and filterable statistics table (count, mean, SD, quantiles, skewness, kurtosis).

*Class imbalance* shows the class counts and shares, the minority:majority ratio and the accuracy of
always predicting the majority class. It also explains why that accuracy is misleading.

## 5. Exploratory data analysis

| Tab | Content |
|---|---|
| **Amount / value** | Mean, median, SD, quartiles, skewness; histograms (linear or log); box plots; statistics and non-parametric tests by class; positive rate by decile |
| **Time** | Events and positives over time, positive rate per interval, hour-of-cycle profile, distribution by class. Elapsed seconds are never presented as clock time. |
| **Features** | Pearson/Spearman heatmap; relevance indicators (KS, univariate AUC, point-biserial r, Cohen's d, mutual information); class-conditional distributions; PCA component properties; exploratory 2-D projection, labelled as a visual sample |
| **Findings** | Factual statements generated from the computed values |

All statistics use every record. Only the scatter projection uses a sample, and it is labelled as one.

## 6. Experiment setup

Open the *Experiment Studio*.

- **Mode:**
  - **Quick Benchmark**: Dummy, Logistic Regression, Random Forest, XGBoost and Isolation Forest.
  - **Full Benchmark**: any selection of the 20 algorithms.
  - **Hyperparameter optimisation**: randomised search on training folds.
- **Split:**
  - Linked train/validation/test sliders that always sum to 100 %, with presets 70/15/15, 80/10/10 and 60/20/20.
  - Methods: **stratified**, random, **chronological** (needs a time column) or **group-aware** (needs an entity ID).
  - A **seed** fixes the split.
  - A live preview shows partition sizes and positive counts and warns about configurations that are too small.
- **Imbalance strategies:** none, class weighting, random undersampling, random oversampling, SMOTE.
  - Each selected strategy runs with each supervised model.
  - Inapplicable combinations are recorded as *skipped*, with the reason.
- **Preprocessing:**
  - imputation, scaling and optional log of the amount column;
  - categorical encoding;
  - optional PCA or feature selection.

  All of it is fitted on **training rows only**. Resampling also happens only inside training.
- **Threshold policy:** maximise F1 on validation (default), target recall, target precision, the model default or a fixed value.
- **Options:**
  - permutation importance;
  - Platt calibration;
  - an extended resource profile that lifts the training-row caps;
  - the anomaly-detector training mode;
  - advanced JSON hyperparameter overrides.

Runs execute in a background worker. Progress, the current algorithm, the elapsed time and any errors
are shown live, and an experiment can be cancelled.

## 7. Model benchmarking

*ML Benchmark* shows a leaderboard for each evaluation protocol (all rows share an identical split).

- Rows are ranked by a **validation** objective that you choose. AP (shown as PR-AUC) is the default.
- **Test** metrics are a final, single-use estimate. Selecting a model by its test score would bias that score.
- PR and ROC curves, plus metric bars, compare the runs.

## 8. Threshold analysis

Ranking metrics (AP, ROC-AUC) judge the whole score ordering. Threshold metrics (precision, recall,
F1, confusion matrix) judge the decisions at one operating point.

*Model Evaluation* shows:

- the metrics for training (in-sample), validation and test;
- the confusion matrix;
- PR, ROC and calibration curves;
- native and permutation importance.

**Threshold Studio** lets you move the threshold and see the trade-off between missed cases and
false alarms. By default it works on validation data; the test view is labelled as exploratory.
The **Business impact** tab applies hypothetical, user-supplied costs to the confusion matrix.

The **Anomaly Explorer** lists record-level predictions on your own local data:

- filters for TP, FP, TN and FN;
- the feature values of the selected record;
- CSV export.

## 9. Metrics

| Metric | Meaning |
|---|---|
| **Average precision (AP)** | Precision averaged over all recall levels (step-wise area under the precision–recall curve). Labelled *PR-AUC* in the UI and compared with its random baseline, the prevalence. |
| **Precision** | Share of raised alerts that are true positives |
| **Recall** | Share of all positives that are detected |
| **F1** | Harmonic mean of precision and recall |
| **ROC-AUC** | Probability that a random positive is scored above a random negative. Can look high under extreme imbalance even when alerts are poor. |
| **Confusion matrix** | TP, FP, TN, FN at the frozen threshold |

Definitions and interpretation: [metrics.md](metrics.md).

## 10. Reports

*Research Reports* builds a bundle containing:

- `report.md` and `report.html`;
- CSV tables and PNG/SVG figures;
- a JSON manifest (dataset fingerprint, schema, configuration, versions).

A report can include an experiment. For fraud-domain schemas it adds an *analysis-coverage* appendix.

You can download the bundle as a ZIP. If `FTAD_EXPORT_DIR` is set, you can also export it to that folder.

## 11. Switching datasets

Uploading another file starts a new **active dataset session**.

- Results, schemas and experiments of other datasets are never shown for it, and the workflow
  indicator starts again at *Schema*.
- Earlier datasets stay in the registry on the *Datasets* page and can be reopened explicitly.
- A new schema for the same dataset creates a new schema version.

## 12. Score new data

*Score New Data* applies a trained pipeline to another uploaded dataset. A compatibility check runs
first: missing or unexpected columns, types, unseen categories, warnings about PCA provenance, and drift.

## Interpreting results responsibly

- Compare AP with the prevalence (the AP of a random ranking).
- Probabilities from class-weighted or resampled models are not calibrated.
- Anomaly scores are not probabilities, and an anomaly is not necessarily fraud.
- EDA associations are not causal effects. A large amount is not by itself evidence of fraud.
- With few positives in the test partition, test metrics carry wide uncertainty.
