# User guide

FTAD follows one research workflow: **Upload → Schema → Explore → Experiment → Evaluate → Report.**
The workflow indicator in the header shows which steps are complete for the active dataset.

## 1. Upload a dataset

*Overview* or *Datasets* → **Browse Files** opens your operating system's file picker (or drag a
file onto the drop zone). CSV (comma/semicolon/tab/pipe) and Parquet are accepted.

- Upload progress is shown; parsing errors are explained (empty file, binary content, duplicate
  column names, invalid Parquet, too large, …).
- Each file is fingerprinted with SHA-256. Uploading an identical file reopens the existing dataset.
- The upload becomes the **active dataset session**. Results of other datasets are never shown for
  it; earlier datasets can be reopened from the registry.

## 2. Map the schema

*Schema Mapping* lists every column with type, cardinality, missing values and detection hints.
Assign one role per column:

| Role | Meaning |
|---|---|
| Target | The label to analyse/predict (choose the positive label explicitly) |
| Numeric feature | Numeric model input |
| Categorical feature | One-hot encoded model input |
| Existing PCA feature | Already-transformed component (modelled as numeric, never interpreted) |
| Time / timestamp | Elapsed seconds, calendar timestamp or ordinal order |
| Transaction ID | Record identifier (shown in the Anomaly Explorer) |
| Group / entity ID | Entity key for group-aware splitting |
| Ignore | Not used |

Suggestions are only suggestions; the schema is validated live and must be confirmed before any
statistics are computed. Changing the schema creates a new schema version (hash); results are
keyed by it.

## 3. Explore

*Exploratory Analysis* tabs:

- **Descriptive statistics** – record/feature counts, prevalence, imbalance ratio, missing values,
  duplicates, quality warnings, sortable/filterable statistics table.
- **Class imbalance** – counts, shares, minority:majority ratio, majority-class baseline accuracy
  and why accuracy misleads for rare events.
- **Amount / value** – mean, median, std, quartiles, skewness, histograms (linear / log), box plots,
  statistics and non-parametric tests by class, positive rate by deciles.
- **Time** – events and positives over time, positive rate per interval, hour-of-cycle profile,
  distribution by class. Elapsed-seconds columns are never presented as calendar time.
- **Features** – Pearson/Spearman heatmap, relevance indicators (KS, AUC, point-biserial r,
  Cohen's d, mutual information), class-conditional distributions, PCA component properties,
  exploratory 2-D projection (visual sample, labelled as such).
- **Findings** – factual statements generated from the computed values.

## 4. Configure and run an experiment

*Experiment Studio*:

- **Mode** – Quick Benchmark (fixed bounded set), Full Benchmark (any selection), Hyperparameter
  Optimisation (RandomizedSearchCV on training folds).
- **Split** – linked sliders that always sum to 100 %, presets 80/10/10, 70/15/15, 60/20/20,
  stratified / random / chronological / group-aware methods, seed, live partition preview with
  positive counts and warnings. Impossible configurations are blocked.
- **Algorithms** – registry cards show score type, cost, resource caps and availability.
- **Imbalance strategies** – none, class weighting, random under/oversampling, SMOTE. Every
  selected strategy is run for each supervised model for a controlled comparison; inapplicable
  combinations are recorded as *skipped* with a reason.
- **Preprocessing**, **threshold policy**, **search settings** and options (permutation
  importance, Platt calibration, extended caps, novelty-mode anomaly detectors, JSON
  hyperparameter overrides).

Runs execute in a background worker; progress, current algorithm, elapsed time and errors are
shown live; experiments can be cancelled.

## 5. Compare and evaluate

- **ML Benchmark** – leaderboard per evaluation protocol (identical split), ranked by a validation
  objective you choose; test metrics are a final, single-use estimate. PR/ROC curves and metric bars.
- **Model Evaluation** – metrics for training (in-sample), validation and final test; confusion
  matrix; PR/ROC/calibration curves; native and permutation importance; **Threshold Studio**
  (validation by default, test view labelled exploratory); **Business impact** simulator with
  user-supplied hypothetical costs.
- **Anomaly Explorer** – record-level predictions with TP/FP/TN/FN filters, sorting, feature values
  of the selected record and CSV export.
- **Score New Data** – apply a trained pipeline to another uploaded dataset after a compatibility
  check (missing/unexpected columns, types, unseen categories, PCA provenance warnings, drift).

## 6. Report

*Research Reports* builds a bundle with `report.md`, `report.html`, CSV tables, PNG/SVG figures and
a manifest; optionally including an experiment. For fraud-domain schemas an analysis-coverage
appendix is added. Download the bundle as ZIP or export it to the
folder configured in `FTAD_EXPORT_DIR` (copied into `ftad_exports/`).

## Interpreting results responsibly

- PR-AUC should be compared with the prevalence (the PR-AUC of a random ranking).
- Probabilities from class-weighted or resampled models are not calibrated.
- Anomaly scores are not probabilities, and an anomaly is not necessarily fraud.
- Associations in EDA are not causal effects; large amounts are not by themselves evidence of fraud.
- With few positives, test metrics carry wide uncertainty.
