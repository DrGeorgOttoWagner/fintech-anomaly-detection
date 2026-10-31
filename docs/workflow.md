# A generic research workflow for rare-event detection

This procedure works for any binary, imbalanced tabular problem: payment fraud, account takeover,
insider anomalies, defect detection. Each phase lists the FTAD screen that supports it.
[`examples/credit-card-fraud/workflow.md`](../examples/credit-card-fraud/workflow.md) applies it to one dataset.

## Phase 1 — Dataset validation *(Datasets, Schema Mapping)*

1. **Load the dataset** and record its SHA-256 fingerprint. Every later result refers to exactly this file.
2. **Inspect the schema.** Check that types were parsed correctly and give each column one role.
3. **Verify the target.** Confirm which value is the positive class. A wrong positive label silently inverts every metric.
4. **Inspect missingness**, per column and in total.
5. **Inspect duplicates.** Exact duplicate rows can cross partition boundaries and inflate test scores (see Phase 7).

## Phase 2 — Descriptive analysis *(Exploratory Analysis → Descriptive statistics, Class imbalance)*

1. **Class distribution:**
   - counts, shares and the imbalance ratio;
   - the accuracy of always predicting the majority class.
2. **Prevalence:** this is the AP of a random ranking, so it is the baseline every model must beat.
3. **Feature distributions:** location, spread, skewness and extremes.
4. **Class-conditional summaries:** the same statistics, split by class.

## Phase 3 — Exploratory data analysis *(Amount, Time, Features, Findings)*

1. **Value patterns:**
   - distributions on linear and log scales;
   - the positive rate per value decile, to check whether the relation is monotonic.
2. **Temporal patterns:**
   - volume and positives over time;
   - the positive rate per interval.

   Never read elapsed seconds as clock time.
3. **Feature separation:** KS statistic, univariate AUC and mutual information per feature.
4. **Correlations** between features, and between each feature and the target.
5. **State findings as associations**, not causes.

## Phase 4 — Experimental design *(Experiment Studio)*

1. **Freeze the protocol before training.** Fix the split fractions, method, seed, algorithms,
   preprocessing, imbalance strategies, threshold policy and the selection metric, and save them.
2. **Use a train / validation / test split.**
   - Training data fit models and preprocessing.
   - Validation data select models and thresholds.
   - Test data are scored once.
3. **Avoid test leakage:**
   - fit preprocessing and resampling inside training only;
   - never tune on test;
   - consider duplicate-safe or group-aware splits.
4. **Choose the selection metric.** For rare positives, use validation AP rather than accuracy or ROC-AUC (see [metrics.md](metrics.md)).

## Phase 5 — Benchmarking *(Experiment Studio → ML Benchmark)*

1. **Supervised methods:** include a majority-class baseline and models of increasing capacity
   (linear models, trees, ensembles, boosting).
2. **Anomaly detectors** (Isolation Forest, LOF, One-Class SVM):
   - they are fitted without labels;
   - labels are used only to choose their alert threshold and to evaluate them.
3. **Select on validation.** Rank by validation AP, and treat test results as a single-use estimate.
4. **Record training budgets.** Models trained on capped subsamples are not directly comparable with full-data models.

## Phase 6 — Threshold evaluation *(Model Evaluation, Threshold Studio)*

1. Freeze the threshold on validation (default policy: maximise F1).
2. At that threshold, report on test:
   - precision and recall;
   - F1;
   - **false positives** (review workload);
   - **false negatives** (missed cases).
3. Remember that the right operating point depends on the real costs of each error. Max-F1 is a reproducible
   default, not a business optimum. The Business impact simulator uses hypothetical costs that you supply.

## Phase 7 — Robustness *(repeat Phase 4–6 with different protocols)*

1. **Duplicate-safe analysis:**
   - re-score test without records that have an exact copy in training;
   - or use a group-aware split that keeps identical records together.
2. **Chronological split:** train on earlier data and test on later data, which estimates performance under drift.
3. **Alternative allocation profiles**, e.g. 70/15/15, 80/10/10, 60/20/20.
   - Each profile has a different test set.
   - Compare stability, not just the best number.

## Phase 8 — Reporting *(Research Reports)*

1. Export the figures and tables together with the manifest: fingerprint, schema hash, split hash,
   configuration and library versions.
2. Keep the experiment protocol and the environment record next to the results.
3. Interpret cautiously:
   - small positive counts mean wide uncertainty;
   - one split per profile gives no confidence interval;
   - associations are not causes.
