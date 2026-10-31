# Metrics for rare-event detection

Notation for one evaluation set:

- **P** positives (e.g. frauds) and **N** negatives;
- a model produces a score per record;
- an **alert** is raised when the score is at or above a threshold *t*.

| | Predicted positive (alert) | Predicted negative |
|---|---|---|
| **Actual positive** | TP (true positive) | FN (false negative, missed case) |
| **Actual negative** | FP (false positive, false alarm) | TN (true negative) |

## Prevalence

`prevalence = P / (P + N)`

Prevalence is the share of positives. It sets the scale for everything below: with a prevalence of
0.17 %, about one record in 580 is positive.

## Accuracy, and why it misleads

`accuracy = (TP + TN) / (P + N)`

A model that never raises an alert has TP = 0 and FP = 0, so its accuracy is `N / (P + N) = 1 − prevalence`.

- At 0.17 % prevalence, that is **99.83 % accuracy with zero detected positives**.
- Under extreme imbalance, accuracy mostly measures the majority class.
- FTAD reports the majority-baseline accuracy explicitly, so that it is never mistaken for skill.

## Precision, recall and F1 (threshold metrics)

| Metric | Formula | Question it answers |
|---|---|---|
| **Precision** | TP / (TP + FP) | Of the alerts raised, how many were real? (alert quality, review workload) |
| **Recall** | TP / (TP + FN) | Of all real positives, how many were caught? (coverage) |
| **F1** | 2 · precision · recall / (precision + recall) | One summary that balances the two |

All three depend on the threshold.

- Lowering the threshold usually raises recall and lowers precision.
- If a model raises no alerts, its precision is **undefined** (0/0). FTAD shows it as *n/a*, not 0.
- FTAD chooses the threshold on **validation** data and freezes it before the single test evaluation.

## ROC-AUC (ranking metric)

ROC-AUC is the area under the curve of true-positive rate (recall) against false-positive rate,
FP / N. Equivalently, it is the probability that a random positive is scored above a random negative.

- It is threshold-free and does not depend on prevalence.
- That is also its weakness here: with N in the hundreds of thousands, ranking many negatives above a
  positive barely moves the false-positive rate. A model can therefore reach ROC-AUC above 0.9 while
  most of its alerts are false.

## Average precision (AP)

AP summarises the precision–recall curve as a weighted mean of precision at each recall step:

`AP = Σₙ (Rₙ − Rₙ₋₁) · Pₙ`

Here *Pₙ* and *Rₙ* are the precision and recall at the n-th threshold. FTAD computes AP with
scikit-learn's `average_precision_score`. The UI labels it *PR-AUC*.

Why AP is informative when positives are rare:

- **It is built from precision**, so every false alarm counts, however large N is.
- **Its random baseline equals the prevalence.** An AP of 0.80 at a prevalence of 0.0017 is about 470× better than chance.
- **It is the step-wise sum, not trapezoidal interpolation.** Linear interpolation between PR points
  is optimistic. On the credit-card showcase, a constant-score baseline gets a trapezoidal PR-AUC of
  about 0.50 but an AP of 0.0017.

FTAD ranks models by **validation AP** by default.

## Reading the confusion matrix

At the frozen threshold:

- **FP** is the workload created for reviewers;
- **FN** is the cost left undetected.

Which error costs more depends on the application. FTAD's *Business impact* simulator lets you enter
hypothetical per-error costs. FTAD does not assume any.

## Uncertainty

With few positives, one positive more or less changes recall noticeably. For example, with 74 test
positives, one case is worth 1.35 recall points. FTAD v1.0.0 reports point estimates from one split per
experiment. Repeated splits or bootstrap confidence intervals are on the roadmap.
