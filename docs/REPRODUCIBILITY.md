# Reproducing experiments

Every experiment records what is needed to repeat it:

| Item | Where |
|---|---|
| Dataset SHA-256 fingerprint | `datasets.sha256`, experiment manifest |
| Schema configuration + hash | `schemas` table, manifest |
| Feature list, preprocessing, imbalance strategies, hyperparameters, threshold policy, HPO settings | experiment `config` (manifest) |
| Split method, fractions, seed, partition sizes, positives, **indices** | `split.npz`, `split.indices_sha256` |
| Package versions (Python, NumPy, pandas, scikit-learn, imbalanced-learn, XGBoost, LightGBM, CatBoost, …) | manifest, each `model.joblib` |
| Per-run threshold, metrics, curves, sweeps, importance, CV results | `runs.result_json`, `detail.json` |
| Fitted pipeline (preprocessing + model) and calibrator | `model.joblib` |
| Timestamps | experiment and run records |

## In the application

*Experiment Studio → history*, or the API: `POST /api/experiments/{id}/reproduce` re-runs an
experiment with the identical configuration on the same dataset fingerprint. The split hash must
match and metrics are identical for deterministic models (the test suite asserts this for Logistic
Regression, Random Forest, XGBoost and Isolation Forest).

`GET /api/experiments/{id}/manifest` returns the exportable JSON manifest; reports include it.

## From scratch on another machine

1. Install FTAD with the pinned dependencies (`backend/requirements.lock`, `frontend/package-lock.json`).
2. Upload the same file — confirm the SHA-256 shown in the UI matches the manifest.
3. Apply the schema from the manifest (same roles, target and positive label → same schema hash).
4. Create an experiment with the manifest's `config` (`POST /api/experiments`).
5. Compare `split.indices_sha256` and metrics.

## Sources of non-determinism

- Multi-threaded tree learners are deterministic for fixed seeds in the pinned versions
  (LightGBM runs with `deterministic=True`, `force_row_wise=True`); different library versions
  or CPU architectures can change results slightly.
- MLP and SGD depend on floating-point summation order; minor differences across platforms are possible.
- Timing measurements (training/inference seconds) naturally vary.
