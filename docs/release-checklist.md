# Release checklist — FinTech Anomaly Detection v1.0.0

Release date: 2026-10-31 · Platform: macOS only — **tested on macOS / Apple Silicon**
(Apple M3 Pro, macOS 27, Python 3.12.4, Node.js 25).

FTAD v1.0.0 is a stable local research software release. It is not a certified or production
fraud-detection service, and no trained model is claimed to be universally valid.

## Scope
- [x] Feature scope frozen; no new features, algorithms, pages or integrations
- [x] Scientific calculations and methodology unchanged
- [x] Previously generated reports left untouched
- [x] macOS only; no Windows/Linux packaging, Docker, cloud or server deployment

## UI cleanup
- [x] Dataset/schema hashes removed from the header; experiment, protocol and report IDs removed from lists and titles
- [x] Absolute local paths no longer shown (export folder, export destination, error messages)
- [x] Implementation metadata (in-memory size) removed; API validation errors shown as readable text
- [x] Developer-style JSON hyperparameter overrides placed under an "Advanced" disclosure
- [x] Binary target excluded from the continuous-statistics table (shown under Class imbalance)
- [x] Hashes, IDs, versions and errors still recorded in the database, manifests and reports

## Reports (new reports only)
- [x] Binary target presented in the class-distribution section, not in continuous statistics
- [x] Print-friendly HTML: A4 page setup, figures kept with captions, no orphaned headings

## Scientific safety (verified by automated tests)
- [x] Train/validation/test partitions disjoint and reproducible for a seed
- [x] Preprocessing fitted on training rows only (scaler statistics = training means)
- [x] Resampling inside the training pipeline only; validation/test prevalence unchanged
- [x] Thresholds and hyperparameter search use validation / training folds only; test scored once
- [x] Metrics match scikit-learn on actual predictions; undefined metrics reported as n/a
- [x] Re-running an experiment reproduces split hash and metrics
- [x] Dataset switching starts a new session with no stale results

## QA results

| Gate | Result |
|---|---|
| Backend tests (pytest) | 44 passed, 0 failed |
| Frontend type-check (tsc) + lint (oxlint) | pass (0 errors) |
| Production build (vite) | pass |
| Browser acceptance workflow — synthetic fixture | 1/1 passed |
| Browser acceptance workflow — public credit-card fraud dataset (local copy) | 1/1 passed |

Workflow steps covered by the browser test: 1 startup · 2 empty dataset state · 3 browser upload
(native file chooser) · 4 schema mapping · 5 EDA (descriptive, imbalance, amount, time, features,
findings) · 6 experiment configuration (linked sliders, presets) · 7 ML benchmark (Quick Benchmark,
API responsive during training) · 8 model evaluation and threshold changes · 9 report generation ·
10 export · 11 dataset switching with a different schema (no stale data, no fake metrics, no console errors).

## Repository
- [x] README, LICENSE (MIT), CHANGELOG, `.gitignore`, `.env.example`
- [x] Pinned dependencies: `backend/requirements.lock`, `frontend/package-lock.json`
- [x] Startup instructions (`scripts/setup.sh`, `scripts/start.sh`) and macOS requirements
- [x] Screenshots regenerated from the release build
- [x] No secrets, datasets, private documents, PDFs, runtime data or personal paths tracked
- [x] Release tagged as `v1.0.0`
