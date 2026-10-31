# Changelog

All notable changes to FinTech Anomaly Detection (FTAD) are documented here.
The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

Documentation and presentation only; no change to models, metrics or methodology.

### Added
- Documentation set: getting started, user guide, generic research workflow, metrics guide, architecture
  and reproducibility (`docs/`).
- Credit-card fraud research showcase with workflow, methodology, results, aggregate tables and figures
  (`examples/credit-card-fraud/`). The dataset itself is not included.
- Synthetic demo dataset and generator (`demo/synthetic/`); all UI screenshots regenerated on it.

### Changed
- The fraud-domain report appendix is now an "Analysis coverage" table.
- Documentation file names are lower-case; author and maintainer metadata were added.

## [1.0.0] — 2026-10-31

First stable local research release. Feature scope frozen.

**Platform:** macOS only — tested on macOS / Apple Silicon (Apple M3 Pro, macOS 27, Python 3.12,
Node.js 25). Other operating systems are not supported in this release.

### Features
- Dataset upload (file picker / drag & drop) with progress, CSV/Parquet validation, SHA-256
  fingerprint, duplicate detection, immutable raw copy and an explicit dataset session.
- Universal schema mapping with eight column roles, explicit positive label and live validation.
- Exploratory analysis on full data: descriptive statistics, class imbalance, amount/value, time,
  correlations, relevance indicators, precomputed-PCA properties, generated findings.
- Experiment Studio: Quick / Full / Hyperparameter-optimisation modes, linked train/validation/test
  sliders, stratified / random / chronological / group-aware splits, 20 model adapters,
  5 class-imbalance strategies, configurable preprocessing and threshold policies, asynchronous
  worker with progress and cancellation.
- ML Benchmark leaderboard per evaluation protocol, Model Evaluation (curves, confusion matrices,
  importance, Threshold Studio, hypothetical cost simulator), Anomaly Explorer with CSV export.
- Scoring of new datasets with persisted pipelines and compatibility checks.
- Research reports (Markdown, HTML, CSV, PNG/SVG, manifest) with an analysis-coverage appendix
  for fraud-domain schemas; ZIP download and export to a configured local folder.
- Reproducibility: dataset hash, schema hash, split indices, seeds, configuration, package versions
  and fitted pipelines persisted per experiment; one-click reproduction.

### Release hardening
- Internal identifiers (dataset/schema hashes, experiment, protocol and report IDs) and absolute
  local paths removed from ordinary UI screens; they remain in records, manifests and reports.
- Validation errors from the API are shown as readable sentences.
- Hyperparameter JSON overrides moved under an "Advanced" disclosure.
- New reports: the binary target is presented in the class-distribution section instead of the
  continuous-variable statistics table; print-friendly HTML (figures kept with captions, no
  orphaned headings, A4 page setup). Previously generated reports are unchanged.

