# Reproducing the study in FTAD

The steps below repeat the study with FTAD's user interface. Screen names refer to the left-hand navigation.

## Step 1 — Download the dataset

Download `creditcard.csv` from
[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). This needs a
free Kaggle account. Keep the file outside the repository; `*.csv` files are git-ignored. The reference
file has this SHA-256:

```
76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
```

## Step 2 — Start FTAD

```bash
./scripts/start.sh        # http://127.0.0.1:8790
```

## Step 3 — Load `creditcard.csv`

On **Overview**, click **Browse Files** and choose the file. After parsing (about 284,807 rows × 31
columns), FTAD shows its fingerprint. Compare it with the value above.

## Step 4 — Map the schema

On **Schema Mapping**, check or set:

| Column | Role |
|---|---|
| `Class` | **Target**, positive label **`1`** (fraud) |
| `Time` | **Time**: elapsed seconds since the first record, not a calendar timestamp |
| `Amount` | **Numeric feature**; also select it as the amount/value column |
| `V1` … `V28` | **Existing PCA feature**: anonymised components, modelled as numeric and never interpreted |

Set the domain terminology to *Fraud detection*, then click **Confirm schema**.

## Step 5 — Run the EDA

Open **Exploratory Analysis** and work through the tabs:

- **Descriptive statistics:** missing values (0), duplicates (1,081 rows beyond their first occurrence).
- **Amount:** skewness, log-scale histograms and the fraud rate per decile.
- **Time:** volume and fraud per hour of elapsed time.
- **Features:** KS statistic per component and the correlation heatmap.
- **Findings.**

## Step 6 — Inspect the imbalance

Open the **Class imbalance** tab. You should see:

- 492 frauds, a prevalence of 0.1727 % and a ratio of about 1 : 578;
- a majority-baseline accuracy of 99.8273 % with a recall of 0.

## Step 7 — Configure the benchmark

In the **Experiment Studio**:

- **Mode:** Full Benchmark, with all 20 algorithms selected.
- **Split:** stratified, preset **70/15/15**, seed **42**.
- **Imbalance strategy:** none (class weighting can be compared as a separate run).
- **Preprocessing:** defaults. They are fitted on training rows only.
- **Threshold policy:** maximise F1 on validation (frozen).
- **Resource profile:** *safe*. Gradient Boosting, k-NN, SVM (RBF), LOF and One-Class SVM then train on
  capped stratified subsamples of 120,000, 50,000, 20,000, 30,000 and 10,000 rows.

## Step 8 — Run the models

Start the run. In the **ML Benchmark** leaderboard:

- rank by **validation AP**;
- treat the test columns as a single final estimate.

Open **Model Evaluation** for confusion matrices, curves and the Threshold Studio.

## Step 9 — Compare the three split profiles

Repeat Steps 7–8 with the presets **80/10/10** and **60/20/20** (same seed and settings). Then compare:

- **which model each profile selects** by validation AP;
- **how stable the leading group is** across profiles;
- **test AP and recall**, keeping in mind that each profile has a different test set (74, 49 and 99 test frauds).

## Step 10 — Robustness checks

- **Duplicate-safe split:** keep identical records in the same partition. The dataset has no entity
  column, so the study derived one that numbers groups of identical rows:

  ```python
  import pandas as pd
  df = pd.read_csv("creditcard.csv")
  df["dup_group"] = df.groupby(list(df.columns), sort=False, dropna=False).ngroup()
  df.to_csv("creditcard_with_dup_group.csv", index=False)
  ```

  1. Upload the derived file.
  2. Map `dup_group` as **Group / entity ID**; all other roles stay as in Step 4.
  3. Run the same profiles with the **group-aware** split method.
- **Chronological split:** choose the *chronological* method, 70/15/15. Training covers the earliest
  records, test the latest. In the study, this comparison used 29 input features (`V1`–`V28`,
  `Amount`), with `Time` used only to order the records (deselect it in the Studio's *Features* card).
  It was compared with a stratified 70/15/15 reference run on the same 29 features.
- Compare test AP and recall with the stratified reference run.

## Step 11 — Export the results

In **Research Reports**, generate a report that includes the experiment. The bundle contains:

- `report.md` and `report.html`;
- CSV tables and PNG/SVG figures;
- `manifest.json`, which holds the fingerprint, schema hash, split hash, configuration and library versions.

Download it as a ZIP, or export it to `FTAD_EXPORT_DIR`.

## Expected agreement

- **Deterministic models** should reproduce the published numbers with the same library versions and split hash.
- **Library or platform differences** can shift results slightly, especially for MLP and SGD.
- **Training times** vary between machines.

The published values are in [results.md](results.md) and [tables/](tables/).
