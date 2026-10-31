# Synthetic demo dataset

> **SYNTHETIC DATA.** `synthetic_transactions.csv` is randomly generated. It contains no real transactions,
> is unrelated to the Kaggle credit-card dataset, and results obtained on it say nothing about real fraud.

It lets you try every FTAD screen without downloading anything. All UI screenshots in this repository use it.

| Property | Value |
|---|---|
| Rows | 12,000 |
| Columns | `Time`, `V1`–`V10`, `Amount`, `Class` |
| Positive class | `Class = 1`, 185 rows (≈ 1.5 %) |
| `Time` | Elapsed seconds over 48 hours, with two day-like volume cycles |
| `V1`–`V10` | Anonymous numeric components; `V1`–`V4` are shifted for positives, with overlap |
| `Amount` | Right-skewed values; positives are more frequent among very small and very large amounts |

Regenerate it deterministically (seed 2026):

```bash
backend/.venv/bin/python demo/synthetic/generate_synthetic.py
```

Suggested schema in FTAD:

- `Class` → target, positive label `1`
- `Time` → time (elapsed seconds)
- `Amount` → numeric, and the amount/value column
- `V1`–`V10` → existing PCA feature
