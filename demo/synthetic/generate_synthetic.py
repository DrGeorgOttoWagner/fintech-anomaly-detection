"""Generate FTAD's synthetic demo dataset (synthetic_transactions.csv).

SYNTHETIC DATA — every value is drawn from simple random distributions with a fixed seed. The file only
imitates the *shape* of a card-transaction table (elapsed time, anonymised numeric components, an amount
and a rare binary label). It contains no real transactions and must never be reported as research results.

Usage: python demo/synthetic/generate_synthetic.py   (requires numpy; writes next to this script)
"""

from pathlib import Path

import numpy as np

N, PREVALENCE, HOURS, SEED = 12_000, 0.015, 48, 2026
rng = np.random.default_rng(SEED)

# elapsed seconds with two day-like volume cycles (busier "daytime", quieter "night")
hour_weight = 0.35 + 0.65 * (np.sin((np.arange(HOURS) - 7) / 24 * 2 * np.pi) + 1) / 2
hours = rng.choice(HOURS, size=N, p=hour_weight / hour_weight.sum())
time = np.sort(hours * 3600 + rng.integers(0, 3600, size=N))

# rare positive class, slightly more frequent in quiet hours
quiet = hour_weight[time // 3600] < np.median(hour_weight)
p = np.where(quiet, PREVALENCE * 1.6, PREVALENCE * 0.7)
y = (rng.random(N) < p).astype(int)

# ten anonymised components; the first four shift for positives (with overlap)
shift = np.array([-2.2, 1.8, -1.5, 1.2, 0.4, 0, 0, 0, 0, 0])
scale = np.linspace(2.0, 0.6, 10)
V = rng.normal(0, 1, size=(N, 10)) * scale + np.outer(y, shift) * scale

# skewed amounts; positives slightly more frequent at very small and very large values
amount = np.exp(rng.normal(3.0, 1.2, size=N))
pos = y == 1
amount[pos] = np.where(rng.random(pos.sum()) < 0.4, rng.uniform(0.5, 3, pos.sum()), np.exp(rng.normal(4.2, 1.3, pos.sum())))

out = Path(__file__).with_name("synthetic_transactions.csv")
with out.open("w") as f:
    f.write("Time," + ",".join(f"V{i}" for i in range(1, 11)) + ",Amount,Class\n")
    for t, v, a, c in zip(time, V, amount, y):
        f.write(f"{t}," + ",".join(f"{x:.5f}" for x in v) + f",{a:.2f},{c}\n")
print(f"wrote {out.name}: {N} rows, {y.sum()} positives ({y.mean():.3%})")
