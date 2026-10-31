# Extending FTAD

The research engine (`ftad.core`) is domain-agnostic. New domains, models and evaluation schemes are
added through registries (`ftad.core.interfaces`) rather than by editing the runner.

## Add a model (e.g. a custom PyTorch network)

Wrap the network as a scikit-learn compatible estimator (`fit`, `predict_proba` — e.g. with
[skorch](https://skorch.readthedocs.io) or a small `BaseEstimator` subclass), then register it:

```python
# ftad/modules/research/models.py
from ftad.core.interfaces import register_model
from ftad.core.models.registry import ModelSpec

def _factory(params, ctx):
    from mypackage.nets import FinancialNetClassifier   # your sklearn-compatible wrapper
    return FinancialNetClassifier(random_state=ctx.seed, **params)

register_model(ModelSpec(
    key="finnet", name="Financial NN (experimental)", family="neural", kind="supervised",
    library="pytorch", factory=_factory, default_params={"epochs": 20, "hidden": 64},
    search_space={}, needs_scaling=True, cost="high", requires=("torch",),
    notes="Experimental research model.",
))
```

Import the module from `ftad/modules/__init__.py`. The model then appears in the Experiment Studio
with availability status, and benefits from the same split, preprocessing, threshold freezing,
metrics, persistence and reports. Checkpoints can be stored in the run directory by the wrapper.

## Add a split strategy (walk-forward, expanding window, purged CV)

```python
import numpy as np
from ftad.core.interfaces import register_split, Split

def walk_forward(n, y, cfg, *, time=None, groups=None, embargo=0.01):
    order = np.argsort(time, kind="stable")
    n_tr, n_va = int(n * cfg.train), int(n * cfg.validation)
    gap = int(n * embargo)                                 # purge/embargo between partitions
    tr = order[:n_tr]
    va = order[n_tr + gap : n_tr + gap + n_va]
    te = order[n_tr + 2 * gap + n_va :]
    return Split(tr, va, te, "walk_forward", [f"Embargo of {gap} records between partitions."])

register_split("walk_forward", walk_forward)
```

`SplitConfig.method` accepts any registered name. For rolling multi-fold evaluation, extend the
runner's HPO step with a time-series CV object (the chronological method already uses
`TimeSeriesSplit` inside the training partition).

## Add a domain module

A domain module can contribute:

- **schema hints** — `register_hint_provider(fn)` refines suggested roles/terminology;
- **report evidence** — `register_evidence_provider(fn)` appends report sections such as the analysis-coverage appendix;
- **domain analytics/routers** — e.g. a market-anomaly module with its own FastAPI router.

`ftad.modules.fraud` is the reference implementation.

## Planned extension areas (not implemented in v1.0)

| Area | Hook |
|---|---|
| Regression / forecasting targets | new task type in `SchemaConfig` + metric family in `ftad.core.evaluation` |
| Multivariate time series windows | `DatasetAdapter` producing windowed features; walk-forward split |
| Financial market data | adapter for OHLCV sources; market-anomaly domain module |
| Model checkpoints / early stopping | model wrapper writing into the run directory |

## Integration boundaries (e.g. with other internal systems)

FTAD exposes a local REST API (`/api/docs`) and file artefacts (`model.joblib`, manifests). Any
production use would require independent validation, monitoring and deterministic safety controls
outside FTAD. FTAD has no trading, order or account functionality and makes no outbound calls.
