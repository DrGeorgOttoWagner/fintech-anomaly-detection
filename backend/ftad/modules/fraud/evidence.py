"""Analysis-coverage appendix for fraud-domain reports.

Each row names an analysis that a fraud-detection study normally covers, points to the report
section, figure or table that holds the computed evidence and states whether that evidence exists
for the current dataset/schema.
"""

from __future__ import annotations

from ftad.core.reporting.report import register_evidence_provider


@register_evidence_provider
def analysis_coverage(doc, ctx, evidence: dict, ov, value, time_res, completed) -> None:
    cfg = ctx.cfg
    if cfg.domain != "fraud":
        return
    cb = ov.get("class_balance") or {}
    have_time = bool(time_res and time_res.get("available"))
    rows = [
        ["C1", "Dataset identification", f"{ov['n_records']:,} records × {ov['n_columns']} columns; fingerprint in Section 1", evidence.get("dataset"), True],
        ["C2", "Descriptive statistics", "Summary statistics for all numeric columns and by class", evidence.get("descriptive"), "descriptive" in evidence],
        ["C3", f"Time ('{cfg.time_column}'), value ('{cfg.amount_column}') and target ('{cfg.target}') analyses",
         "Dedicated analyses of each of the three columns", ", ".join(filter(None, [evidence.get("time"), evidence.get("amount"), evidence.get("class_distribution")])),
         bool(have_time and value and cb)],
        ["C4", "Class-imbalance quantification", f"Prevalence {cb.get('prevalence', 0):.4%}; ratio 1 : {cb.get('majority_to_minority', 0):,.0f}", evidence.get("class_distribution"), bool(cb)],
        ["C5", "Consequences of imbalance for evaluation", f"Majority baseline accuracy {cb.get('majority_baseline_accuracy', 0):.4%} with 0% recall; metric choice; resampling caveats"
         + ("; model comparison" if completed else ""), evidence.get("imbalance"), bool(cb)],
        ["C6", "Exploratory data analysis", "Distributions, box plots, temporal patterns, correlations, feature separation, PCA properties", evidence.get("eda"), "eda" in evidence],
        ["C7", "Findings and limitations", "Computed findings and limitations", "Sections 11–12", True],
        ["C8 (optional)", "Supervised and unsupervised model benchmark", "Leakage-safe model comparison with frozen thresholds", evidence.get("models"), bool(completed)],
    ]
    doc.h(1, "Appendix A. Analysis coverage")
    doc.note("'Evidence present' means the corresponding computation exists in this report; "
             "interpreting the results remains the analyst's responsibility.")
    doc.table(["ID", "Analysis", "Evidence", "Location", "Evidence present"], rows)
