"""
Detection QC — Quality control utilities for interest point detection.

Provides per-view metrics computation, parameter sweep analysis,
and diagnostic plotting for IP detection outputs.
"""
from Rhapso.evaluation.detection_qc.view_metrics import (
    ViewIPMetrics,
    compute_view_metrics,
    compute_all_view_metrics,
)
from Rhapso.evaluation.detection_qc.sweep_analyzer import (
    SweepTrialResult,
    SweepAnalyzer,
)

__all__ = [
    "ViewIPMetrics",
    "compute_view_metrics",
    "compute_all_view_metrics",
    "SweepTrialResult",
    "SweepAnalyzer",
]
