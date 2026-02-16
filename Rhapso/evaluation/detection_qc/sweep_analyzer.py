"""
Parameter sweep analysis for interest point detection QC.

Aggregates per-view metrics across sweep trials and produces a
summary comparing different (multiscale, sigma) parameter combinations.

The sweep summary uses a labeled metrics format where each metric is
a dict with 'name', 'value', 'description', and optional context keys.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from Rhapso.evaluation.detection_qc.view_metrics import ViewIPMetrics

logger = logging.getLogger(__name__)


@dataclass
class SweepTrialResult:
    """Results from a single parameter combination trial.

    Parameters
    ----------
    multiscale : str
        Multiscale level tried (e.g. "3").
    sigma : float
        Sigma value tried.
    trial_index : int
        Order in which this trial was attempted (0-indexed).
    success : bool
        Whether >= 50% of views met the target.
    n5_output_path : str
        Path where detection output was written.
    view_metrics : list of ViewIPMetrics
        Per-view QC metrics for this trial.
    """

    multiscale: str
    sigma: float
    trial_index: int
    success: bool
    n5_output_path: str
    view_metrics: List[ViewIPMetrics] = field(default_factory=list)

    @property
    def total_ip_count(self) -> int:
        """Total interest points across all views."""
        return sum(vm.ip_count for vm in self.view_metrics)

    @property
    def views_meeting_target(self) -> int:
        """Number of views that met the IP target."""
        return sum(1 for vm in self.view_metrics if vm.meets_target)

    @property
    def num_views(self) -> int:
        """Number of views analyzed."""
        return len(self.view_metrics)

    @property
    def success_rate(self) -> float:
        """Fraction of views meeting target (0.0 to 1.0)."""
        if not self.view_metrics:
            return 0.0
        return self.views_meeting_target / len(self.view_metrics)

    @property
    def mean_ip_count(self) -> float:
        """Mean IP count per view."""
        if not self.view_metrics:
            return 0.0
        return self.total_ip_count / len(self.view_metrics)

    @property
    def mean_density(self) -> float:
        """Mean spatial density across views."""
        if not self.view_metrics:
            return 0.0
        return sum(vm.density for vm in self.view_metrics) / len(self.view_metrics)

    def to_metric_list(self) -> list:
        """Serialize trial-level metrics as labeled dicts.

        Returns
        -------
        list of dict
            Each dict has 'name', 'value', 'description' and context keys.
        """
        return [
            {
                "name": "multiscale",
                "value": self.multiscale,
                "description": "Zarr pyramid level used for detection",
                "trial_index": self.trial_index,
            },
            {
                "name": "sigma",
                "value": self.sigma,
                "description": "DoG sigma parameter for blob detection",
                "trial_index": self.trial_index,
            },
            {
                "name": "success",
                "value": self.success,
                "description": "Whether >= 50% of views met the IP target",
                "trial_index": self.trial_index,
            },
            {
                "name": "num_views",
                "value": self.num_views,
                "description": "Number of tile views analyzed",
                "trial_index": self.trial_index,
            },
            {
                "name": "views_meeting_target",
                "value": self.views_meeting_target,
                "description": "Number of views with IP count >= target",
                "trial_index": self.trial_index,
            },
            {
                "name": "success_rate",
                "value": round(self.success_rate, 4),
                "description": "Fraction of views meeting target (0.0-1.0)",
                "trial_index": self.trial_index,
            },
            {
                "name": "total_ip_count",
                "value": self.total_ip_count,
                "description": "Total interest points across all views",
                "trial_index": self.trial_index,
            },
            {
                "name": "mean_ip_count",
                "value": round(self.mean_ip_count, 2),
                "description": "Mean interest points per view",
                "trial_index": self.trial_index,
            },
            {
                "name": "mean_density",
                "value": round(self.mean_density, 8),
                "description": "Mean spatial density (IPs per unit volume)",
                "trial_index": self.trial_index,
            },
        ]

    def to_dict(self) -> dict:
        """Full serialization including per-view metrics."""
        return {
            "multiscale": self.multiscale,
            "sigma": self.sigma,
            "trial_index": self.trial_index,
            "success": self.success,
            "n5_output_path": self.n5_output_path,
            "metrics": self.to_metric_list(),
            "view_metrics": [vm.to_dict() for vm in self.view_metrics],
        }


class SweepAnalyzer:
    """Analyze a collection of sweep trial results.

    Parameters
    ----------
    trials : list of SweepTrialResult
        All trials from the parameter sweep.
    target_interest_points : int
        The IP target used during the sweep.
    """

    def __init__(
        self,
        trials: List[SweepTrialResult],
        target_interest_points: int,
    ):
        self._trials = trials
        self._target_interest_points = target_interest_points

    def get_selected_trial(self) -> Optional[SweepTrialResult]:
        """Return the first successful trial, or None if all failed."""
        for trial in self._trials:
            if trial.success:
                return trial
        return None

    def get_summary(self) -> dict:
        """Produce a sweep summary with labeled metrics.

        Returns
        -------
        dict
            Summary with 'summary_metrics' (list of labeled dicts)
            and 'trials' (list of per-trial dicts).
        """
        selected = self.get_selected_trial()
        num_succeeded = sum(1 for t in self._trials if t.success)

        summary_metrics = [
            {
                "name": "target_interest_points",
                "value": self._target_interest_points,
                "description": "IP count threshold for each view to pass",
            },
            {
                "name": "num_trials_attempted",
                "value": len(self._trials),
                "description": "Total parameter combinations tested",
            },
            {
                "name": "num_trials_succeeded",
                "value": num_succeeded,
                "description": "Trials where >= 50% of views met target",
            },
            {
                "name": "first_success_trial_index",
                "value": selected.trial_index if selected else None,
                "description": "Index of first successful trial (None if all failed)",
            },
            {
                "name": "selected_multiscale",
                "value": selected.multiscale if selected else None,
                "description": "Multiscale level of selected (first successful) trial",
            },
            {
                "name": "selected_sigma",
                "value": selected.sigma if selected else None,
                "description": "Sigma of selected (first successful) trial",
            },
        ]

        if selected:
            summary_metrics.extend([
                {
                    "name": "selected_total_ip_count",
                    "value": selected.total_ip_count,
                    "description": "Total IPs in the selected trial",
                },
                {
                    "name": "selected_success_rate",
                    "value": round(selected.success_rate, 4),
                    "description": "View success rate of the selected trial",
                },
                {
                    "name": "selected_mean_ip_count",
                    "value": round(selected.mean_ip_count, 2),
                    "description": "Mean IPs per view in the selected trial",
                },
            ])

        return {
            "summary_metrics": summary_metrics,
            "trials": [t.to_dict() for t in self._trials],
        }

    def get_all_view_metrics_flat(self) -> list:
        """Collect all view metrics across all trials as a flat list.

        Useful for cross-trial comparison and plotting.

        Returns
        -------
        list of dict
            Each dict has trial context (multiscale, sigma, trial_index)
            merged with the view metric dict.
        """
        flat = []
        for trial in self._trials:
            for vm in trial.view_metrics:
                entry = vm.to_dict()
                entry["trial_multiscale"] = trial.multiscale
                entry["trial_sigma"] = trial.sigma
                entry["trial_index"] = trial.trial_index
                entry["trial_success"] = trial.success
                flat.append(entry)
        return flat
