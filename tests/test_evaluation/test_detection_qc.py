"""Tests for Rhapso.evaluation.detection_qc module."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from Rhapso.evaluation.detection_qc.view_metrics import (
    ViewIPMetrics,
    compute_view_metrics,
    compute_all_view_metrics,
    _get_ip_count_from_attributes,
)
from Rhapso.evaluation.detection_qc.sweep_analyzer import (
    SweepTrialResult,
    SweepAnalyzer,
)
from Rhapso.evaluation.detection_qc.plotting import (
    generate_all_plots,
    plot_sweep_ip_counts,
    plot_sweep_success_rates,
)


@pytest.fixture
def mock_n5_dir(tmp_path):
    """Create a mock N5 directory structure with IP detection outputs."""
    views = {
        "tpId_0_viewSetupId_0": {
            "loc": np.array([[10.0, 20.0, 5.0], [30.0, 40.0, 10.0], [50.0, 60.0, 15.0]], dtype=np.float64),
            "intensities": np.array([100.0, 200.0, 300.0], dtype=np.float32),
        },
        "tpId_0_viewSetupId_1": {
            "loc": np.array([[5.0, 10.0, 2.0], [15.0, 25.0, 8.0]], dtype=np.float64),
            "intensities": np.array([150.0, 250.0], dtype=np.float32),
        },
    }

    n5_dir = tmp_path / "interestpoints.n5"
    n5_store = zarr.N5Store(str(n5_dir))
    root = zarr.open(n5_store, mode="w")

    for view_name, data in views.items():
        loc = data["loc"]
        intensities = data["intensities"]
        ip_count = len(loc)

        group_path = f"{view_name}/beads/interestpoints"
        root.create_dataset(f"{group_path}/loc", data=loc, chunks=loc.shape)
        root.create_dataset(f"{group_path}/intensities", data=intensities, chunks=intensities.shape)
        root.create_dataset(f"{group_path}/id", data=np.arange(ip_count, dtype=np.uint64), chunks=(ip_count,))

        # Write attributes.json for the id dataset (fast count path)
        id_dir = n5_dir / view_name / "beads" / "interestpoints" / "id"
        id_dir.mkdir(parents=True, exist_ok=True)
        attrs_path = id_dir / "attributes.json"
        with open(attrs_path, "w") as f:
            json.dump({"dimensions": [3, ip_count]}, f)

    return tmp_path


@pytest.fixture
def mock_empty_n5_dir(tmp_path):
    """Create a mock N5 directory with no views."""
    (tmp_path / "interestpoints.n5").mkdir()
    return tmp_path


@pytest.fixture
def sample_view_metrics():
    """Sample ViewIPMetrics for testing."""
    return [
        ViewIPMetrics(
            view_id="tpId_0_viewSetupId_0",
            ip_count=500,
            spatial_extent_xyz=(100.0, 80.0, 30.0),
            spatial_std_xyz=(25.0, 20.0, 8.0),
            density=0.002,
            intensity_mean=200.0,
            intensity_std=50.0,
            intensity_min=50.0,
            intensity_max=400.0,
            meets_target=True,
        ),
        ViewIPMetrics(
            view_id="tpId_0_viewSetupId_1",
            ip_count=100,
            spatial_extent_xyz=(60.0, 50.0, 20.0),
            spatial_std_xyz=(15.0, 12.0, 5.0),
            density=0.0017,
            intensity_mean=180.0,
            intensity_std=40.0,
            intensity_min=60.0,
            intensity_max=350.0,
            meets_target=False,
        ),
    ]


@pytest.fixture
def sample_trials(sample_view_metrics):
    """Sample sweep trials for testing."""
    return [
        SweepTrialResult(
            multiscale="5",
            sigma=4.0,
            trial_index=0,
            success=False,
            n5_output_path="/scratch/trial_0",
            view_metrics=[
                ViewIPMetrics(
                    view_id="v0", ip_count=50,
                    spatial_extent_xyz=(10.0, 10.0, 5.0),
                    spatial_std_xyz=(3.0, 3.0, 1.5),
                    density=0.1, intensity_mean=100.0, intensity_std=20.0,
                    intensity_min=50.0, intensity_max=200.0, meets_target=False,
                ),
            ],
        ),
        SweepTrialResult(
            multiscale="4",
            sigma=3.0,
            trial_index=1,
            success=False,
            n5_output_path="/scratch/trial_1",
            view_metrics=[
                ViewIPMetrics(
                    view_id="v0", ip_count=200,
                    spatial_extent_xyz=(40.0, 35.0, 15.0),
                    spatial_std_xyz=(10.0, 9.0, 4.0),
                    density=0.01, intensity_mean=150.0, intensity_std=30.0,
                    intensity_min=60.0, intensity_max=300.0, meets_target=False,
                ),
            ],
        ),
        SweepTrialResult(
            multiscale="3",
            sigma=2.5,
            trial_index=2,
            success=True,
            n5_output_path="/scratch/trial_2",
            view_metrics=sample_view_metrics,
        ),
    ]


# --- ViewIPMetrics tests ---


class TestViewIPMetrics:
    def test_to_dict(self, sample_view_metrics):
        d = sample_view_metrics[0].to_dict()
        assert d["view_id"] == "tpId_0_viewSetupId_0"
        assert d["ip_count"] == 500
        assert d["meets_target"] is True
        assert isinstance(d["spatial_extent_xyz"], list)

    def test_to_metric_list(self, sample_view_metrics):
        metrics = sample_view_metrics[0].to_metric_list()
        assert isinstance(metrics, list)
        assert len(metrics) == 13
        names = {m["name"] for m in metrics}
        assert "ip_count" in names
        assert "density" in names
        assert "meets_target" in names
        for m in metrics:
            assert "name" in m
            assert "value" in m
            assert "description" in m
            assert "view_id" in m

    def test_frozen(self, sample_view_metrics):
        with pytest.raises(AttributeError):
            sample_view_metrics[0].ip_count = 999


class TestGetIPCountFromAttributes:
    def test_valid_attributes(self, tmp_path):
        attrs_path = tmp_path / "attributes.json"
        with open(attrs_path, "w") as f:
            json.dump({"dimensions": [3, 42]}, f)
        assert _get_ip_count_from_attributes(attrs_path) == 42

    def test_missing_file(self, tmp_path):
        result = _get_ip_count_from_attributes(tmp_path / "nonexistent.json")
        assert result is None

    def test_malformed_json(self, tmp_path):
        attrs_path = tmp_path / "attributes.json"
        attrs_path.write_text("{bad json")
        assert _get_ip_count_from_attributes(attrs_path) is None


class TestComputeAllViewMetrics:
    def test_discovers_views(self, mock_n5_dir):
        metrics = compute_all_view_metrics(
            str(mock_n5_dir), target_interest_points=2, compute_spatial=False,
        )
        assert len(metrics) == 2
        view_ids = {m.view_id for m in metrics}
        assert "tpId_0_viewSetupId_0" in view_ids
        assert "tpId_0_viewSetupId_1" in view_ids

    def test_meets_target_flag(self, mock_n5_dir):
        metrics = compute_all_view_metrics(
            str(mock_n5_dir), target_interest_points=3, compute_spatial=False,
        )
        counts = {m.view_id: m.ip_count for m in metrics}
        targets = {m.view_id: m.meets_target for m in metrics}
        assert targets["tpId_0_viewSetupId_0"] is True  # 3 IPs == 3 target
        assert targets["tpId_0_viewSetupId_1"] is False  # 2 IPs < 3 target

    def test_empty_dir(self, mock_empty_n5_dir):
        metrics = compute_all_view_metrics(
            str(mock_empty_n5_dir), target_interest_points=10,
        )
        assert metrics == []


# --- SweepTrialResult tests ---


class TestSweepTrialResult:
    def test_properties(self, sample_trials):
        success_trial = sample_trials[2]
        assert success_trial.total_ip_count == 600
        assert success_trial.views_meeting_target == 1
        assert success_trial.num_views == 2
        assert success_trial.success_rate == 0.5
        assert success_trial.mean_ip_count == 300.0

    def test_empty_trial(self):
        trial = SweepTrialResult(
            multiscale="5", sigma=4.0, trial_index=0,
            success=False, n5_output_path="/scratch/empty",
        )
        assert trial.total_ip_count == 0
        assert trial.success_rate == 0.0
        assert trial.mean_ip_count == 0.0
        assert trial.mean_density == 0.0

    def test_to_dict(self, sample_trials):
        d = sample_trials[0].to_dict()
        assert d["multiscale"] == "5"
        assert d["sigma"] == 4.0
        assert "metrics" in d
        assert "view_metrics" in d

    def test_to_metric_list(self, sample_trials):
        metrics = sample_trials[2].to_metric_list()
        assert len(metrics) == 9
        names = {m["name"] for m in metrics}
        assert "success_rate" in names
        assert "total_ip_count" in names


# --- SweepAnalyzer tests ---


class TestSweepAnalyzer:
    def test_get_selected_trial(self, sample_trials):
        analyzer = SweepAnalyzer(sample_trials, target_interest_points=300)
        selected = analyzer.get_selected_trial()
        assert selected is not None
        assert selected.trial_index == 2
        assert selected.multiscale == "3"

    def test_no_success(self):
        trials = [
            SweepTrialResult(
                multiscale="5", sigma=4.0, trial_index=0,
                success=False, n5_output_path="/scratch/0",
            ),
        ]
        analyzer = SweepAnalyzer(trials, target_interest_points=500)
        assert analyzer.get_selected_trial() is None

    def test_get_summary_structure(self, sample_trials):
        analyzer = SweepAnalyzer(sample_trials, target_interest_points=300)
        summary = analyzer.get_summary()

        assert "summary_metrics" in summary
        assert "trials" in summary
        assert len(summary["trials"]) == 3

        metrics = summary["summary_metrics"]
        names = {m["name"]: m for m in metrics}
        assert names["num_trials_attempted"]["value"] == 3
        assert names["num_trials_succeeded"]["value"] == 1
        assert names["first_success_trial_index"]["value"] == 2
        assert names["selected_multiscale"]["value"] == "3"
        assert names["selected_sigma"]["value"] == 2.5

    def test_get_summary_all_failed(self):
        trials = [
            SweepTrialResult(
                multiscale="5", sigma=4.0, trial_index=0,
                success=False, n5_output_path="/scratch/0",
            ),
        ]
        analyzer = SweepAnalyzer(trials, target_interest_points=500)
        summary = analyzer.get_summary()
        names = {m["name"]: m for m in summary["summary_metrics"]}
        assert names["selected_multiscale"]["value"] is None
        assert names["selected_sigma"]["value"] is None
        assert names["first_success_trial_index"]["value"] is None

    def test_get_all_view_metrics_flat(self, sample_trials):
        analyzer = SweepAnalyzer(sample_trials, target_interest_points=300)
        flat = analyzer.get_all_view_metrics_flat()
        assert len(flat) == 4  # 1 + 1 + 2 views
        assert all("trial_multiscale" in entry for entry in flat)
        assert all("trial_sigma" in entry for entry in flat)

    def test_summary_is_json_serializable(self, sample_trials):
        analyzer = SweepAnalyzer(sample_trials, target_interest_points=300)
        summary = analyzer.get_summary()
        # Should not raise
        json_str = json.dumps(summary)
        assert isinstance(json_str, str)


# --- Plotting tests ---


class TestPlotting:
    def test_generate_all_plots(self, sample_trials, tmp_path):
        paths = generate_all_plots(
            trials=sample_trials,
            target_interest_points=300,
            output_dir=str(tmp_path / "plots"),
        )
        assert len(paths) >= 2
        for p in paths:
            assert os.path.exists(p)
            assert p.endswith(".png")

    def test_plot_sweep_ip_counts(self, sample_trials, tmp_path):
        path = plot_sweep_ip_counts(
            sample_trials, target_interest_points=300,
            output_dir=str(tmp_path),
        )
        assert os.path.exists(path)

    def test_plot_sweep_success_rates(self, sample_trials, tmp_path):
        path = plot_sweep_success_rates(
            sample_trials, output_dir=str(tmp_path),
        )
        assert os.path.exists(path)

    def test_empty_trials(self, tmp_path):
        paths = generate_all_plots(
            trials=[], target_interest_points=300,
            output_dir=str(tmp_path / "plots"),
        )
        assert paths == []
