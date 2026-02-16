"""
Per-view interest point detection QC metrics.

Reads N5 interest point outputs and computes spatial distribution,
density, and intensity statistics for each view.

N5 output structure (written by Rhapso SaveInterestPoints):
    {n5_base}/tpId_{tp}_viewSetupId_{vs}/beads/interestpoints/
        id/   — uint64 sequential IDs
        loc/  — float64 (N, 3) XYZ locations
        intensities/ — float32 (N,) values

Also reads attributes.json for fast IP count (dimensions field).
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import zarr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ViewIPMetrics:
    """QC metrics for interest points detected in a single view.

    Parameters
    ----------
    view_id : str
        Directory name, e.g. "tpId_0_viewSetupId_5".
    ip_count : int
        Total number of interest points detected.
    spatial_extent_xyz : tuple
        Range (max - min) of IP coordinates in X, Y, Z.
    spatial_std_xyz : tuple
        Standard deviation of IP coordinates in X, Y, Z.
    density : float
        Points per unit volume (bounding box volume).
    intensity_mean : float
        Mean intensity of detected IPs.
    intensity_std : float
        Standard deviation of IP intensities.
    intensity_min : float
        Minimum IP intensity.
    intensity_max : float
        Maximum IP intensity.
    meets_target : bool
        Whether ip_count >= the target threshold.
    """

    view_id: str
    ip_count: int
    spatial_extent_xyz: tuple
    spatial_std_xyz: tuple
    density: float
    intensity_mean: float
    intensity_std: float
    intensity_min: float
    intensity_max: float
    meets_target: bool

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "view_id": self.view_id,
            "ip_count": self.ip_count,
            "spatial_extent_xyz": list(self.spatial_extent_xyz),
            "spatial_std_xyz": list(self.spatial_std_xyz),
            "density": round(self.density, 8),
            "intensity_mean": round(self.intensity_mean, 4),
            "intensity_std": round(self.intensity_std, 4),
            "intensity_min": round(self.intensity_min, 4),
            "intensity_max": round(self.intensity_max, 4),
            "meets_target": self.meets_target,
        }

    def to_metric_list(self) -> list:
        """Serialize as a list of labeled metric dicts.

        Each entry has 'name', 'value', and 'description' keys,
        plus optional extra keys for context.

        Returns
        -------
        list of dict
            Metrics with labeled keys suitable for JSON reporting.
        """
        return [
            {
                "name": "ip_count",
                "value": self.ip_count,
                "description": "Total interest points detected in this view",
                "view_id": self.view_id,
            },
            {
                "name": "spatial_extent_x",
                "value": round(self.spatial_extent_xyz[0], 2),
                "description": "Range of IP X coordinates (max - min, pixels)",
                "view_id": self.view_id,
            },
            {
                "name": "spatial_extent_y",
                "value": round(self.spatial_extent_xyz[1], 2),
                "description": "Range of IP Y coordinates (max - min, pixels)",
                "view_id": self.view_id,
            },
            {
                "name": "spatial_extent_z",
                "value": round(self.spatial_extent_xyz[2], 2),
                "description": "Range of IP Z coordinates (max - min, pixels)",
                "view_id": self.view_id,
            },
            {
                "name": "spatial_std_x",
                "value": round(self.spatial_std_xyz[0], 4),
                "description": "Std dev of IP X coordinates (pixels)",
                "view_id": self.view_id,
            },
            {
                "name": "spatial_std_y",
                "value": round(self.spatial_std_xyz[1], 4),
                "description": "Std dev of IP Y coordinates (pixels)",
                "view_id": self.view_id,
            },
            {
                "name": "spatial_std_z",
                "value": round(self.spatial_std_xyz[2], 4),
                "description": "Std dev of IP Z coordinates (pixels)",
                "view_id": self.view_id,
            },
            {
                "name": "density",
                "value": round(self.density, 8),
                "description": "Interest points per unit volume (bounding box)",
                "view_id": self.view_id,
            },
            {
                "name": "intensity_mean",
                "value": round(self.intensity_mean, 4),
                "description": "Mean intensity of detected interest points",
                "view_id": self.view_id,
            },
            {
                "name": "intensity_std",
                "value": round(self.intensity_std, 4),
                "description": "Standard deviation of IP intensities",
                "view_id": self.view_id,
            },
            {
                "name": "intensity_min",
                "value": round(self.intensity_min, 4),
                "description": "Minimum IP intensity",
                "view_id": self.view_id,
            },
            {
                "name": "intensity_max",
                "value": round(self.intensity_max, 4),
                "description": "Maximum IP intensity",
                "view_id": self.view_id,
            },
            {
                "name": "meets_target",
                "value": self.meets_target,
                "description": "Whether IP count meets the target threshold",
                "view_id": self.view_id,
            },
        ]


def _get_ip_count_from_attributes(attrs_path: Path) -> Optional[int]:
    """Read IP count from attributes.json (fast path, no array loading).

    Parameters
    ----------
    attrs_path : Path
        Path to attributes.json inside interestpoints/id/.

    Returns
    -------
    int or None
        IP count from the dimensions field, or None on error.
    """
    try:
        with open(attrs_path, "r") as f:
            attrs = json.load(f)
        return attrs.get("dimensions", [0])[-1]
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read attributes from {attrs_path}: {e}")
        return None


def _read_n5_array(n5_store_path: str, dataset_rel_path: str) -> Optional[np.ndarray]:
    """Read an array from an N5 store.

    Parameters
    ----------
    n5_store_path : str
        Path to the .n5 directory or parent directory containing N5 data.
    dataset_rel_path : str
        Relative path within the store to the dataset.

    Returns
    -------
    np.ndarray or None
        The array data, or None if not found.
    """
    try:
        store = zarr.N5Store(n5_store_path)
        root = zarr.open(store, mode="r")
        if dataset_rel_path in root:
            return root[dataset_rel_path][:]
        return None
    except Exception as e:
        logger.warning(f"Could not read {dataset_rel_path} from {n5_store_path}: {e}")
        return None


def _find_n5_store_and_rel_path(full_path: str) -> Optional[tuple]:
    """Split a full path into N5 store path and relative dataset path.

    Parameters
    ----------
    full_path : str
        Full path like '/scratch/.../interestpoints.n5/tpId_0_.../beads/interestpoints/loc'.

    Returns
    -------
    tuple of (str, str) or None
        (n5_store_path, dataset_rel_path), or None if no .n5 found.
    """
    parts = full_path.split("/")
    for i, part in enumerate(parts):
        if part.endswith(".n5"):
            store_path = "/".join(parts[: i + 1])
            rel_path = "/".join(parts[i + 1 :])
            return store_path, rel_path
    return None


def compute_view_metrics(
    n5_base_path: str,
    view_dir_name: str,
    target_interest_points: int,
    compute_spatial: bool = True,
) -> Optional[ViewIPMetrics]:
    """Compute QC metrics for a single view's IP detection output.

    Parameters
    ----------
    n5_base_path : str
        Path to the output directory containing tpId_ subdirectories.
        May contain an interestpoints.n5 subdirectory, or the tpId_ dirs
        may be directly under this path.
    view_dir_name : str
        Directory name like "tpId_0_viewSetupId_5".
    target_interest_points : int
        Threshold for the meets_target flag.
    compute_spatial : bool
        If True, read full loc/intensities arrays for spatial metrics.
        If False, only read attributes.json for IP count (faster).

    Returns
    -------
    ViewIPMetrics or None
        Computed metrics, or None if data could not be read.
    """
    base = Path(n5_base_path)

    # Find the interestpoints location — may be under interestpoints.n5/ or beads/
    # Try common patterns
    candidates = [
        base / view_dir_name / "beads" / "interestpoints",
        base / "interestpoints.n5" / view_dir_name / "beads" / "interestpoints",
    ]

    ip_dir = None
    for candidate in candidates:
        if candidate.exists():
            ip_dir = candidate
            break

    if ip_dir is None:
        logger.debug(f"No interestpoints directory found for {view_dir_name} in {n5_base_path}")
        return None

    # Fast path: get count from attributes.json
    attrs_path = ip_dir / "id" / "attributes.json"
    ip_count = _get_ip_count_from_attributes(attrs_path)
    if ip_count is None:
        ip_count = 0

    if not compute_spatial or ip_count == 0:
        return ViewIPMetrics(
            view_id=view_dir_name,
            ip_count=ip_count,
            spatial_extent_xyz=(0.0, 0.0, 0.0),
            spatial_std_xyz=(0.0, 0.0, 0.0),
            density=0.0,
            intensity_mean=0.0,
            intensity_std=0.0,
            intensity_min=0.0,
            intensity_max=0.0,
            meets_target=ip_count >= target_interest_points,
        )

    # Full path: read loc and intensities arrays
    loc_path = str(ip_dir / "loc")
    intensities_path = str(ip_dir / "intensities")

    loc_result = _find_n5_store_and_rel_path(loc_path)
    int_result = _find_n5_store_and_rel_path(intensities_path)

    loc_data = None
    int_data = None

    if loc_result:
        loc_data = _read_n5_array(loc_result[0], loc_result[1])
    if int_result:
        int_data = _read_n5_array(int_result[0], int_result[1])

    # If N5 store approach didn't work, try direct zarr open
    if loc_data is None:
        try:
            store = zarr.N5Store(str(ip_dir.parent.parent.parent / "interestpoints.n5"))
            root = zarr.open(store, mode="r")
            rel_base = f"{view_dir_name}/beads/interestpoints"
            if f"{rel_base}/loc" in root:
                loc_data = root[f"{rel_base}/loc"][:]
            if f"{rel_base}/intensities" in root:
                int_data = root[f"{rel_base}/intensities"][:]
        except Exception:
            pass

    # Compute spatial metrics from loc array
    if loc_data is not None and len(loc_data) > 0:
        ip_count = len(loc_data)
        extent = np.ptp(loc_data, axis=0)
        spatial_extent = tuple(float(v) for v in extent)
        spatial_std = tuple(float(v) for v in np.std(loc_data, axis=0))
        volume = float(np.prod(np.maximum(extent, 1.0)))
        density = ip_count / volume
    else:
        spatial_extent = (0.0, 0.0, 0.0)
        spatial_std = (0.0, 0.0, 0.0)
        density = 0.0

    # Compute intensity metrics
    if int_data is not None and len(int_data) > 0:
        intensity_mean = float(np.mean(int_data))
        intensity_std = float(np.std(int_data))
        intensity_min = float(np.min(int_data))
        intensity_max = float(np.max(int_data))
    else:
        intensity_mean = 0.0
        intensity_std = 0.0
        intensity_min = 0.0
        intensity_max = 0.0

    return ViewIPMetrics(
        view_id=view_dir_name,
        ip_count=ip_count,
        spatial_extent_xyz=spatial_extent,
        spatial_std_xyz=spatial_std,
        density=density,
        intensity_mean=intensity_mean,
        intensity_std=intensity_std,
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        meets_target=ip_count >= target_interest_points,
    )


def compute_all_view_metrics(
    n5_base_path: str,
    target_interest_points: int,
    compute_spatial: bool = True,
) -> List[ViewIPMetrics]:
    """Compute QC metrics for all views found under n5_base_path.

    Discovers tpId_*_viewSetupId_* directories automatically by
    searching recursively for attributes.json files (matching the
    pattern used by R2R's _check_ip_detection_success).

    Parameters
    ----------
    n5_base_path : str
        Path to the output directory.
    target_interest_points : int
        Threshold for meets_target flag.
    compute_spatial : bool
        If True, read full arrays for spatial/intensity metrics.

    Returns
    -------
    list of ViewIPMetrics
        One entry per discovered view, sorted by view_id.
    """
    base = Path(n5_base_path)
    metrics = []

    # Discover view directories by finding attributes.json files
    view_dirs = set()
    for attrs_file in base.rglob("attributes.json"):
        if attrs_file.parent.name != "id":
            continue
        if "interestpoints" not in attrs_file.parent.parent.name:
            continue
        # Walk up to find tpId_*_viewSetupId_* directory
        for part in attrs_file.parts:
            if part.startswith("tpId_") and "viewSetupId" in part:
                view_dirs.add(part)
                break

    for view_dir in sorted(view_dirs):
        result = compute_view_metrics(
            n5_base_path=n5_base_path,
            view_dir_name=view_dir,
            target_interest_points=target_interest_points,
            compute_spatial=compute_spatial,
        )
        if result is not None:
            metrics.append(result)

    logger.info(
        f"Computed metrics for {len(metrics)} views from {n5_base_path}"
    )
    return metrics
