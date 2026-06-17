#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import boto3
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import zarr
from matplotlib.widgets import Slider

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


# ----------------------------
# Hardcoded params
# ----------------------------

XML_PATH = "s3://aind-scratch-data/sean.fite/exaSPIM_730904-test/9/rhapso-detection.xml"
INTERESTPOINTS_BASE = "s3://aind-scratch-data/sean.fite/exaSPIM_730904-test/9/interestpoints.n5"

SCALE_LEVEL = "4"
TIMEPOINT = 0
TRANSFORM_NAME = "Translation to Nominal Grid"
S3_ANON = False

# Pick a specific overlapping pair.
# If None, prints overlaps and uses the first one.
TARGET_PAIR: Optional[Tuple[int, int]] = None
# TARGET_PAIR = (5, 9)

# Padding around nominal overlap in full-res XY pixels.
# Set to 0 for strict overlap only.
OVERLAP_PAD_XY = 800

# Points are shown if within this many scaled Z slices of current z.
POINT_Z_RADIUS_SCALED = 1.5

# Display settings
DISPLAY_NORM_MODE = "global"  # "global" or "per_slice"
P_LO = 1
P_HI = 99

# Point style
POINT_A_COLOR = "red"
POINT_B_COLOR = "dodgerblue"
POINT_A_SIZE = 28
POINT_B_SIZE = 22
POINT_ALPHA = 0.85
POINT_EDGE_COLOR = "black"
POINT_LINEWIDTH = 0.4

# Keep image shape natural.
IMAGE_ASPECT = "equal"


# ----------------------------
# Match-readiness scoring params
# ----------------------------

# Broad search window around nominal position, in scaled px.
# If scale level 4 is ~16x, 50 scaled px ~= 800 full-res px.
MATCH_SEARCH_RADIUS_SCALED = 50.0

# Translation voting bin size, scaled px.
TRANSLATION_BIN_SCALED = 4.0

# After estimating dominant translation, score red/blue proximity at these radii.
MATCH_RADII_SCALED = (2.0, 4.0, 6.0, 8.0)

# Affine/RANSAC proxy thresholds.
AFFINE_INLIER_RADIUS_SCALED = 4.0
MIN_AFFINE_PAIRS = 12

# Spatial spread of affine proxy inliers.
INLIER_GRID_BINS = (4, 8, 8)


# ----------------------------
# Data model
# ----------------------------

@dataclass
class TileRecord:
    setup: int
    tp: int
    rel_path: str
    full_path: str
    size_x: int
    size_y: int
    size_z: int
    nominal: np.ndarray


@dataclass
class ScaleInfo:
    scale_x: float
    scale_y: float
    scale_z: float


@dataclass
class CropFullRes:
    x0: int
    x1: int
    y0: int
    y1: int


@dataclass
class CropScaled:
    x0: int
    x1: int
    y0: int
    y1: int


# ----------------------------
# XML / IO helpers
# ----------------------------

def load_xml_root(xml_path: str) -> ET.Element:
    if xml_path.startswith("s3://"):
        parsed = urlparse(xml_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return ET.fromstring(obj["Body"].read())

    return ET.parse(xml_path).getroot()


def affine_12_to_4x4(affine_text: str) -> np.ndarray:
    vals = [float(v) for v in affine_text.split()]
    if len(vals) != 12:
        raise RuntimeError(f"Expected 12 affine values, got {len(vals)}")

    mat = np.eye(4, dtype=np.float64)
    mat[0, 0:4] = vals[0:4]
    mat[1, 0:4] = vals[4:8]
    mat[2, 0:4] = vals[8:12]
    return mat


def parse_view_setup_sizes(root: ET.Element):
    sizes = {}

    for vs in root.findall(".//ViewSetup"):
        setup_id_text = vs.findtext("id")
        size_text = vs.findtext("size")

        if setup_id_text is None or size_text is None:
            continue

        setup_id = int(setup_id_text)
        sx, sy, sz = [int(v) for v in size_text.split()]
        sizes[setup_id] = (sx, sy, sz)

    return sizes


def parse_named_transforms_from_xml(root: ET.Element, transform_name: str):
    transforms = {}

    for vr in root.findall(".//ViewRegistration"):
        setup = int(vr.get("setup"))
        tp = int(vr.get("timepoint", 0))

        if tp != TIMEPOINT:
            continue

        for vt in vr.findall("ViewTransform"):
            name = vt.findtext("Name")
            affine_text = vt.findtext("affine")

            if name == transform_name and affine_text:
                transforms[setup] = affine_12_to_4x4(affine_text)
                break

    return transforms


def parse_zarr_tile_records(root: ET.Element, transform_name: str):
    setup_sizes = parse_view_setup_sizes(root)
    nominal_transforms = parse_named_transforms_from_xml(root, transform_name)

    image_loader = root.find(".//ImageLoader")
    if image_loader is None:
        raise RuntimeError("No <ImageLoader> found in XML")

    s3bucket = image_loader.findtext("s3bucket")
    zarr_base = image_loader.findtext("zarr")

    if not zarr_base:
        raise RuntimeError("No <zarr> base path found in XML ImageLoader")

    if zarr_base.startswith("s3://"):
        zarr_base = zarr_base.rstrip("/") + "/"
    elif s3bucket:
        zarr_base = f"s3://{s3bucket}/{zarr_base.lstrip('/')}".rstrip("/") + "/"
    elif XML_PATH.startswith("s3://"):
        parsed = urlparse(XML_PATH)
        zarr_base = f"s3://{parsed.netloc}/{zarr_base.lstrip('/')}".rstrip("/") + "/"
    else:
        zarr_base = zarr_base.rstrip("/") + "/"

    records = []

    for zg in image_loader.findall(".//zgroup"):
        rel_path = zg.get("path") or zg.findtext("path")
        if not rel_path:
            continue

        setup = int(zg.get("setup"))
        tp = int(zg.get("tp", zg.get("timepoint", 0)))

        if setup not in setup_sizes:
            raise RuntimeError(f"Missing ViewSetup size for setup {setup}")

        if setup not in nominal_transforms:
            raise RuntimeError(f"Missing transform '{transform_name}' for setup {setup}")

        sx, sy, sz = setup_sizes[setup]

        records.append(
            TileRecord(
                setup=setup,
                tp=tp,
                rel_path=rel_path,
                full_path=zarr_base + rel_path,
                size_x=sx,
                size_y=sy,
                size_z=sz,
                nominal=nominal_transforms[setup],
            )
        )

    return sorted(records, key=lambda r: r.setup)


def open_ome_zarr_level(zarr_path: str, scale_level: str):
    if zarr_path.startswith("s3://"):
        s3 = s3fs.S3FileSystem(anon=S3_ANON)
        store = s3fs.S3Map(root=zarr_path.rstrip("/"), s3=s3, check=False)
        return da.from_zarr(store, component=scale_level)

    return da.from_zarr(zarr_path.rstrip("/"), component=scale_level)


def open_zarr_or_n5_array(full_path: str):
    if full_path.startswith("s3://"):
        s3 = s3fs.S3FileSystem(anon=False)
        parsed = urlparse(full_path)
        components = parsed.path.lstrip("/").split("/")

        try:
            n5_index = next(i for i, c in enumerate(components) if c.endswith(".n5"))
        except StopIteration:
            store = s3fs.S3Map(root=full_path.rstrip("/"), s3=s3, check=False)
            return zarr.open_array(store, mode="r")

        dataset_root = f"s3://{parsed.netloc}/" + "/".join(components[: n5_index + 1])
        dataset_rel_path = "/".join(components[n5_index + 1:])

        store = s3fs.S3Map(root=dataset_root.rstrip("/"), s3=s3, check=False)
        root = zarr.open(store, mode="r")

        if dataset_rel_path not in root:
            raise KeyError(f"Dataset not found in S3 N5: {dataset_rel_path}")

        return root[dataset_rel_path]

    full_path = full_path.rstrip("/")
    components = full_path.split("/")

    try:
        n5_index = next(i for i, c in enumerate(components) if c.endswith(".n5"))
    except StopIteration:
        return zarr.open_array(full_path, mode="r")

    dataset_path = "/".join(components[: n5_index + 1])
    dataset_rel_path = "/".join(components[n5_index + 1:])

    store = zarr.N5Store(dataset_path)
    root = zarr.open(store, mode="r")

    if dataset_rel_path not in root:
        raise KeyError(f"Dataset not found in N5: {dataset_rel_path}")

    return root[dataset_rel_path]


def read_interest_points(base_path: str, setup: int, timepoint: int):
    loc_path = (
        f"{base_path.rstrip('/')}/"
        f"tpId_{timepoint}_viewSetupId_{setup}/"
        f"beads/interestpoints/loc"
    )

    arr = open_zarr_or_n5_array(loc_path)
    pts = np.asarray(arr[:], dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError(f"Expected loc shape N x 3 for setup {setup}, got {pts.shape}")

    return pts


# ----------------------------
# Geometry
# ----------------------------

def open_tile_volume_zyx(rec: TileRecord):
    arr = open_ome_zarr_level(rec.full_path, SCALE_LEVEL)

    if arr.ndim == 5:
        return arr[0, 0, :, :, :].astype(np.float32)

    if arr.ndim == 3:
        return arr.astype(np.float32)

    raise RuntimeError(f"Unexpected array shape for {rec.full_path}: {arr.shape}")


def infer_scale_info(rec: TileRecord, vol_zyx: da.Array) -> ScaleInfo:
    z_scaled, y_scaled, x_scaled = vol_zyx.shape

    return ScaleInfo(
        scale_x=float(rec.size_x) / float(x_scaled),
        scale_y=float(rec.size_y) / float(y_scaled),
        scale_z=float(rec.size_z) / float(z_scaled),
    )


def get_nominal_xy_box_fullres(rec: TileRecord):
    tx = float(rec.nominal[0, 3])
    ty = float(rec.nominal[1, 3])

    return (
        tx,
        tx + float(rec.size_x),
        ty,
        ty + float(rec.size_y),
    )


def compute_pairwise_overlap_fullres(rec_a: TileRecord, rec_b: TileRecord):
    ax0, ax1, ay0, ay1 = get_nominal_xy_box_fullres(rec_a)
    bx0, bx1, by0, by1 = get_nominal_xy_box_fullres(rec_b)

    ox0 = max(ax0, bx0)
    ox1 = min(ax1, bx1)
    oy0 = max(ay0, by0)
    oy1 = min(ay1, by1)

    if ox1 <= ox0 or oy1 <= oy0:
        return None

    return ox0, ox1, oy0, oy1


def list_overlapping_pairs(records: List[TileRecord]):
    overlaps = []

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            rec_a = records[i]
            rec_b = records[j]

            overlap = compute_pairwise_overlap_fullres(rec_a, rec_b)
            if overlap is not None:
                overlaps.append((rec_a.setup, rec_b.setup, overlap))

    return overlaps


def nominal_overlap_to_local_padded_crop(
    rec: TileRecord,
    overlap_box_fullres,
    pad_xy: int,
):
    ox0, ox1, oy0, oy1 = overlap_box_fullres

    tx = float(rec.nominal[0, 3])
    ty = float(rec.nominal[1, 3])

    lx0 = int(np.floor(ox0 - tx - pad_xy))
    lx1 = int(np.ceil(ox1 - tx + pad_xy))
    ly0 = int(np.floor(oy0 - ty - pad_xy))
    ly1 = int(np.ceil(oy1 - ty + pad_xy))

    lx0 = max(0, lx0)
    ly0 = max(0, ly0)
    lx1 = min(rec.size_x, lx1)
    ly1 = min(rec.size_y, ly1)

    if lx1 <= lx0 or ly1 <= ly0:
        return None

    return CropFullRes(x0=lx0, x1=lx1, y0=ly0, y1=ly1)


def fullres_crop_to_scaled(
    crop: CropFullRes,
    scale: ScaleInfo,
    vol_zyx: da.Array,
):
    _, y_max, x_max = vol_zyx.shape

    sx0 = int(np.floor(crop.x0 / scale.scale_x))
    sx1 = int(np.ceil(crop.x1 / scale.scale_x))
    sy0 = int(np.floor(crop.y0 / scale.scale_y))
    sy1 = int(np.ceil(crop.y1 / scale.scale_y))

    sx0 = max(0, sx0)
    sy0 = max(0, sy0)
    sx1 = min(x_max, sx1)
    sy1 = min(y_max, sy1)

    if sx1 <= sx0 or sy1 <= sy0:
        return None

    return CropScaled(x0=sx0, x1=sx1, y0=sy0, y1=sy1)


def filter_points_in_xy_crop(pts_xyz: np.ndarray, crop: CropFullRes):
    if pts_xyz is None or len(pts_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    mask = (
        (pts_xyz[:, 0] >= crop.x0) &
        (pts_xyz[:, 0] < crop.x1) &
        (pts_xyz[:, 1] >= crop.y0) &
        (pts_xyz[:, 1] < crop.y1)
    )

    return pts_xyz[mask]


def points_to_global_scaled_coords(
    pts_full_xyz: np.ndarray,
    rec: TileRecord,
    scale: ScaleInfo,
):
    """
    Local tile full-res XYZ -> nominal/global scaled XYZ.
    """
    if pts_full_xyz is None or len(pts_full_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    tx = float(rec.nominal[0, 3])
    ty = float(rec.nominal[1, 3])
    tz = float(rec.nominal[2, 3])

    out = np.empty_like(pts_full_xyz, dtype=np.float32)
    out[:, 0] = (pts_full_xyz[:, 0] + tx) / scale.scale_x
    out[:, 1] = (pts_full_xyz[:, 1] + ty) / scale.scale_y
    out[:, 2] = (pts_full_xyz[:, 2] + tz) / scale.scale_z

    return out


def global_points_to_a_crop_display(
    pts_global_xyz: np.ndarray,
    rec_a: TileRecord,
    crop_a_full: CropFullRes,
    scale_a: ScaleInfo,
    image_a_crop_shape,
):
    """
    Convert global-scaled XYZ points into tile-A-crop display coordinates.

    This keeps the image in A's local crop coordinate frame.
    Tile B points are transformed into this same frame.
    """
    if pts_global_xyz is None or len(pts_global_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    tx_a = float(rec_a.nominal[0, 3])
    ty_a = float(rec_a.nominal[1, 3])
    tz_a = float(rec_a.nominal[2, 3])

    origin_x = (tx_a + crop_a_full.x0) / scale_a.scale_x
    origin_y = (ty_a + crop_a_full.y0) / scale_a.scale_y
    origin_z = tz_a / scale_a.scale_z

    pts = pts_global_xyz.copy()
    pts[:, 0] -= origin_x
    pts[:, 1] -= origin_y
    pts[:, 2] -= origin_z

    Z, Y, X = image_a_crop_shape

    mask = (
        (pts[:, 0] >= 0) & (pts[:, 0] < X) &
        (pts[:, 1] >= 0) & (pts[:, 1] < Y) &
        (pts[:, 2] >= 0) & (pts[:, 2] < Z)
    )

    return pts[mask].astype(np.float32)


# ----------------------------
# Display helpers
# ----------------------------

def normalize_slice(x: np.ndarray, lo: float, hi: float):
    x = x.astype(np.float32, copy=False)

    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float32)

    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo)


def compute_global_display_range(vol_zyx: np.ndarray):
    x = vol_zyx.astype(np.float32, copy=False)
    mask = np.isfinite(x) & (x > 0)

    if not np.any(mask):
        return 0.0, 1.0

    lo = float(np.percentile(x[mask], P_LO))
    hi = float(np.percentile(x[mask], P_HI))

    if hi <= lo:
        lo = float(np.min(x[mask]))
        hi = float(np.max(x[mask]))

    if hi <= lo:
        hi = lo + 1.0

    return lo, hi


def normalize_for_display(slice2d: np.ndarray, global_lo: float, global_hi: float, mode: str):
    if mode == "global":
        return normalize_slice(slice2d, global_lo, global_hi)

    if mode == "per_slice":
        x = slice2d.astype(np.float32, copy=False)
        mask = np.isfinite(x) & (x > 0)

        if not np.any(mask):
            return np.zeros_like(x, dtype=np.float32)

        lo = float(np.percentile(x[mask], P_LO))
        hi = float(np.percentile(x[mask], P_HI))

        if hi <= lo:
            lo = float(np.min(x[mask]))
            hi = float(np.max(x[mask]))

        if hi <= lo:
            hi = lo + 1.0

        return normalize_slice(x, lo, hi)

    raise ValueError("DISPLAY_NORM_MODE must be 'global' or 'per_slice'")


def get_visible_points_for_slice(
    pts_xyz: np.ndarray,
    z: int,
    z_radius: float,
):
    if len(pts_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    dz = np.abs(pts_xyz[:, 2] - float(z))
    mask = dz <= z_radius

    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)

    return pts_xyz[mask]


# ----------------------------
# Match-readiness metrics
# ----------------------------

def safe_div(n, d):
    return float(n) / float(d) if d else np.nan


def estimate_dominant_translation(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    search_radius_scaled: float = MATCH_SEARCH_RADIUS_SCALED,
    bin_size_scaled: float = TRANSLATION_BIN_SCALED,
):
    """
    Estimate dominant A->B residual translation by local neighbor voting.

    For every A point, collect B points within a broad radius.
    Vote on delta = B - A in coarse 3D bins.
    The strongest bin gives likely residual shift.
    """
    if cKDTree is None or len(pts_a) == 0 or len(pts_b) == 0:
        return None

    tree_b = cKDTree(pts_b.astype(np.float32))

    deltas = []
    for pa in pts_a.astype(np.float32):
        idxs = tree_b.query_ball_point(pa, r=float(search_radius_scaled))
        if not idxs:
            continue

        pb = pts_b[np.asarray(idxs, dtype=np.int64)]
        d = pb - pa[None, :]
        deltas.append(d)

    if not deltas:
        return None

    deltas = np.vstack(deltas).astype(np.float32)

    bins = np.floor(deltas / float(bin_size_scaled)).astype(np.int32)

    unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
    best_i = int(np.argmax(counts))
    best_bin = unique_bins[best_i]
    best_count = int(counts[best_i])

    # Include immediate neighboring bins for support/refinement.
    bin_dist = np.max(np.abs(bins - best_bin[None, :]), axis=1)
    support_mask = bin_dist <= 1

    support_deltas = deltas[support_mask]
    shift = np.median(support_deltas, axis=0).astype(np.float32)

    support_pct_min = 100.0 * safe_div(len(support_deltas), min(len(pts_a), len(pts_b)))

    return {
        "shift_xyz": shift,
        "support_pairs": int(len(support_deltas)),
        "support_pct_min_cloud": float(support_pct_min),
        "raw_delta_count": int(len(deltas)),
        "best_bin_count": best_count,
    }


def nearest_scores_after_shift(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    shift_xyz: np.ndarray,
    radii=MATCH_RADII_SCALED,
):
    """
    Apply dominant shift to A points, then score nearest B points.
    shift convention: A_shifted = A + shift.
    """
    if cKDTree is None or len(pts_a) == 0 or len(pts_b) == 0:
        return {}

    a_shifted = pts_a + shift_xyz[None, :]

    tree_b = cKDTree(pts_b.astype(np.float32))
    dist_ab, idx_ab = tree_b.query(a_shifted.astype(np.float32), k=1)

    tree_a = cKDTree(a_shifted.astype(np.float32))
    dist_ba, idx_ba = tree_a.query(pts_b.astype(np.float32), k=1)

    out = {
        "median_ab": float(np.median(dist_ab)) if len(dist_ab) else np.nan,
        "median_ba": float(np.median(dist_ba)) if len(dist_ba) else np.nan,
        "median_sym": float(np.median(np.concatenate([dist_ab, dist_ba]))),
    }

    for r in radii:
        r_key = f"{r:g}"
        out[f"ab_within_{r_key}"] = 100.0 * float(np.mean(dist_ab <= r))
        out[f"ba_within_{r_key}"] = 100.0 * float(np.mean(dist_ba <= r))
        out[f"sym_within_{r_key}"] = 0.5 * (
            out[f"ab_within_{r_key}"] + out[f"ba_within_{r_key}"]
        )

    # Mutual nearest-neighbor pairs after shift.
    mutual_pairs = []
    for ia, ib in enumerate(idx_ab):
        if ib < len(idx_ba) and idx_ba[ib] == ia:
            mutual_pairs.append((ia, ib, float(dist_ab[ia])))

    out["mutual_pairs"] = mutual_pairs
    out["n_mutual"] = len(mutual_pairs)

    # Ambiguity: nearest / second-nearest distance.
    # Lower is more distinctive. Near 1.0 means ambiguous.
    if len(pts_b) >= 2:
        dist2, _ = tree_b.query(a_shifted.astype(np.float32), k=2)
        d1 = dist2[:, 0]
        d2 = dist2[:, 1]
        valid = np.isfinite(d1) & np.isfinite(d2) & (d2 > 1e-6)

        if np.any(valid):
            out["ambiguity_ratio_median"] = float(np.median(d1[valid] / d2[valid]))
        else:
            out["ambiguity_ratio_median"] = np.nan
    else:
        out["ambiguity_ratio_median"] = np.nan

    return out


def fit_affine_lstsq(src: np.ndarray, dst: np.ndarray):
    """
    Fit affine transform dst ~= src @ A.T + t.
    Returns 4x4 matrix.
    """
    n = len(src)
    if n < 4:
        return None

    X = np.ones((n, 4), dtype=np.float64)
    X[:, :3] = src.astype(np.float64)

    Y = dst.astype(np.float64)

    # Solve X @ M.T = Y, where M is 3x4.
    M_t, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    M = M_t.T

    aff = np.eye(4, dtype=np.float64)
    aff[:3, :4] = M
    return aff


def apply_affine_to_points(pts: np.ndarray, aff: np.ndarray):
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32)

    X = np.ones((len(pts), 4), dtype=np.float64)
    X[:, :3] = pts.astype(np.float64)
    Y = X @ aff[:3, :4].T
    return Y.astype(np.float32)


def affine_proxy_score(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    mutual_pairs,
    inlier_radius_scaled: float = AFFINE_INLIER_RADIUS_SCALED,
):
    """
    Fit a simple affine using mutual nearest pairs discovered after dominant translation.
    This is not the final matcher, but gives a RANSAC-readiness proxy.
    """
    if len(mutual_pairs) < MIN_AFFINE_PAIRS:
        return {
            "n_pairs": len(mutual_pairs),
            "n_inliers": 0,
            "inlier_ratio": np.nan,
            "residual_median": np.nan,
            "residual_p90": np.nan,
            "affine": None,
            "inlier_src": np.empty((0, 3), dtype=np.float32),
        }

    ia = np.asarray([p[0] for p in mutual_pairs], dtype=np.int64)
    ib = np.asarray([p[1] for p in mutual_pairs], dtype=np.int64)

    src = pts_a[ia]
    dst = pts_b[ib]

    aff = fit_affine_lstsq(src, dst)
    if aff is None:
        return {
            "n_pairs": len(mutual_pairs),
            "n_inliers": 0,
            "inlier_ratio": np.nan,
            "residual_median": np.nan,
            "residual_p90": np.nan,
            "affine": None,
            "inlier_src": np.empty((0, 3), dtype=np.float32),
        }

    pred = apply_affine_to_points(src, aff)
    residuals = np.sqrt(np.sum((pred - dst) ** 2, axis=1))

    inlier_mask = residuals <= float(inlier_radius_scaled)
    n_in = int(np.count_nonzero(inlier_mask))

    return {
        "n_pairs": int(len(mutual_pairs)),
        "n_inliers": n_in,
        "inlier_ratio": 100.0 * safe_div(n_in, len(mutual_pairs)),
        "residual_median": float(np.median(residuals)) if len(residuals) else np.nan,
        "residual_p90": float(np.percentile(residuals, 90)) if len(residuals) else np.nan,
        "affine": aff,
        "inlier_src": src[inlier_mask].astype(np.float32),
    }


def grid_occupancy_points(pts_xyz: np.ndarray, shape_zyx, bins=INLIER_GRID_BINS):
    """
    Spatial coverage of inlier points.
    Affine/RANSAC needs points spread out, not all in one small cluster.
    """
    if len(pts_xyz) == 0:
        return np.nan

    Z, Y, X = shape_zyx

    pts_zyx = np.stack(
        [pts_xyz[:, 2], pts_xyz[:, 1], pts_xyz[:, 0]],
        axis=1,
    )

    hist, _ = np.histogramdd(
        pts_zyx,
        bins=bins,
        range=((0, Z), (0, Y), (0, X)),
    )

    flat = hist.ravel()
    return safe_div(np.count_nonzero(flat), np.prod(bins))


def print_match_readiness_qc(
    pts_a_display: np.ndarray,
    pts_b_display: np.ndarray,
    image_shape_zyx,
    scale: ScaleInfo,
):
    print("\n" + "=" * 72)
    print("MATCH-READINESS QC")
    print("=" * 72)

    n_a = len(pts_a_display)
    n_b = len(pts_b_display)

    print(f"Points setup A:             {n_a:,}")
    print(f"Points setup B:             {n_b:,}")

    if n_a == 0 or n_b == 0:
        print("VERDICT: CHECK")
        print("Reason: one side has no points in the overlap display crop.")
        print("=" * 72 + "\n")
        return

    if cKDTree is None:
        print("VERDICT: CHECK")
        print("Reason: scipy.spatial.cKDTree is not available.")
        print("=" * 72 + "\n")
        return

    count_balance = min(n_a, n_b) / max(n_a, n_b)
    print(f"Count balance min/max:      {count_balance:.3f}")

    trans = estimate_dominant_translation(pts_a_display, pts_b_display)

    if trans is None:
        print("\nDominant translation:       not found")
        print("VERDICT: CHECK")
        print("Reason: no red/blue point pairs within broad search radius.")
        print("=" * 72 + "\n")
        return

    shift = trans["shift_xyz"]

    print("\nDominant residual translation A→B:")
    print(
        f"  shift scaled xyz:         "
        f"dx={shift[0]:.2f}, dy={shift[1]:.2f}, dz={shift[2]:.2f}"
    )
    print(
        f"  shift full-res xyz:       "
        f"dx={shift[0] * scale.scale_x:.1f}, "
        f"dy={shift[1] * scale.scale_y:.1f}, "
        f"dz={shift[2] * scale.scale_z:.1f}"
    )
    print(f"  vote support pairs:       {trans['support_pairs']:,}")
    print(f"  support / min cloud:      {trans['support_pct_min_cloud']:.2f}%")
    print(f"  raw local pair deltas:    {trans['raw_delta_count']:,}")
    print(f"  best translation bin:     {trans['best_bin_count']:,}")

    nn = nearest_scores_after_shift(pts_a_display, pts_b_display, shift)

    if not nn:
        print("\nNearest-neighbor after shift: unavailable")
        print("VERDICT: CHECK")
        print("=" * 72 + "\n")
        return

    print("\nNearest red/blue after applying dominant shift:")
    print(f"  median A→B dist:          {nn['median_ab']:.2f} scaled px")
    print(f"  median B→A dist:          {nn['median_ba']:.2f} scaled px")
    print(f"  median symmetric dist:    {nn['median_sym']:.2f} scaled px")

    for r in MATCH_RADII_SCALED:
        r_key = f"{r:g}"
        print(
            f"  symmetric within {r_key}px:   "
            f"{nn[f'sym_within_{r_key}']:.2f}% "
            f"(A→B {nn[f'ab_within_{r_key}']:.2f}%, "
            f"B→A {nn[f'ba_within_{r_key}']:.2f}%)"
        )

    print(f"  mutual nearest pairs:     {nn['n_mutual']:,}")
    print(f"  ambiguity median d1/d2:   {nn['ambiguity_ratio_median']:.3f}")

    aff = affine_proxy_score(
        pts_a_display,
        pts_b_display,
        nn["mutual_pairs"],
    )

    inlier_grid = grid_occupancy_points(
        aff["inlier_src"],
        image_shape_zyx,
        bins=INLIER_GRID_BINS,
    )

    print("\nAffine/RANSAC-readiness proxy:")
    print(f"  candidate mutual pairs:   {aff['n_pairs']:,}")
    print(f"  affine inliers <= {AFFINE_INLIER_RADIUS_SCALED:g}px: {aff['n_inliers']:,}")
    print(f"  affine inlier ratio:      {aff['inlier_ratio']:.2f}%")
    print(f"  affine residual median:   {aff['residual_median']:.2f} scaled px")
    print(f"  affine residual p90:      {aff['residual_p90']:.2f} scaled px")
    print(f"  inlier grid occupancy:    {inlier_grid:.3f}")

    good_translation = trans["support_pct_min_cloud"] >= 35.0
    good_within6 = nn["sym_within_6"] >= 45.0
    good_affine = (
        aff["n_inliers"] >= 20 and
        np.isfinite(aff["inlier_ratio"]) and
        aff["inlier_ratio"] >= 50.0
    )
    good_spread = np.isfinite(inlier_grid) and inlier_grid >= 0.20
    not_too_ambiguous = (
        np.isfinite(nn["ambiguity_ratio_median"]) and
        nn["ambiguity_ratio_median"] <= 0.80
    )

    if all([good_translation, good_within6, good_affine, good_spread]):
        verdict = "GOOD"
    elif good_translation and (good_within6 or good_affine):
        verdict = "USABLE / CHECK MATCHING"
    else:
        verdict = "CHECK"

    print("")
    print(f"VERDICT: {verdict}")

    if verdict != "GOOD":
        print("Reason flags:")
        if not good_translation:
            print("  - weak dominant translation support")
        if not good_within6:
            print("  - low symmetric red/blue proximity within 6 scaled px after shift")
        if not good_affine:
            print("  - weak affine/RANSAC proxy inlier set")
        if not good_spread:
            print("  - inlier candidates are not spatially spread out")
        if not not_too_ambiguous:
            print("  - candidate matches may be ambiguous/repetitive")

    print("=" * 72 + "\n")


# ----------------------------
# Main data prep
# ----------------------------

def prepare_pair_data(rec: TileRecord, overlap_box_fullres):
    print(f"Opening setup {rec.setup} image: {rec.full_path}")

    vol = open_tile_volume_zyx(rec)
    scale = infer_scale_info(rec, vol)

    print(
        f"  setup {rec.setup} scaled image shape zyx={vol.shape}, "
        f"scale xyz=({scale.scale_x:.3f}, {scale.scale_y:.3f}, {scale.scale_z:.3f})"
    )

    crop_full = nominal_overlap_to_local_padded_crop(
        rec,
        overlap_box_fullres,
        OVERLAP_PAD_XY,
    )

    if crop_full is None:
        raise RuntimeError(f"Could not make overlap crop for setup {rec.setup}")

    crop_scaled = fullres_crop_to_scaled(crop_full, scale, vol)

    if crop_scaled is None:
        raise RuntimeError(f"Could not convert crop to scaled coords for setup {rec.setup}")

    vol_crop = vol[
        :,
        crop_scaled.y0:crop_scaled.y1,
        crop_scaled.x0:crop_scaled.x1,
    ].compute().astype(np.float32)

    print(
        f"  setup {rec.setup} crop full x[{crop_full.x0}:{crop_full.x1}] "
        f"y[{crop_full.y0}:{crop_full.y1}]"
    )
    print(f"  setup {rec.setup} crop scaled zyx={vol_crop.shape}")

    print(f"Loading setup {rec.setup} points")
    pts_all = read_interest_points(INTERESTPOINTS_BASE, rec.setup, TIMEPOINT)
    print(f"  setup {rec.setup} total points: {len(pts_all):,}")

    pts_xy = filter_points_in_xy_crop(pts_all, crop_full)
    print(f"  setup {rec.setup} points in padded XY overlap crop: {len(pts_xy):,}")

    pts_global_scaled = points_to_global_scaled_coords(pts_xy, rec, scale)

    return {
        "vol_crop": vol_crop,
        "scale": scale,
        "crop_full": crop_full,
        "crop_scaled": crop_scaled,
        "pts_global_scaled": pts_global_scaled,
    }


# ----------------------------
# Same-space point-only viewer
# ----------------------------

def view_points_on_a_crop(
    image_a_crop: np.ndarray,
    pts_a_display: np.ndarray,
    pts_b_display: np.ndarray,
    setup_a: int,
    setup_b: int,
    title: str = "",
    point_z_radius_scaled: float = 1.5,
):
    """
    Tile A image is shown in its own local crop frame.
    Tile A points are red.
    Tile B points are transformed into A's local crop frame and shown blue.
    No match links.
    """

    assert image_a_crop.ndim == 3

    Z, Y, X = image_a_crop.shape
    lo, hi = compute_global_display_range(image_a_crop)

    def norm(z):
        return normalize_for_display(image_a_crop[z], lo, hi, DISPLAY_NORM_MODE)

    z0 = 0

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.suptitle(title or "A-crop point overlay viewer", fontsize=14)

    im = ax.imshow(
        norm(z0),
        cmap="gray",
        interpolation="nearest",
        aspect=IMAGE_ASPECT,
    )
    ax.axis("off")

    pts_a0 = get_visible_points_for_slice(pts_a_display, z0, point_z_radius_scaled)
    pts_b0 = get_visible_points_for_slice(pts_b_display, z0, point_z_radius_scaled)

    sc_a = ax.scatter(
        pts_a0[:, 0] if len(pts_a0) else [],
        pts_a0[:, 1] if len(pts_a0) else [],
        s=POINT_A_SIZE,
        c=POINT_A_COLOR,
        alpha=POINT_ALPHA,
        edgecolors=POINT_EDGE_COLOR,
        linewidths=POINT_LINEWIDTH,
        label=f"setup {setup_a}",
    )

    sc_b = ax.scatter(
        pts_b0[:, 0] if len(pts_b0) else [],
        pts_b0[:, 1] if len(pts_b0) else [],
        s=POINT_B_SIZE,
        c=POINT_B_COLOR,
        alpha=POINT_ALPHA,
        edgecolors=POINT_EDGE_COLOR,
        linewidths=POINT_LINEWIDTH,
        label=f"setup {setup_b}",
    )

    ax.legend(loc="upper right")

    plt.subplots_adjust(bottom=0.16)
    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.04])
    slider = Slider(slider_ax, "Z", 0, Z - 1, valinit=z0, valstep=1)

    def update(val):
        z = int(slider.val)

        im.set_data(norm(z))

        pts_a_now = get_visible_points_for_slice(
            pts_a_display,
            z,
            point_z_radius_scaled,
        )

        pts_b_now = get_visible_points_for_slice(
            pts_b_display,
            z,
            point_z_radius_scaled,
        )

        if len(pts_a_now):
            sc_a.set_offsets(pts_a_now[:, :2])
        else:
            sc_a.set_offsets(np.empty((0, 2)))

        if len(pts_b_now):
            sc_b.set_offsets(pts_b_now[:, :2])
        else:
            sc_b.set_offsets(np.empty((0, 2)))

        ax.set_title(
            f"z={z} | "
            f"red setup {setup_a}: {len(pts_a_now):,} | "
            f"blue setup {setup_b}: {len(pts_b_now):,}",
            fontsize=12,
        )

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(z0)
    plt.show()


# ----------------------------
# Main
# ----------------------------

def main():
    print("Loading XML...")
    root = load_xml_root(XML_PATH)

    records = parse_zarr_tile_records(root, TRANSFORM_NAME)
    print(f"Found tile records: {len(records)}")

    if len(records) == 0:
        print("No tile records found.")
        return

    by_setup = {r.setup: r for r in records}
    overlaps = list_overlapping_pairs(records)

    if len(overlaps) == 0:
        print("No overlapping pairs found.")
        return

    print("\nOverlapping pairs:")
    for a, b, (ox0, ox1, oy0, oy1) in overlaps:
        print(
            f"  ({a}, {b}) overlap full-res "
            f"x[{ox0:.1f}:{ox1:.1f}] y[{oy0:.1f}:{oy1:.1f}]"
        )

    if TARGET_PAIR is None:
        setup_a, setup_b, overlap_box = overlaps[0]
        print(f"\nTARGET_PAIR is None, using first overlap: ({setup_a}, {setup_b})")
    else:
        a, b = TARGET_PAIR

        if a not in by_setup or b not in by_setup:
            raise RuntimeError(f"TARGET_PAIR {TARGET_PAIR} not found in XML records")

        overlap_box = compute_pairwise_overlap_fullres(by_setup[a], by_setup[b])

        if overlap_box is None:
            raise RuntimeError(f"TARGET_PAIR {TARGET_PAIR} does not overlap in XY")

        setup_a, setup_b = a, b
        print(f"\nUsing TARGET_PAIR: ({setup_a}, {setup_b})")

    rec_a = by_setup[setup_a]
    rec_b = by_setup[setup_b]

    data_a = prepare_pair_data(rec_a, overlap_box)
    data_b = prepare_pair_data(rec_b, overlap_box)

    pts_a_display = global_points_to_a_crop_display(
        pts_global_xyz=data_a["pts_global_scaled"],
        rec_a=rec_a,
        crop_a_full=data_a["crop_full"],
        scale_a=data_a["scale"],
        image_a_crop_shape=data_a["vol_crop"].shape,
    )

    pts_b_display = global_points_to_a_crop_display(
        pts_global_xyz=data_b["pts_global_scaled"],
        rec_a=rec_a,
        crop_a_full=data_a["crop_full"],
        scale_a=data_a["scale"],
        image_a_crop_shape=data_a["vol_crop"].shape,
    )

    print("")
    print("A-crop display point counts:")
    print(f"  setup {setup_a} red points:  {len(pts_a_display):,}")
    print(f"  setup {setup_b} blue points: {len(pts_b_display):,}")
    print("")

    print_match_readiness_qc(
        pts_a_display=pts_a_display,
        pts_b_display=pts_b_display,
        image_shape_zyx=data_a["vol_crop"].shape,
        scale=data_a["scale"],
    )

    title = (
        f"IP overlay in setup {setup_a} crop frame | "
        f"red setup {setup_a} vs blue setup {setup_b} | "
        f"scale level {SCALE_LEVEL} | "
        f"±{POINT_Z_RADIUS_SCALED} Z slices"
    )

    view_points_on_a_crop(
        image_a_crop=data_a["vol_crop"],
        pts_a_display=pts_a_display,
        pts_b_display=pts_b_display,
        setup_a=setup_a,
        setup_b=setup_b,
        title=title,
        point_z_radius_scaled=POINT_Z_RADIUS_SCALED,
    )


if __name__ == "__main__":
    main()