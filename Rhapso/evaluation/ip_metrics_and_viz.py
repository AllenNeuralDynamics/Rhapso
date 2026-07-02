import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import boto3
import dask.array as da
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
from botocore import UNSIGNED
from botocore.config import Config
from matplotlib.widgets import Slider
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

XML_PATH = "/Users/sean.fite/Desktop/exaSPIM_791116_2tiles.xml"
ALIGNMENT_STORE_BASE = "/Users/sean.fite/Desktop/exaSPIM-test"
POINT_LABEL = "beads"

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
# Texture / detector-quality QC params
# ----------------------------

TEXTURE_SIGMAS_SCALED = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
TEXTURE_PEAK_Z = 4.0
TEXTURE_GRID_BINS_XY = (8, 8)


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
# New Parquet/JSON point-store reader
# ----------------------------

class ParquetPointStore:
    def __init__(self, base_path: str, label: str = "beads", storage_options=None):
        self.base_path = str(base_path).rstrip("/")
        self.label = label
        self.storage_options = storage_options or {}

        self.manifest = {}
        self.point_index_df = None

    def join_uri(self, *parts):
        cleaned = []

        for i, part in enumerate(parts):
            if part is None:
                continue

            part = str(part)
            cleaned.append(part.rstrip("/") if i == 0 else part.strip("/"))

        return "/".join(cleaned)

    def get_fs_and_path(self, uri):
        return fsspec.core.url_to_fs(uri, **self.storage_options)

    def exists(self, uri):
        try:
            fs, path = self.get_fs_and_path(uri)
            return fs.exists(path)
        except Exception:
            return False

    def read_json(self, uri):
        fs, path = self.get_fs_and_path(uri)

        with fs.open(path, "r") as f:
            return json.load(f)

    def read_parquet(self, uri):
        fs, path = self.get_fs_and_path(uri)

        with fs.open(path, "rb") as f:
            return pd.read_parquet(f, engine="pyarrow")

    def default_point_relative_path(self, timepoint: int, setup: int, label: str):
        return (
            f"points/"
            f"timepoint={int(timepoint)}/"
            f"setup={int(setup)}/"
            f"label={label}/"
            f"points.parquet"
        )

    def view_label_key(self, timepoint: int, setup: int, label: str):
        return f"{int(timepoint)}/{int(setup)}/{label}"

    def load_manifest(self):
        if self.manifest:
            return

        manifest_uri = self.join_uri(self.base_path, "manifest.json")

        if self.exists(manifest_uri):
            self.manifest = self.read_json(manifest_uri)
            print(f"Loaded point manifest: {manifest_uri}")
        else:
            self.manifest = {}
            print(f"⚠️ No point manifest found at: {manifest_uri}")

    def load_point_index(self):
        if self.point_index_df is not None:
            return

        index_uri = self.join_uri(self.base_path, "point_index.parquet")

        if self.exists(index_uri):
            df = self.read_parquet(index_uri)
            print(f"Loaded point index: {index_uri}")
        else:
            print(f"⚠️ No point_index.parquet found at: {index_uri}")
            df = pd.DataFrame(
                columns=["timepoint", "setup", "label", "path", "num_points"]
            )

        if len(df) > 0:
            if "view_setup" in df.columns and "setup" not in df.columns:
                df = df.rename(columns={"view_setup": "setup"})

            df["timepoint"] = df["timepoint"].astype("int32")
            df["setup"] = df["setup"].astype("int32")
            df["label"] = df["label"].astype(str)

            if "path" not in df.columns:
                df["path"] = [
                    self.default_point_relative_path(row.timepoint, row.setup, row.label)
                    for row in df.itertuples(index=False)
                ]

        self.point_index_df = df

    def resolve_point_relative_path(self, timepoint: int, setup: int, label: str):
        self.load_manifest()
        self.load_point_index()

        key = self.view_label_key(timepoint, setup, label)
        manifest_points = self.manifest.get("points", {}) or {}

        if key in manifest_points:
            return manifest_points[key]

        if self.point_index_df is not None and len(self.point_index_df) > 0:
            rows = self.point_index_df[
                (self.point_index_df["timepoint"].astype(int) == int(timepoint))
                & (self.point_index_df["setup"].astype(int) == int(setup))
                & (self.point_index_df["label"].astype(str) == str(label))
            ]

            if len(rows) > 0 and "path" in rows.columns:
                return str(rows.iloc[0]["path"])

        return self.default_point_relative_path(timepoint, setup, label)

    def read_points(self, setup: int, timepoint: int = 0, label: str = None):
        label = label or self.label

        rel_path = self.resolve_point_relative_path(
            timepoint=timepoint,
            setup=setup,
            label=label,
        )

        point_uri = self.join_uri(self.base_path, rel_path)

        if not self.exists(point_uri):
            raise FileNotFoundError(
                f"Missing points parquet for setup={setup}, "
                f"timepoint={timepoint}, label={label}: {point_uri}"
            )

        df = self.read_parquet(point_uri)

        if len(df) == 0:
            return np.empty((0, 3), dtype=np.float32)

        missing = {"x", "y", "z"}.difference(df.columns)

        if missing:
            raise ValueError(
                f"Missing required point columns {sorted(missing)} in {point_uri}"
            )

        return df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)


# ----------------------------
# XML / IO helpers
# ----------------------------

def load_xml_root(xml_path: str) -> ET.Element:
    if xml_path.startswith("s3://"):
        parsed = urlparse(xml_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        if S3_ANON:
            client = boto3.client(
                "s3",
                config=Config(signature_version=UNSIGNED),
            )
        else:
            client = boto3.client("s3")

        obj = client.get_object(Bucket=bucket, Key=key)
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
    """
    Open one OME-Zarr scale level.
    """
    zarr_path = str(zarr_path).rstrip("/")
    scale_level = str(scale_level).strip("/")

    if zarr_path.startswith("s3://"):
        level_path = f"{zarr_path}/{scale_level}"

        s3 = s3fs.S3FileSystem(
            anon=S3_ANON,
            skip_instance_cache=True,
        )

        store = s3fs.S3Map(
            root=level_path,
            s3=s3,
            check=False,
        )

        return da.from_zarr(store)

    level_path = str(Path(zarr_path).expanduser() / scale_level)
    return da.from_zarr(level_path)


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
        (pts_xyz[:, 0] >= crop.x0)
        & (pts_xyz[:, 0] < crop.x1)
        & (pts_xyz[:, 1] >= crop.y0)
        & (pts_xyz[:, 1] < crop.y1)
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

    z_size, y_size, x_size = image_a_crop_shape

    mask = (
        (pts[:, 0] >= 0)
        & (pts[:, 0] < x_size)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < y_size)
        & (pts[:, 2] >= 0)
        & (pts[:, 2] < z_size)
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
    if len(pts_a) == 0 or len(pts_b) == 0:
        return None

    tree_b = cKDTree(pts_b.astype(np.float32))

    deltas = []

    for pa in pts_a.astype(np.float32):
        idxs = tree_b.query_ball_point(pa, r=float(search_radius_scaled))

        if not idxs:
            continue

        pb = pts_b[np.asarray(idxs, dtype=np.int64)]
        deltas.append(pb - pa[None, :])

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
    if len(pts_a) == 0 or len(pts_b) == 0:
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

    mutual_pairs = []

    for ia, ib in enumerate(idx_ab):
        if ib < len(idx_ba) and idx_ba[ib] == ia:
            mutual_pairs.append((ia, ib, float(dist_ab[ia])))

    out["mutual_pairs"] = mutual_pairs
    out["n_mutual"] = len(mutual_pairs)

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

    x = np.ones((n, 4), dtype=np.float64)
    x[:, :3] = src.astype(np.float64)

    y = dst.astype(np.float64)

    # Solve X @ M.T = Y, where M is 3x4.
    m_t, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    m = m_t.T

    aff = np.eye(4, dtype=np.float64)
    aff[:3, :4] = m

    return aff


def apply_affine_to_points(pts: np.ndarray, aff: np.ndarray):
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32)

    x = np.ones((len(pts), 4), dtype=np.float64)
    x[:, :3] = pts.astype(np.float64)
    y = x @ aff[:3, :4].T

    return y.astype(np.float32)


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

    z_size, y_size, x_size = shape_zyx

    pts_zyx = np.stack(
        [pts_xyz[:, 2], pts_xyz[:, 1], pts_xyz[:, 0]],
        axis=1,
    )

    hist, _ = np.histogramdd(
        pts_zyx,
        bins=bins,
        range=((0, z_size), (0, y_size), (0, x_size)),
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
        aff["n_inliers"] >= 20
        and np.isfinite(aff["inlier_ratio"])
        and aff["inlier_ratio"] >= 50.0
    )
    good_spread = np.isfinite(inlier_grid) and inlier_grid >= 0.20
    not_too_ambiguous = (
        np.isfinite(nn["ambiguity_ratio_median"])
        and nn["ambiguity_ratio_median"] <= 0.80
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
# Texture / detector-quality QC
# ----------------------------

def robust_normalize01(x: np.ndarray):
    x = x.astype(np.float32, copy=False)
    mask = np.isfinite(x) & (x > 0)

    if not np.any(mask):
        return np.zeros_like(x, dtype=np.float32)

    lo = float(np.percentile(x[mask], 1))
    hi = float(np.percentile(x[mask], 99.8))

    if hi <= lo:
        hi = lo + 1.0

    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def robust_zscore(x: np.ndarray):
    x = x.astype(np.float32, copy=False)

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-6

    return (x - med) / (1.4826 * mad)


def point_grid_occupancy_xy(pts_xyz: np.ndarray, shape_zyx, bins=TEXTURE_GRID_BINS_XY):
    if pts_xyz is None or len(pts_xyz) == 0:
        return np.nan

    _, y_size, x_size = shape_zyx
    pts_xy = pts_xyz[:, :2]

    hist, _ = np.histogramdd(
        pts_xy,
        bins=bins,
        range=((0, x_size), (0, y_size)),
    )

    return float(np.count_nonzero(hist)) / float(np.prod(bins))


def texture_metrics_from_crop(vol_zyx: np.ndarray):
    """
    Uses the crop max projection so this is fast and directly comparable
    to the point overlay viewer.
    """
    mip = np.max(vol_zyx, axis=0)
    img = robust_normalize01(mip)

    tissue_mask = img > 0.08
    tissue_fraction = float(np.mean(tissue_mask))

    if tissue_fraction <= 0:
        return {
            "classification": "blank / no tissue",
            "tissue_fraction": 0.0,
            "high_freq_ratio": np.nan,
            "best_sigma": np.nan,
            "best_peak_density_per_mpix": 0.0,
            "sigma_rows": [],
        }

    # High-frequency ratio: sharper / punctate crops tend to be higher.
    smooth = ndi.gaussian_filter(img, sigma=3.0)
    high = img - smooth
    high_freq_ratio = float(
        np.std(high[tissue_mask]) / (np.std(img[tissue_mask]) + 1e-6)
    )

    sigma_rows = []

    for sigma in TEXTURE_SIGMAS_SCALED:
        # Bright blob response. Smaller winning sigma means more punctate/cellular.
        response = -float(sigma ** 2) * ndi.gaussian_laplace(img, sigma=float(sigma))
        response = np.maximum(response, 0)

        rz = robust_zscore(response)

        local_max = response == ndi.maximum_filter(response, size=3)
        peak_mask = local_max & tissue_mask & (rz > TEXTURE_PEAK_Z)

        n_peaks = int(np.count_nonzero(peak_mask))
        area_mpix = float(img.size) / 1_000_000.0
        peak_density = n_peaks / area_mpix if area_mpix > 0 else np.nan

        sigma_rows.append(
            {
                "sigma": float(sigma),
                "n_peaks": n_peaks,
                "peak_density_per_mpix": float(peak_density),
                "p998_response": float(np.percentile(response[tissue_mask], 99.8)),
            }
        )

    # Pick the scale with strongest high-percentile LoG response.
    best = max(sigma_rows, key=lambda r: r["p998_response"])
    best_sigma = best["sigma"]
    best_peak_density = best["peak_density_per_mpix"]

    # Heuristic labels. These are meant as tuning guidance, not absolute truth.
    if best_sigma <= 2.0 and high_freq_ratio >= 0.25 and best_peak_density >= 100:
        classification = "punctate / cellular"
    elif best_sigma >= 4.0 and high_freq_ratio < 0.20 and best_peak_density < 100:
        classification = "broad / smooth"
    else:
        classification = "mixed / ambiguous"

    return {
        "classification": classification,
        "tissue_fraction": tissue_fraction,
        "high_freq_ratio": high_freq_ratio,
        "best_sigma": best_sigma,
        "best_peak_density_per_mpix": best_peak_density,
        "sigma_rows": sigma_rows,
    }


def print_texture_and_point_qc(
    label: str,
    vol_zyx: np.ndarray,
    pts_display_xyz: np.ndarray,
):
    print("\n" + "=" * 72)
    print(f"TEXTURE + DETECTOR QC: {label}")
    print("=" * 72)

    tex = texture_metrics_from_crop(vol_zyx)

    z_size, y_size, x_size = vol_zyx.shape

    n_pts = len(pts_display_xyz)
    area_mpix = float(y_size * x_size) / 1_000_000.0
    pts_per_mpix = n_pts / area_mpix if area_mpix > 0 else np.nan
    xy_occupancy = point_grid_occupancy_xy(pts_display_xyz, vol_zyx.shape)

    print(f"Image texture class:        {tex['classification']}")
    print(f"Signal/tissue fraction:     {tex['tissue_fraction']:.3f}")
    print(f"High-frequency ratio:       {tex['high_freq_ratio']:.3f}")
    print(f"Best LoG sigma scaled px:   {tex['best_sigma']}")
    print(f"LoG peaks / Mpix:           {tex['best_peak_density_per_mpix']:.1f}")

    print("")
    print(f"Detected Rhapso points:     {n_pts:,}")
    print(f"Rhapso points / Mpix:       {pts_per_mpix:.1f}")
    print(f"XY point grid occupancy:    {xy_occupancy:.3f}")

    print("")
    print("LoG response by scale:")

    for row in tex["sigma_rows"]:
        print(
            f"  sigma={row['sigma']:<4g} "
            f"peaks/Mpix={row['peak_density_per_mpix']:>8.1f} "
            f"p99.8 response={row['p998_response']:.5f}"
        )

    # Detector-readiness heuristic.
    enough_points = n_pts >= 50
    spread_ok = np.isfinite(xy_occupancy) and xy_occupancy >= 0.20
    not_blank = tex["tissue_fraction"] >= 0.05

    if not not_blank:
        verdict = "CHECK: mostly blank overlap"
    elif enough_points and spread_ok:
        verdict = "GOOD: detector has usable spatial coverage"
    elif enough_points and not spread_ok:
        verdict = "CHECK: points are clustered"
    else:
        verdict = "CHECK: too few detected points"

    print("")
    print(f"VERDICT: {verdict}")

    if tex["classification"] == "punctate / cellular":
        print("Param hint: test smaller/sharper detection, e.g. sigma 1.1–1.8.")
    elif tex["classification"] == "broad / smooth":
        print("Param hint: use smoother/permissive detection, and expect matching to be harder.")
    else:
        print("Param hint: mixed signal; compare sigma 1.1 vs 1.8 using overlays.")

    print("=" * 72 + "\n")


# ----------------------------
# Main data prep
# ----------------------------

def prepare_pair_data(
    rec: TileRecord,
    overlap_box_fullres,
    point_store: ParquetPointStore,
):
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

    print(f"Loading setup {rec.setup} points from Parquet")
    pts_all = point_store.read_points(
        setup=rec.setup,
        timepoint=TIMEPOINT,
        label=POINT_LABEL,
    )
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

    z_size, _, _ = image_a_crop.shape
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
    slider = Slider(slider_ax, "Z", 0, z_size - 1, valinit=z0, valstep=1)

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

    point_store_storage_options = {}

    if ALIGNMENT_STORE_BASE.startswith("s3://"):
        point_store_storage_options = {
            "anon": S3_ANON,
            "skip_instance_cache": True,
        }

    point_store = ParquetPointStore(
        base_path=ALIGNMENT_STORE_BASE,
        label=POINT_LABEL,
        storage_options=point_store_storage_options,
    )

    rec_a = by_setup[setup_a]
    rec_b = by_setup[setup_b]

    data_a = prepare_pair_data(rec_a, overlap_box, point_store)
    data_b = prepare_pair_data(rec_b, overlap_box, point_store)

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

    print_texture_and_point_qc(
        label=f"setup {setup_a} crop",
        vol_zyx=data_a["vol_crop"],
        pts_display_xyz=pts_a_display,
    )

    print_texture_and_point_qc(
        label=f"setup {setup_b} projected into setup {setup_a} crop frame",
        vol_zyx=data_b["vol_crop"],
        pts_display_xyz=pts_b_display,
    )

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