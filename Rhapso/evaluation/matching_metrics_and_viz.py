#!/usr/bin/env python3

import json
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
from urllib.parse import urlparse
from pathlib import Path

import boto3
import dask.array as da
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
from scipy.ndimage import map_coordinates


# ----------------------------
# Params
# ----------------------------

# Use detection xml for rigid match metrics - rigid xml for affine match metrics - etc.
XML_PATH = "/Users/sean.fite/Desktop/exaSPIM_791116_2tiles.xml"
ALIGNMENT_BASE = "/Users/sean.fite/Desktop/exaSPIM-test"

SCALE_LEVEL = "4"
TIMEPOINT = 0
MATCH_LABEL = "beads"

TARGET_PAIR: Optional[Tuple[int, int]] = None
# TARGET_PAIR = (5, 9)

ZOOM_Y_CHUNKS = 2
ZOOM_LINK_BY = "midpoint"  # "midpoint" or "either_endpoint"

OVERLAP_PAD_XY = 0
POINT_Z_RADIUS_SCALED = 1.5
LINK_Z_RADIUS_SCALED = 2.0

DISPLAY_NORM_MODE = "global"  # "global" or "per_slice"
P_LO = 1
P_HI = 99

# Display-only brightness controls. These do not change points, matches, QC, or input data.
# Lower P_HI / lower gamma / higher gain brightens dim tissue while allowing bright spots to clip.
DISPLAY_BRIGHTNESS_GAIN = 1.00
DISPLAY_GAMMA = 1.00
DISPLAY_BLACK_LEVEL = 0.00

IMAGE_ASPECT = "equal"

GOOD_RESIDUAL_PX = 2.0
OK_RESIDUAL_PX = 4.0
BAD_RESIDUAL_PX = 8.0
INLIER_RESIDUAL_PX = 4.0
GRID_BINS = (4, 8, 8)

# Tile overlay display
# A tile is shown as green. B tile is resampled into A display frame and shown as purple.
SHOW_TILE_B_PURPLE_OVERLAY = True
TILE_A_BASE_RGB = np.array([0.20, 1.00, 0.25], dtype=np.float32)
TILE_B_OVERLAY_RGB = np.array([0.78, 0.35, 0.95], dtype=np.float32)
TILE_B_OVERLAY_ALPHA = 0.35
TILE_B_MIN_VISIBLE = 0.03
WARP_Y_BLOCK = 512

# Sideways display tweaks.
# These are display-only rotations: image, match dots, links, and rings rotate together.
# They do not change warp math, match math, QC, or saved data.
FULL_VIEW_ROTATE_90 = True
FULL_VIEW_ROTATE_DIRECTION = "clockwise"  # "clockwise" or "counterclockwise"
FULL_VIEW_AREA_SCALE = 1.0  # keep contained; old 10x area made the sideways view huge/off-screen
FULL_VIEW_BASE_FIGSIZE = (12.0, 7.5)
FULL_VIEW_MAX_FIGSIZE = (13.5, 8.5)

SPLIT_VIEW_ROTATE_90 = True
SPLIT_VIEW_ROTATE_DIRECTION = "clockwise"  # "clockwise" or "counterclockwise"
SPLIT_VIEW_FIGSIZE = (12.0, 8.5)
SPLIT_VIEW_LINK_SHOW_HALO = False
SPLIT_VIEW_LINK_LINEWIDTH = 1.1
SPLIT_VIEW_LINK_ALPHA = 0.75
SPLIT_VIEW_LINK_LINESTYLE = "dashed"


# ----------------------------
# Generic IO helpers
# ----------------------------

def join_uri(*parts):
    return "/".join(
        str(part).strip("/") if i > 0 else str(part).rstrip("/")
        for i, part in enumerate(parts)
        if part is not None
    )


def read_json(uri):
    fs, path = fsspec.core.url_to_fs(uri)
    with fs.open(path, "r") as f:
        return json.load(f)


def read_parquet(uri):
    fs, path = fsspec.core.url_to_fs(uri)
    with fs.open(path, "rb") as f:
        return pd.read_parquet(f, engine="pyarrow")


def load_xml_root(xml_path: str) -> ET.Element:
    if xml_path.startswith("s3://"):
        parsed = urlparse(xml_path)
        obj = boto3.client("s3").get_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
        )
        return ET.fromstring(obj["Body"].read())

    return ET.parse(xml_path).getroot()


# ----------------------------
# XML / transform helpers
# ----------------------------

def affine_12_to_4x4(affine_text: str) -> np.ndarray:
    vals = [float(v) for v in str(affine_text).replace(",", " ").split()]
    if len(vals) != 12:
        raise RuntimeError(f"Expected 12 affine values, got {len(vals)}")

    mat = np.eye(4, dtype=np.float64)
    mat[0, 0:4] = vals[0:4]
    mat[1, 0:4] = vals[4:8]
    mat[2, 0:4] = vals[8:12]
    return mat


def apply_affine_xyz(pts_xyz: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if pts_xyz is None or len(pts_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    ones = np.ones((len(pts_xyz), 1), dtype=np.float64)
    homo = np.concatenate([pts_xyz.astype(np.float64), ones], axis=1)
    out = homo @ mat.T
    return out[:, :3].astype(np.float32)


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


def parse_all_transforms_from_xml(root: ET.Element):
    transforms = {}

    for vr in root.findall(".//ViewRegistration"):
        setup = int(vr.get("setup"))
        tp = int(vr.get("timepoint", 0))

        if tp != TIMEPOINT:
            continue

        composed = np.eye(4, dtype=np.float64)
        names = []

        for vt in vr.findall("ViewTransform"):
            name = vt.findtext("Name") or ""
            affine_text = vt.findtext("affine")

            if not affine_text:
                continue

            mat = affine_12_to_4x4(affine_text)
            composed = mat @ composed
            names.append(name)

        if not names:
            raise RuntimeError(f"No ViewTransform matrices found for setup {setup}, timepoint {tp}")

        transforms[setup] = (composed, names)

    return transforms


def parse_zarr_tile_records(root: ET.Element):
    setup_sizes = parse_view_setup_sizes(root)
    all_transforms = parse_all_transforms_from_xml(root)

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

        if tp != TIMEPOINT:
            continue

        if setup not in setup_sizes:
            raise RuntimeError(f"Missing ViewSetup size for setup {setup}")

        if setup not in all_transforms:
            raise RuntimeError(f"Missing transform stack for setup {setup}")

        sx, sy, sz = setup_sizes[setup]
        transform, transform_names = all_transforms[setup]

        print(f"Setup {setup} transform stack: {transform_names}")

        records.append({
            "setup": setup,
            "tp": tp,
            "rel_path": rel_path,
            "full_path": zarr_base + rel_path,
            "size_x": sx,
            "size_y": sy,
            "size_z": sz,
            "transform": transform,
            "transform_names": transform_names,
        })

    return sorted(records, key=lambda r: r["setup"])


def open_ome_zarr_level(zarr_path: str, scale_level: str):
    zarr_path = str(zarr_path).rstrip("/")
    scale_level = str(scale_level).strip("/")

    if zarr_path.startswith("s3://"):
        level_path = f"{zarr_path}/{scale_level}"
        s3 = s3fs.S3FileSystem(anon=False, skip_instance_cache=True)
        store = s3fs.S3Map(root=level_path, s3=s3, check=False)
        return da.from_zarr(store)

    level_path = str(Path(zarr_path).expanduser() / scale_level)
    return da.from_zarr(level_path)


# ----------------------------
# Parquet points / matches
# ----------------------------

def load_point_manifest(alignment_base: str):
    manifest = read_json(join_uri(alignment_base, "manifest.json"))
    return manifest["points"]


def load_match_index(alignment_base: str):
    return read_parquet(join_uri(alignment_base, "matches", "match_index.parquet"))


def point_key(tp_id: int, setup_id: int, label: str):
    return f"{int(tp_id)}/{int(setup_id)}/{label}"


def read_interest_points(point_manifest, setup: int, timepoint: int, label: str):
    rel_path = point_manifest[point_key(timepoint, setup, label)]
    uri = join_uri(ALIGNMENT_BASE, rel_path)
    df = read_parquet(uri)

    if len(df) == 0:
        return np.empty((0, 3), dtype=np.float32)

    return df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)


def match_row(match_index_df, tp_id: int, setup_id: int, label: str):
    rows = match_index_df[
        (match_index_df["timepoint"].astype(int) == int(tp_id)) &
        (match_index_df["setup"].astype(int) == int(setup_id)) &
        (match_index_df["label"].astype(str) == str(label))
    ]

    if len(rows) == 0:
        raise RuntimeError(f"No match_index row for timepoint={tp_id}, setup={setup_id}, label={label}")

    return rows.iloc[0]


def read_correspondences(match_index_df, tp_id: int, setup_id: int, label: str):
    row = match_row(match_index_df, tp_id, setup_id, label)
    uri = join_uri(ALIGNMENT_BASE, str(row["correspondences_path"]))
    return read_parquet(uri)


def get_pair_matches_one_direction(
    point_manifest,
    match_index_df,
    tp_id: int,
    setup_a: int,
    setup_b: int,
    label: str,
):
    corr_df = read_correspondences(match_index_df, tp_id, setup_a, label)

    if len(corr_df) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), []

    corr_df = corr_df[
        (corr_df["target_timepoint"].astype(int) == int(tp_id)) &
        (corr_df["target_setup"].astype(int) == int(setup_b))
    ]

    if len(corr_df) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), []

    src_loc = read_interest_points(point_manifest, setup_a, tp_id, label)

    chunks_a = []
    chunks_b = []
    metadata = []

    print(f"\nReading matches {setup_a} -> {setup_b}")
    print(f"  correspondence rows: {len(corr_df):,}")

    for target_label, rows in corr_df.groupby("target_label"):
        target_label = str(target_label)
        dst_loc = read_interest_points(point_manifest, setup_b, tp_id, target_label)

        src_idx = rows["source_point_id"].to_numpy(dtype=np.int64)
        dst_idx = rows["target_point_id"].to_numpy(dtype=np.int64)

        if len(src_idx) == 0:
            continue

        if src_idx.min() < 0 or src_idx.max() >= len(src_loc):
            raise IndexError(
                f"Source index out of bounds for setup {setup_a}: "
                f"min={src_idx.min()}, max={src_idx.max()}, len={len(src_loc)}"
            )

        if dst_idx.min() < 0 or dst_idx.max() >= len(dst_loc):
            raise IndexError(
                f"Target index out of bounds for setup {setup_b}: "
                f"min={dst_idx.min()}, max={dst_idx.max()}, len={len(dst_loc)}"
            )

        chunks_a.append(np.asarray(src_loc[src_idx], dtype=np.float32))
        chunks_b.append(np.asarray(dst_loc[dst_idx], dtype=np.float32))

        if "target_view_id" in rows.columns:
            target_view_ids = rows["target_view_id"].to_numpy(dtype=np.int64)
        else:
            target_view_ids = np.full(len(rows), -1)

        metadata.extend(
            {
                "src_setup": setup_a,
                "src_index": int(si),
                "dst_setup": setup_b,
                "dst_index": int(di),
                "target_view_id": int(gid),
                "label": target_label,
            }
            for si, di, gid in zip(src_idx, dst_idx, target_view_ids)
        )

    if not chunks_a:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), []

    return np.vstack(chunks_a).astype(np.float32), np.vstack(chunks_b).astype(np.float32), metadata


def get_pair_matches_bidirectional(point_manifest, match_index_df, tp_id: int, setup_a: int, setup_b: int, label: str):
    a1, b1, meta1 = get_pair_matches_one_direction(
        point_manifest=point_manifest,
        match_index_df=match_index_df,
        tp_id=tp_id,
        setup_a=setup_a,
        setup_b=setup_b,
        label=label,
    )

    b2, a2, meta2_raw = get_pair_matches_one_direction(
        point_manifest=point_manifest,
        match_index_df=match_index_df,
        tp_id=tp_id,
        setup_a=setup_b,
        setup_b=setup_a,
        label=label,
    )

    chunks_a = []
    chunks_b = []
    metadata = []

    if len(a1):
        chunks_a.append(a1)
        chunks_b.append(b1)
        metadata.extend(meta1)

    if len(a2):
        chunks_a.append(a2)
        chunks_b.append(b2)

        for m in meta2_raw:
            m2 = dict(m)
            m2["direction_was_reversed"] = True
            metadata.append(m2)

    if not chunks_a:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), []

    return np.vstack(chunks_a).astype(np.float32), np.vstack(chunks_b).astype(np.float32), metadata


# ----------------------------
# Geometry
# ----------------------------

def open_tile_volume_zyx(rec: dict):
    arr = open_ome_zarr_level(rec["full_path"], SCALE_LEVEL)

    if arr.ndim == 5:
        return arr[0, 0, :, :, :].astype(np.float32)

    if arr.ndim == 3:
        return arr.astype(np.float32)

    raise RuntimeError(f"Unexpected array shape for {rec['full_path']}: {arr.shape}")


def infer_scale_info(rec: dict, vol_zyx: da.Array) -> dict:
    z_scaled, y_scaled, x_scaled = vol_zyx.shape

    return {
        "scale_x": float(rec["size_x"]) / float(x_scaled),
        "scale_y": float(rec["size_y"]) / float(y_scaled),
        "scale_z": float(rec["size_z"]) / float(z_scaled),
    }


def transformed_xy_box_fullres(rec: dict):
    corners = np.asarray(
        [
            [0, 0, 0],
            [rec["size_x"], 0, 0],
            [0, rec["size_y"], 0],
            [rec["size_x"], rec["size_y"], 0],
        ],
        dtype=np.float32,
    )

    out = apply_affine_xyz(corners, rec["transform"])
    return float(np.min(out[:, 0])), float(np.max(out[:, 0])), float(np.min(out[:, 1])), float(np.max(out[:, 1]))


def compute_pairwise_overlap_fullres(rec_a: dict, rec_b: dict):
    ax0, ax1, ay0, ay1 = transformed_xy_box_fullres(rec_a)
    bx0, bx1, by0, by1 = transformed_xy_box_fullres(rec_b)

    ox0 = max(ax0, bx0)
    ox1 = min(ax1, bx1)
    oy0 = max(ay0, by0)
    oy1 = min(ay1, by1)

    if ox1 <= ox0 or oy1 <= oy0:
        return None

    return ox0, ox1, oy0, oy1


def list_overlapping_pairs(records):
    overlaps = []

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            rec_a = records[i]
            rec_b = records[j]
            overlap = compute_pairwise_overlap_fullres(rec_a, rec_b)

            if overlap is not None:
                overlaps.append((rec_a["setup"], rec_b["setup"], overlap))

    return overlaps


def nominal_overlap_to_local_padded_crop(rec: dict, overlap_box_fullres, pad_xy: int):
    ox0, ox1, oy0, oy1 = overlap_box_fullres
    inv = np.linalg.inv(rec["transform"])

    aligned_corners = np.asarray(
        [[ox0, oy0, 0], [ox1, oy0, 0], [ox0, oy1, 0], [ox1, oy1, 0]],
        dtype=np.float32,
    )
    local = apply_affine_xyz(aligned_corners, inv)

    lx0 = int(np.floor(np.min(local[:, 0]) - pad_xy))
    lx1 = int(np.ceil(np.max(local[:, 0]) + pad_xy))
    ly0 = int(np.floor(np.min(local[:, 1]) - pad_xy))
    ly1 = int(np.ceil(np.max(local[:, 1]) + pad_xy))

    lx0 = max(0, lx0)
    ly0 = max(0, ly0)
    lx1 = min(rec["size_x"], lx1)
    ly1 = min(rec["size_y"], ly1)

    if lx1 <= lx0 or ly1 <= ly0:
        return None

    return {"x0": lx0, "x1": lx1, "y0": ly0, "y1": ly1}


def fullres_crop_to_scaled(crop: dict, scale: dict, vol_zyx: da.Array):
    _, y_max, x_max = vol_zyx.shape

    sx0 = int(np.floor(crop["x0"] / scale["scale_x"]))
    sx1 = int(np.ceil(crop["x1"] / scale["scale_x"]))
    sy0 = int(np.floor(crop["y0"] / scale["scale_y"]))
    sy1 = int(np.ceil(crop["y1"] / scale["scale_y"]))

    sx0 = max(0, sx0)
    sy0 = max(0, sy0)
    sx1 = min(x_max, sx1)
    sy1 = min(y_max, sy1)

    if sx1 <= sx0 or sy1 <= sy0:
        return None

    return {"x0": sx0, "x1": sx1, "y0": sy0, "y1": sy1}


def filter_points_in_xy_crop(pts_xyz: np.ndarray, crop: dict):
    if pts_xyz is None or len(pts_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    mask = (
        (pts_xyz[:, 0] >= crop["x0"]) &
        (pts_xyz[:, 0] < crop["x1"]) &
        (pts_xyz[:, 1] >= crop["y0"]) &
        (pts_xyz[:, 1] < crop["y1"])
    )

    return pts_xyz[mask].astype(np.float32)


def filter_paired_points_in_xy_crops(pts_a_xyz: np.ndarray, pts_b_xyz: np.ndarray, crop_a: dict, crop_b: dict):
    if len(pts_a_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    mask_a = (
        (pts_a_xyz[:, 0] >= crop_a["x0"]) &
        (pts_a_xyz[:, 0] < crop_a["x1"]) &
        (pts_a_xyz[:, 1] >= crop_a["y0"]) &
        (pts_a_xyz[:, 1] < crop_a["y1"])
    )

    mask_b = (
        (pts_b_xyz[:, 0] >= crop_b["x0"]) &
        (pts_b_xyz[:, 0] < crop_b["x1"]) &
        (pts_b_xyz[:, 1] >= crop_b["y0"]) &
        (pts_b_xyz[:, 1] < crop_b["y1"])
    )

    mask = mask_a & mask_b
    return pts_a_xyz[mask].astype(np.float32), pts_b_xyz[mask].astype(np.float32)


def local_fullres_to_global_scaled(pts_full_xyz: np.ndarray, rec: dict, scale: dict):
    if pts_full_xyz is None or len(pts_full_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    transformed_full = apply_affine_xyz(pts_full_xyz, rec["transform"])

    out = np.empty_like(transformed_full, dtype=np.float32)
    out[:, 0] = transformed_full[:, 0] / scale["scale_x"]
    out[:, 1] = transformed_full[:, 1] / scale["scale_y"]
    out[:, 2] = transformed_full[:, 2] / scale["scale_z"]

    return out


def display_bounds_mask(pts_display_xyz: np.ndarray, shape_zyx):
    if pts_display_xyz is None or len(pts_display_xyz) == 0:
        return np.zeros((0,), dtype=bool)

    z, y, x = shape_zyx

    return (
        (pts_display_xyz[:, 0] >= 0) & (pts_display_xyz[:, 0] < x) &
        (pts_display_xyz[:, 1] >= 0) & (pts_display_xyz[:, 1] < y) &
        (pts_display_xyz[:, 2] >= 0) & (pts_display_xyz[:, 2] < z)
    )


def global_scaled_to_a_crop_display_unfiltered(pts_global_xyz: np.ndarray, rec_a: dict, crop_a_full: dict, scale_a: dict):
    if pts_global_xyz is None or len(pts_global_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    local_origin = np.asarray([[crop_a_full["x0"], crop_a_full["y0"], 0.0]], dtype=np.float32)
    transformed_origin_full = apply_affine_xyz(local_origin, rec_a["transform"])

    origin_x = transformed_origin_full[0, 0] / scale_a["scale_x"]
    origin_y = transformed_origin_full[0, 1] / scale_a["scale_y"]
    origin_z = transformed_origin_full[0, 2] / scale_a["scale_z"]

    pts = pts_global_xyz.astype(np.float32).copy()
    pts[:, 0] -= origin_x
    pts[:, 1] -= origin_y
    pts[:, 2] -= origin_z

    return pts.astype(np.float32)


def global_scaled_to_a_crop_display_filtered(
    pts_global_xyz: np.ndarray,
    rec_a: dict,
    crop_a_full: dict,
    scale_a: dict,
    image_a_crop_shape,
):
    pts = global_scaled_to_a_crop_display_unfiltered(pts_global_xyz, rec_a, crop_a_full, scale_a)
    mask = display_bounds_mask(pts, image_a_crop_shape)
    return pts[mask].astype(np.float32)


# ----------------------------
# Tile-B image overlay in setup-A display frame
# ----------------------------

def compute_a_display_origin_scaled(rec_a: dict, crop_a_full: dict, scale_a: dict):
    """
    Same origin convention used by global_scaled_to_a_crop_display_unfiltered().
    This keeps the B image overlay in the same display frame as transformed matches.
    """
    local_origin = np.asarray([[crop_a_full["x0"], crop_a_full["y0"], 0.0]], dtype=np.float32)
    transformed_origin_full = apply_affine_xyz(local_origin, rec_a["transform"])[0]

    return np.asarray(
        [
            transformed_origin_full[0] / scale_a["scale_x"],
            transformed_origin_full[1] / scale_a["scale_y"],
            transformed_origin_full[2] / scale_a["scale_z"],
        ],
        dtype=np.float32,
    )


def warp_b_into_a_display_volume(
    vol_a_crop: np.ndarray,
    rec_a: dict,
    scale_a: dict,
    crop_a_full: dict,
    rec_b: dict,
    scale_b: dict,
    crop_b_full: dict,
    vol_b_crop: np.ndarray,
):
    """
    Resample tile B into the same setup-A display frame used for plotting matches.

    Output is zyx and has exactly the same shape as vol_a_crop. This is only for QC
    visualization, not for fusion/math output.
    """
    z_max, y_max, x_max = vol_a_crop.shape
    out = np.zeros_like(vol_a_crop, dtype=np.float32)

    origin_a_scaled = compute_a_display_origin_scaled(rec_a, crop_a_full, scale_a)
    inv_b = np.linalg.inv(rec_b["transform"])

    x_coords = np.arange(x_max, dtype=np.float32)

    print(
        f"  warping setup {rec_b['setup']} into setup {rec_a['setup']} display frame: "
        f"output zyx={out.shape}, source-B-crop zyx={vol_b_crop.shape}"
    )

    for z in range(z_max):

        for y_start in range(0, y_max, WARP_Y_BLOCK):
            y_end = min(y_start + WARP_Y_BLOCK, y_max)
            y_coords = np.arange(y_start, y_end, dtype=np.float32)

            xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")

            gx_scaled = xx + origin_a_scaled[0]
            gy_scaled = yy + origin_a_scaled[1]
            gz_scaled = np.full_like(xx, z + origin_a_scaled[2], dtype=np.float32)

            global_full = np.stack(
                [
                    gx_scaled * scale_a["scale_x"],
                    gy_scaled * scale_a["scale_y"],
                    gz_scaled * scale_a["scale_z"],
                ],
                axis=-1,
            ).reshape(-1, 3)

            local_b_full = apply_affine_xyz(global_full, inv_b)

            xb = (local_b_full[:, 0] - crop_b_full["x0"]) / scale_b["scale_x"]
            yb = (local_b_full[:, 1] - crop_b_full["y0"]) / scale_b["scale_y"]
            zb = local_b_full[:, 2] / scale_b["scale_z"]

            valid = (
                (xb >= 0) & (xb < vol_b_crop.shape[2] - 1) &
                (yb >= 0) & (yb < vol_b_crop.shape[1] - 1) &
                (zb >= 0) & (zb < vol_b_crop.shape[0] - 1)
            )

            sampled = np.zeros((len(xb),), dtype=np.float32)

            if np.any(valid):
                sampled[valid] = map_coordinates(
                    vol_b_crop,
                    [zb[valid], yb[valid], xb[valid]],
                    order=1,
                    mode="constant",
                    cval=0.0,
                ).astype(np.float32)

            out[z, y_start:y_end, :] = sampled.reshape(y_end - y_start, x_max)

    return out


# ----------------------------
# Display normalization / RGB overlay
# ----------------------------

def normalize_slice(x: np.ndarray, lo: float, hi: float):
    """
    Display-only contrast/brightness normalization.

    P_LO/P_HI choose the intensity window. DISPLAY_GAMMA < 1 lifts dim tissue;
    DISPLAY_BRIGHTNESS_GAIN brightens after gamma. This does not affect any match
    coordinates, residuals, QC numbers, or saved data.
    """
    x = x.astype(np.float32, copy=False)

    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float32)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)

    if DISPLAY_BLACK_LEVEL > 0:
        denom = max(1e-6, 1.0 - float(DISPLAY_BLACK_LEVEL))
        x = np.clip((x - float(DISPLAY_BLACK_LEVEL)) / denom, 0.0, 1.0)

    if abs(float(DISPLAY_GAMMA) - 1.0) > 1e-6:
        x = np.power(np.clip(x, 0.0, 1.0), float(DISPLAY_GAMMA))

    if abs(float(DISPLAY_BRIGHTNESS_GAIN) - 1.0) > 1e-6:
        x = x * float(DISPLAY_BRIGHTNESS_GAIN)

    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


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


def normalize_for_display(slice2d: np.ndarray, global_lo: float, global_hi: float):
    if DISPLAY_NORM_MODE == "global":
        return normalize_slice(slice2d, global_lo, global_hi)

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


def make_a_rgb(slice_a: np.ndarray, lo_a: float, hi_a: float):
    """Render setup A as green instead of grayscale."""
    a_norm = normalize_for_display(slice_a, lo_a, hi_a)
    rgb = a_norm[..., None] * TILE_A_BASE_RGB[None, None, :]
    return np.clip(rgb, 0.0, 1.0)


def make_overlay_rgb(slice_a: np.ndarray, slice_b_warped: np.ndarray, lo_a: float, hi_a: float, lo_b: float, hi_b: float):
    """A = green base. B = purple transparent overlay. Overlay alpha follows B intensity."""
    a_rgb = make_a_rgb(slice_a, lo_a, hi_a)
    b_norm = normalize_for_display(slice_b_warped, lo_b, hi_b)

    alpha = TILE_B_OVERLAY_ALPHA * b_norm
    alpha = np.where(b_norm >= TILE_B_MIN_VISIBLE, alpha, 0.0).astype(np.float32)

    rgb = a_rgb * (1.0 - alpha[..., None]) + TILE_B_OVERLAY_RGB[None, None, :] * alpha[..., None]
    return np.clip(rgb, 0.0, 1.0)


# ----------------------------
# QC metrics
# ----------------------------

def safe_div(n, d):
    return float(n) / float(d) if d else np.nan


def grid_stats(pts_xyz: np.ndarray, shape_zyx, bins=GRID_BINS):
    if len(pts_xyz) == 0:
        return {"occupancy": np.nan, "max_cell_fraction": np.nan}

    z, y, x = shape_zyx
    pts_zyx = np.stack([pts_xyz[:, 2], pts_xyz[:, 1], pts_xyz[:, 0]], axis=1)
    hist, _ = np.histogramdd(pts_zyx, bins=bins, range=((0, z), (0, y), (0, x)))
    flat = hist.ravel()
    n = float(np.sum(flat))

    return {
        "occupancy": safe_div(np.count_nonzero(flat), np.prod(bins)),
        "max_cell_fraction": float(np.max(flat) / n) if n > 0 else np.nan,
    }


def duplicate_fraction(pts_xyz: np.ndarray, decimals: int = 2):
    if len(pts_xyz) == 0:
        return np.nan

    rounded = np.round(pts_xyz, decimals=decimals)
    unique = np.unique(rounded, axis=0)
    return 1.0 - safe_div(len(unique), len(pts_xyz))


def compute_match_qc(match_a_display: np.ndarray, match_b_display: np.ndarray, image_shape_zyx, scale: dict):
    if len(match_a_display) == 0:
        return None

    deltas = match_b_display - match_a_display
    raw_link_lengths = np.sqrt(np.sum(deltas ** 2, axis=1))

    median_delta = np.median(deltas, axis=0)
    centered_deltas = deltas - median_delta[None, :]
    consistency_residuals = np.sqrt(np.sum(centered_deltas ** 2, axis=1))

    inlier_mask = consistency_residuals <= INLIER_RESIDUAL_PX
    inlier_a = match_a_display[inlier_mask]
    grid = grid_stats(inlier_a, image_shape_zyx)

    qc = {
        "n_matches": int(len(consistency_residuals)),
        "n_inliers": int(np.count_nonzero(inlier_mask)),
        "inlier_ratio": 100.0 * safe_div(np.count_nonzero(inlier_mask), len(consistency_residuals)),
        "raw_median_link_scaled": float(np.median(raw_link_lengths)),
        "raw_p90_link_scaled": float(np.percentile(raw_link_lengths, 90)),
        "raw_p95_link_scaled": float(np.percentile(raw_link_lengths, 95)),
        "raw_max_link_scaled": float(np.max(raw_link_lengths)),
        "median_consistency_residual_scaled": float(np.median(consistency_residuals)),
        "p90_consistency_residual_scaled": float(np.percentile(consistency_residuals, 90)),
        "p95_consistency_residual_scaled": float(np.percentile(consistency_residuals, 95)),
        "max_consistency_residual_scaled": float(np.max(consistency_residuals)),
        "within_2px": 100.0 * float(np.mean(consistency_residuals <= 2.0)),
        "within_4px": 100.0 * float(np.mean(consistency_residuals <= 4.0)),
        "within_8px": 100.0 * float(np.mean(consistency_residuals <= 8.0)),
        "median_dx": float(median_delta[0]),
        "median_dy": float(median_delta[1]),
        "median_dz": float(median_delta[2]),
        "mad_dx": float(np.median(np.abs(deltas[:, 0] - median_delta[0]))),
        "mad_dy": float(np.median(np.abs(deltas[:, 1] - median_delta[1]))),
        "mad_dz": float(np.median(np.abs(deltas[:, 2] - median_delta[2]))),
        "grid_occupancy": grid["occupancy"],
        "max_cell_fraction": grid["max_cell_fraction"],
        "duplicate_a_fraction": duplicate_fraction(match_a_display),
        "duplicate_b_fraction": duplicate_fraction(match_b_display),
        "raw_link_lengths": raw_link_lengths,
        "consistency_residuals": consistency_residuals,
        "inlier_mask": inlier_mask,
        "median_delta": median_delta,
    }

    flags = []

    if qc["n_inliers"] < 50:
        flags.append("too few inlier matches")

    if not np.isfinite(qc["inlier_ratio"]) or qc["inlier_ratio"] < 35.0:
        flags.append("low inlier ratio")

    if qc["median_consistency_residual_scaled"] > 3.0:
        flags.append("high median consistency residual")

    if qc["p90_consistency_residual_scaled"] > 8.0:
        flags.append("high p90 consistency residual")

    if np.isfinite(qc["grid_occupancy"]) and qc["grid_occupancy"] < 0.02:
        flags.append("poor spatial spread across overlap")

    if np.isfinite(qc["max_cell_fraction"]) and qc["max_cell_fraction"] > 0.75:
        flags.append("matches concentrated in one grid cell")

    if np.isfinite(qc["duplicate_a_fraction"]) and qc["duplicate_a_fraction"] > 0.20:
        flags.append("many duplicate A match coordinates")

    if np.isfinite(qc["duplicate_b_fraction"]) and qc["duplicate_b_fraction"] > 0.20:
        flags.append("many duplicate B match coordinates")

    if not flags:
        verdict = "GOOD"
    elif len(flags) <= 2 and qc["n_inliers"] >= 20:
        verdict = "CHECK"
    else:
        verdict = "BAD"

    qc["flags"] = flags
    qc["verdict"] = verdict
    return qc


def print_match_qc(qc, scale: dict):
    print("\n" + "=" * 72)
    print("MATCH QC")
    print("=" * 72)

    if qc is None:
        print("No visible matches in display frame.")
        print("=" * 72 + "\n")
        return

    print(f"matches:                     {qc['n_matches']:,}")
    print(f"inliers <= {INLIER_RESIDUAL_PX:.1f}px:          {qc['n_inliers']:,}")
    print(f"inlier ratio:                {qc['inlier_ratio']:.2f}%")

    print("\nRaw A->B link length in scaled display pixels:")
    print(f"  median:                    {qc['raw_median_link_scaled']:.3f}")
    print(f"  p90:                       {qc['raw_p90_link_scaled']:.3f}")
    print(f"  p95:                       {qc['raw_p95_link_scaled']:.3f}")
    print(f"  max:                       {qc['raw_max_link_scaled']:.3f}")

    print("\nConsistency residual after subtracting median shift:")
    print(f"  median:                    {qc['median_consistency_residual_scaled']:.3f}")
    print(f"  p90:                       {qc['p90_consistency_residual_scaled']:.3f}")
    print(f"  p95:                       {qc['p95_consistency_residual_scaled']:.3f}")
    print(f"  max:                       {qc['max_consistency_residual_scaled']:.3f}")

    print("\nMedian shift B-A in scaled display pixels:")
    print(f"  dx:                        {qc['median_dx']:.3f}")
    print(f"  dy:                        {qc['median_dy']:.3f}")
    print(f"  dz:                        {qc['median_dz']:.3f}")
    print("\nMedian absolute deviation:")
    print(f"  mad dx:                    {qc['mad_dx']:.3f}")
    print(f"  mad dy:                    {qc['mad_dy']:.3f}")
    print(f"  mad dz:                    {qc['mad_dz']:.3f}")

    print("\nConsistency thresholds:")
    print(f"  within 2px:                 {qc['within_2px']:.2f}%")
    print(f"  within 4px:                 {qc['within_4px']:.2f}%")
    print(f"  within 8px:                 {qc['within_8px']:.2f}%")

    print("\nSpatial spread / duplicates:")
    print(f"  grid occupancy:             {qc['grid_occupancy']:.4f}")
    print(f"  max cell fraction:          {qc['max_cell_fraction']:.4f}")
    print(f"  duplicate A fraction:       {qc['duplicate_a_fraction']:.4f}")
    print(f"  duplicate B fraction:       {qc['duplicate_b_fraction']:.4f}")

    print("")
    print(f"VERDICT: {qc['verdict']}")

    if qc["flags"]:
        print("Reason flags:")
        for flag in qc["flags"]:
            print(f"  - {flag}")

    print("=" * 72 + "\n")


# ----------------------------
# Match viewer helpers
# ----------------------------

def color_for_residual(r: float):
    if r <= GOOD_RESIDUAL_PX:
        return "lime"
    if r <= OK_RESIDUAL_PX:
        return "yellow"
    if r <= BAD_RESIDUAL_PX:
        return "orange"
    return "magenta"


def make_pair_ring_segments(
    a_xy: np.ndarray,
    b_xy: np.ndarray,
    min_radius_px: float = 7.0,
    pad_px: float = 4.0,
    n_vertices: int = 48,
):
    if len(a_xy) == 0:
        return []

    theta = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=True)
    unit = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)

    rings = []

    for a, b in zip(a_xy, b_xy):
        center = 0.5 * (a + b)
        dist = float(np.linalg.norm(b - a))
        radius = max(min_radius_px, 0.5 * dist + pad_px)
        rings.append(center[None, :] + radius * unit)

    return rings


def visible_matches_for_slice(match_a: np.ndarray, match_b: np.ndarray, residuals: np.ndarray, z: int, z_radius: float):
    if len(match_a) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2, 2), dtype=np.float32),
            [],
            [],
            np.empty((0,), dtype=np.float32),
        )

    za = match_a[:, 2]
    zb = match_b[:, 2]
    z_mid = 0.5 * (za + zb)
    mask = np.abs(z_mid - float(z)) <= z_radius

    if not np.any(mask):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2, 2), dtype=np.float32),
            [],
            [],
            np.empty((0,), dtype=np.float32),
        )

    a = match_a[mask].astype(np.float32)
    b = match_b[mask].astype(np.float32)
    r = residuals[mask].astype(np.float32)

    segments = np.stack([a[:, :2], b[:, :2]], axis=1).astype(np.float32)
    link_colors = [color_for_residual(float(x)) for x in r]
    ring_segments = make_pair_ring_segments(a[:, :2], b[:, :2])

    return a, b, segments, ring_segments, link_colors, r


def full_view_figsize():
    """
    Keep the first/full slider contained in a normal notebook/browser viewport.
    The previous 10x area scaling made the rotated long-skinny view render off-screen.
    """
    linear = float(np.sqrt(max(1.0, float(FULL_VIEW_AREA_SCALE))))
    width = float(FULL_VIEW_BASE_FIGSIZE[0]) * linear
    height = float(FULL_VIEW_BASE_FIGSIZE[1]) * linear

    max_width = float(FULL_VIEW_MAX_FIGSIZE[0])
    max_height = float(FULL_VIEW_MAX_FIGSIZE[1])

    shrink = min(1.0, max_width / max(width, 1e-6), max_height / max(height, 1e-6))
    return (width * shrink, height * shrink)


def rotate_image_90(image_rgb: np.ndarray, rotate_90: bool, direction: str):
    if not rotate_90:
        return image_rgb

    if direction == "counterclockwise":
        return np.rot90(image_rgb, k=1)

    return np.rot90(image_rgb, k=3)


def rotate_xy_90(points_xy: np.ndarray, original_shape_hw, rotate_90: bool, direction: str):
    pts = points_xy.astype(np.float32, copy=True)

    if not rotate_90 or len(pts) == 0:
        return pts

    h, w = original_shape_hw
    x_old = pts[:, 0].copy()
    y_old = pts[:, 1].copy()

    if direction == "counterclockwise":
        # np.rot90(k=1): x' = y, y' = w - 1 - x
        pts[:, 0] = y_old
        pts[:, 1] = (w - 1) - x_old
    else:
        # np.rot90(k=3): x' = h - 1 - y, y' = x
        pts[:, 0] = (h - 1) - y_old
        pts[:, 1] = x_old

    return pts


def rotate_segments_and_rings_90(
    segments: np.ndarray,
    rings,
    original_shape_hw,
    rotate_90: bool,
    direction: str,
):
    if not rotate_90:
        return segments, rings

    if len(segments):
        seg = segments.astype(np.float32, copy=True)
        flat = seg.reshape(-1, 2)
        flat_rot = rotate_xy_90(flat, original_shape_hw, rotate_90, direction)
        seg = flat_rot.reshape(seg.shape)
    else:
        seg = segments

    rotated_rings = [
        rotate_xy_90(ring, original_shape_hw, rotate_90, direction)
        for ring in rings
    ]

    return seg, rotated_rings


def rotate_full_view_image(image_rgb: np.ndarray):
    return rotate_image_90(image_rgb, FULL_VIEW_ROTATE_90, FULL_VIEW_ROTATE_DIRECTION)


def rotate_full_view_xy(points_xy: np.ndarray, original_shape_hw):
    return rotate_xy_90(points_xy, original_shape_hw, FULL_VIEW_ROTATE_90, FULL_VIEW_ROTATE_DIRECTION)


def rotate_full_view_segments_and_rings(segments: np.ndarray, rings, original_shape_hw):
    return rotate_segments_and_rings_90(
        segments,
        rings,
        original_shape_hw,
        FULL_VIEW_ROTATE_90,
        FULL_VIEW_ROTATE_DIRECTION,
    )


def rotate_split_view_image(image_rgb: np.ndarray):
    return rotate_image_90(image_rgb, SPLIT_VIEW_ROTATE_90, SPLIT_VIEW_ROTATE_DIRECTION)


def rotate_split_view_xy(points_xy: np.ndarray, original_shape_hw):
    return rotate_xy_90(points_xy, original_shape_hw, SPLIT_VIEW_ROTATE_90, SPLIT_VIEW_ROTATE_DIRECTION)


def rotate_split_view_segments_and_rings(segments: np.ndarray, rings, original_shape_hw):
    return rotate_segments_and_rings_90(
        segments,
        rings,
        original_shape_hw,
        SPLIT_VIEW_ROTATE_90,
        SPLIT_VIEW_ROTATE_DIRECTION,
    )


def set_image_axes_limits(ax, image_rgb: np.ndarray):
    h, w = image_rgb.shape[:2]
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)


# ----------------------------
# First/full slider view - rotated sideways
# ----------------------------

def view_match_pair_slider(
    vol_a_crop: np.ndarray,
    vol_b_in_a_crop: Optional[np.ndarray],
    pts_a_display: np.ndarray,
    pts_b_display: np.ndarray,
    match_a_display: np.ndarray,
    match_b_display: np.ndarray,
    qc,
    setup_a: int,
    setup_b: int,
):
    z_max, _, _ = vol_a_crop.shape
    residuals = qc["consistency_residuals"] if qc is not None else np.empty((0,), dtype=np.float32)

    global_lo_a, global_hi_a = compute_global_display_range(vol_a_crop)

    if SHOW_TILE_B_PURPLE_OVERLAY and vol_b_in_a_crop is not None:
        global_lo_b, global_hi_b = compute_global_display_range(vol_b_in_a_crop)

        def render_image(z):
            return make_overlay_rgb(
                vol_a_crop[z],
                vol_b_in_a_crop[z],
                global_lo_a,
                global_hi_a,
                global_lo_b,
                global_hi_b,
            )

        image_kwargs = {"origin": "upper", "aspect": IMAGE_ASPECT}
        title_prefix = f"green=A setup {setup_a} | purple=B setup {setup_b}"
    else:
        def render_image(z):
            return make_a_rgb(vol_a_crop[z], global_lo_a, global_hi_a)

        image_kwargs = {"origin": "upper", "aspect": IMAGE_ASPECT}
        title_prefix = f"green=A setup {setup_a} only"

    z0 = z_max // 2

    fig, ax = plt.subplots(figsize=full_view_figsize())
    plt.subplots_adjust(bottom=0.20, top=0.94, left=0.04, right=0.98)

    image0_raw = render_image(z0)
    original_shape_hw = image0_raw.shape[:2]
    image0 = rotate_full_view_image(image0_raw)

    im = ax.imshow(image0, **image_kwargs)
    set_image_axes_limits(ax, image0)

    a0, b0, seg0, rings0, colors0, _ = visible_matches_for_slice(
        match_a_display,
        match_b_display,
        residuals,
        z0,
        LINK_Z_RADIUS_SCALED,
    )

    a0_xy = rotate_full_view_xy(
        a0[:, :2] if len(a0) else np.empty((0, 2), dtype=np.float32),
        original_shape_hw,
    )
    b0_xy = rotate_full_view_xy(
        b0[:, :2] if len(b0) else np.empty((0, 2), dtype=np.float32),
        original_shape_hw,
    )
    seg0, rings0 = rotate_full_view_segments_and_rings(seg0, rings0, original_shape_hw)

    pair_rings = LineCollection(rings0, colors="cyan", linewidths=1.6, alpha=0.95, zorder=7)
    ax.add_collection(pair_rings)

    link_halo = LineCollection(seg0, colors="white", linewidths=4.0, alpha=0.95, zorder=8)
    ax.add_collection(link_halo)

    links = LineCollection(seg0, colors=colors0, linewidths=2.0, alpha=1.0, zorder=9)
    ax.add_collection(links)

    sc_a_match = ax.scatter(
        a0_xy[:, 0] if len(a0_xy) else [],
        a0_xy[:, 1] if len(a0_xy) else [],
        s=36,
        c="red",
        marker="o",
        alpha=0.98,
        edgecolors="white",
        linewidths=0.8,
        label=f"A match setup {setup_a}",
        zorder=10,
    )

    sc_b_match = ax.scatter(
        b0_xy[:, 0] if len(b0_xy) else [],
        b0_xy[:, 1] if len(b0_xy) else [],
        s=36,
        c="blue",
        marker="o",
        alpha=0.98,
        edgecolors="white",
        linewidths=0.8,
        label=f"B match setup {setup_b}",
        zorder=10,
    )

    rotate_label = "rotated sideways" if FULL_VIEW_ROTATE_90 else "normal orientation"
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.set_title(
        f"FULL VIEW {rotate_label} | Match QC after full XML transform stack | "
        f"{title_prefix} | red=A match | blue=B match",
        fontsize=11,
        pad=8,
    )

    info_text = fig.text(0.08, 0.125, "", ha="left", va="center", fontsize=10, family="monospace")
    slider_ax = fig.add_axes([0.08, 0.055, 0.86, 0.035])
    slider = Slider(slider_ax, "Z", 0, z_max - 1, valinit=z0, valstep=1)
    fig._match_qc_slider = slider  # keep matplotlib widget alive

    def update(val):
        z = int(val)

        image_now_raw = render_image(z)
        original_shape_hw = image_now_raw.shape[:2]
        image_now = rotate_full_view_image(image_now_raw)

        a_now, b_now, seg, rings, colors, r = visible_matches_for_slice(
            match_a_display,
            match_b_display,
            residuals,
            z,
            LINK_Z_RADIUS_SCALED,
        )

        a_now_xy = rotate_full_view_xy(
            a_now[:, :2] if len(a_now) else np.empty((0, 2), dtype=np.float32),
            original_shape_hw,
        )
        b_now_xy = rotate_full_view_xy(
            b_now[:, :2] if len(b_now) else np.empty((0, 2), dtype=np.float32),
            original_shape_hw,
        )
        seg, rings = rotate_full_view_segments_and_rings(seg, rings, original_shape_hw)

        im.set_data(image_now)
        set_image_axes_limits(ax, image_now)

        sc_a_match.set_offsets(a_now_xy if len(a_now_xy) else np.empty((0, 2)))
        sc_b_match.set_offsets(b_now_xy if len(b_now_xy) else np.empty((0, 2)))

        pair_rings.set_segments(rings)
        link_halo.set_segments(seg)
        links.set_segments(seg)
        links.set_color(colors)

        med = float(np.median(r)) if len(r) else np.nan
        p90 = float(np.percentile(r, 90)) if len(r) else np.nan
        verdict = qc["verdict"] if qc is not None else "NO MATCHES"

        info_text.set_text(
            f"z={z:4d} | visible matched pairs={len(seg):5,d} | "
            f"residual med={med:5.2f}px p90={p90:5.2f}px | {verdict}"
        )

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(z0)
    plt.show()


# ----------------------------
# Chunk sliders - rotated sideways and independently scrollable
# ----------------------------

def visible_matches_for_slice_y_window(
    match_a: np.ndarray,
    match_b: np.ndarray,
    residuals: np.ndarray,
    z: int,
    z_radius: float,
    y0: int,
    y1: int,
    link_by: str = ZOOM_LINK_BY,
):
    a, b, _, _, _, r = visible_matches_for_slice(
        match_a,
        match_b,
        residuals,
        z,
        z_radius,
    )

    if len(a) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2, 2), dtype=np.float32),
            [],
            [],
            np.empty((0,), dtype=np.float32),
        )

    if link_by == "either_endpoint":
        mask = (
            ((a[:, 1] >= y0) & (a[:, 1] < y1)) |
            ((b[:, 1] >= y0) & (b[:, 1] < y1))
        )
    else:
        y_mid = 0.5 * (a[:, 1] + b[:, 1])
        mask = (y_mid >= y0) & (y_mid < y1)

    if not np.any(mask):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2, 2), dtype=np.float32),
            [],
            [],
            np.empty((0,), dtype=np.float32),
        )

    a_now = a[mask].astype(np.float32).copy()
    b_now = b[mask].astype(np.float32).copy()
    r_now = r[mask].astype(np.float32)

    a_now[:, 1] -= float(y0)
    b_now[:, 1] -= float(y0)

    seg = np.stack([a_now[:, :2], b_now[:, :2]], axis=1).astype(np.float32)
    ring_segments = make_pair_ring_segments(a_now[:, :2], b_now[:, :2])
    link_colors = [color_for_residual(float(x)) for x in r_now]

    return a_now, b_now, seg, ring_segments, link_colors, r_now


def view_match_pair_y_chunk_sliders(
    vol_a_crop: np.ndarray,
    vol_b_in_a_crop: Optional[np.ndarray],
    match_a_display: np.ndarray,
    match_b_display: np.ndarray,
    qc,
    setup_a: int,
    setup_b: int,
    n_y_chunks: int = ZOOM_Y_CHUNKS,
):
    z_max, y_max, x_max = vol_a_crop.shape
    residuals = qc["consistency_residuals"] if qc is not None else np.empty((0,), dtype=np.float32)

    global_lo_a, global_hi_a = compute_global_display_range(vol_a_crop)

    if SHOW_TILE_B_PURPLE_OVERLAY and vol_b_in_a_crop is not None:
        global_lo_b, global_hi_b = compute_global_display_range(vol_b_in_a_crop)

        def render_chunk(z, y0, y1):
            return make_overlay_rgb(
                vol_a_crop[z, y0:y1, :],
                vol_b_in_a_crop[z, y0:y1, :],
                global_lo_a,
                global_hi_a,
                global_lo_b,
                global_hi_b,
            )

        image_kwargs = {"origin": "upper", "aspect": IMAGE_ASPECT}
        title_prefix = f"green=A setup {setup_a} | purple=B setup {setup_b}"
    else:
        def render_chunk(z, y0, y1):
            return make_a_rgb(vol_a_crop[z, y0:y1, :], global_lo_a, global_hi_a)

        image_kwargs = {"origin": "upper", "aspect": IMAGE_ASPECT}
        title_prefix = f"green=A setup {setup_a} only"

    z0 = z_max // 2
    chunk_edges = np.linspace(0, y_max, int(n_y_chunks) + 1).astype(int)
    chunk_sliders = []

    for chunk_i in range(int(n_y_chunks)):
        y0 = int(chunk_edges[chunk_i])
        y1 = int(chunk_edges[chunk_i + 1])

        if y1 <= y0:
            continue

        fig, ax = plt.subplots(figsize=SPLIT_VIEW_FIGSIZE)
        plt.subplots_adjust(bottom=0.20, top=0.91, left=0.04, right=0.98)

        image0_raw = render_chunk(z0, y0, y1)
        original_shape_hw = image0_raw.shape[:2]
        image0 = rotate_split_view_image(image0_raw)

        im = ax.imshow(image0, **image_kwargs)
        set_image_axes_limits(ax, image0)

        a0, b0, seg0, rings0, colors0, _ = visible_matches_for_slice_y_window(
            match_a_display,
            match_b_display,
            residuals,
            z0,
            LINK_Z_RADIUS_SCALED,
            y0,
            y1,
        )

        a0_xy = rotate_split_view_xy(
            a0[:, :2] if len(a0) else np.empty((0, 2), dtype=np.float32),
            original_shape_hw,
        )
        b0_xy = rotate_split_view_xy(
            b0[:, :2] if len(b0) else np.empty((0, 2), dtype=np.float32),
            original_shape_hw,
        )
        seg0, rings0 = rotate_split_view_segments_and_rings(seg0, rings0, original_shape_hw)

        pair_rings = LineCollection(rings0, colors="cyan", linewidths=1.6, alpha=0.95, zorder=7)
        ax.add_collection(pair_rings)

        split_halo_alpha = 0.95 if SPLIT_VIEW_LINK_SHOW_HALO else 0.0
        link_halo = LineCollection(seg0, colors="white", linewidths=4.0, alpha=split_halo_alpha, zorder=8)
        ax.add_collection(link_halo)

        links = LineCollection(
            seg0,
            colors=colors0,
            linewidths=SPLIT_VIEW_LINK_LINEWIDTH,
            alpha=SPLIT_VIEW_LINK_ALPHA,
            linestyles=SPLIT_VIEW_LINK_LINESTYLE,
            zorder=9,
        )
        ax.add_collection(links)

        sc_a_match = ax.scatter(
            a0_xy[:, 0] if len(a0_xy) else [],
            a0_xy[:, 1] if len(a0_xy) else [],
            s=42,
            facecolors="none",
            edgecolors="red",
            marker="o",
            alpha=1.0,
            linewidths=1.4,
            label=f"A match setup {setup_a}",
            zorder=10,
        )

        sc_b_match = ax.scatter(
            b0_xy[:, 0] if len(b0_xy) else [],
            b0_xy[:, 1] if len(b0_xy) else [],
            s=42,
            facecolors="none",
            edgecolors="blue",
            marker="o",
            alpha=1.0,
            linewidths=1.4,
            label=f"B match setup {setup_b}",
            zorder=10,
        )

        split_rotate_label = "rotated sideways" if SPLIT_VIEW_ROTATE_90 else "normal orientation"
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.set_title(
            f"ZOOMED Y chunk {chunk_i + 1}/{n_y_chunks} {split_rotate_label} | "
            f"display y[{y0}:{y1}] | "
            f"{title_prefix} | red=A match | blue=B match",
            fontsize=11,
            pad=8,
        )

        info_text = fig.text(0.08, 0.125, "", ha="left", va="center", fontsize=10, family="monospace")
        slider_ax = fig.add_axes([0.08, 0.055, 0.86, 0.035])
        slider = Slider(slider_ax, "Z", 0, z_max - 1, valinit=z0, valstep=1)
        chunk_sliders.append(slider)
        fig._match_qc_slider = slider  # keep this chunk's widget alive

        def update(
            val,
            *,
            im=im,
            sc_a_match=sc_a_match,
            sc_b_match=sc_b_match,
            pair_rings=pair_rings,
            link_halo=link_halo,
            links=links,
            info_text=info_text,
            fig=fig,
            ax=ax,
            y0=y0,
            y1=y1,
            chunk_i=chunk_i,
        ):
            z = int(val)

            image_now_raw = render_chunk(z, y0, y1)
            original_shape_hw = image_now_raw.shape[:2]
            image_now = rotate_split_view_image(image_now_raw)

            im.set_data(image_now)
            set_image_axes_limits(ax, image_now)

            a_now, b_now, seg, rings, colors, r = visible_matches_for_slice_y_window(
                match_a_display,
                match_b_display,
                residuals,
                z,
                LINK_Z_RADIUS_SCALED,
                y0,
                y1,
            )

            a_now_xy = rotate_split_view_xy(
                a_now[:, :2] if len(a_now) else np.empty((0, 2), dtype=np.float32),
                original_shape_hw,
            )
            b_now_xy = rotate_split_view_xy(
                b_now[:, :2] if len(b_now) else np.empty((0, 2), dtype=np.float32),
                original_shape_hw,
            )
            seg, rings = rotate_split_view_segments_and_rings(seg, rings, original_shape_hw)

            sc_a_match.set_offsets(a_now_xy if len(a_now_xy) else np.empty((0, 2)))
            sc_b_match.set_offsets(b_now_xy if len(b_now_xy) else np.empty((0, 2)))

            pair_rings.set_segments(rings)
            link_halo.set_segments(seg)
            links.set_segments(seg)
            links.set_color(colors)

            med = float(np.median(r)) if len(r) else np.nan
            p90 = float(np.percentile(r, 90)) if len(r) else np.nan
            verdict = qc["verdict"] if qc is not None else "NO MATCHES"

            info_text.set_text(
                f"chunk={chunk_i + 1}/{n_y_chunks} | y=[{y0}:{y1}] | z={z:4d} | "
                f"visible matched pairs={len(seg):5,d} | "
                f"residual med={med:5.2f}px p90={p90:5.2f}px | {verdict}"
            )

            fig.canvas.draw_idle()

        slider.on_changed(update)
        update(z0)

    plt.show()


# ----------------------------
# Pair prep / main
# ----------------------------

def dedupe_match_pairs(match_a_local, match_b_local, decimals=3):
    if len(match_a_local) == 0:
        return match_a_local, match_b_local

    if len(match_a_local) != len(match_b_local):
        raise RuntimeError(f"Cannot dedupe unpaired match arrays: A={len(match_a_local)}, B={len(match_b_local)}")

    key_arr = np.concatenate(
        [np.round(match_a_local, decimals=decimals), np.round(match_b_local, decimals=decimals)],
        axis=1,
    )
    _, keep_idx = np.unique(key_arr, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)

    return match_a_local[keep_idx].astype(np.float32), match_b_local[keep_idx].astype(np.float32)


def prepare_pair_data(rec: dict, overlap_box_fullres, point_manifest):
    print(f"\nPreparing setup {rec['setup']}...")

    vol = open_tile_volume_zyx(rec)
    scale = infer_scale_info(rec, vol)

    print(f"  input level shape zyx: {tuple(vol.shape)}")
    print(
        "  scale full/scaled xyz: "
        f"x={scale['scale_x']:.4f}, y={scale['scale_y']:.4f}, z={scale['scale_z']:.4f}"
    )

    crop_full = nominal_overlap_to_local_padded_crop(rec, overlap_box_fullres, OVERLAP_PAD_XY)
    if crop_full is None:
        raise RuntimeError(f"Could not make full-res crop for setup {rec['setup']}")

    crop_scaled = fullres_crop_to_scaled(crop_full, scale, vol)
    if crop_scaled is None:
        raise RuntimeError(f"Could not make scaled crop for setup {rec['setup']}")

    print(
        f"  full-res local crop x[{crop_full['x0']}:{crop_full['x1']}] "
        f"y[{crop_full['y0']}:{crop_full['y1']}]"
    )
    print(
        f"  scaled crop x[{crop_scaled['x0']}:{crop_scaled['x1']}] "
        f"y[{crop_scaled['y0']}:{crop_scaled['y1']}]"
    )

    vol_crop = vol[
        :,
        crop_scaled["y0"]:crop_scaled["y1"],
        crop_scaled["x0"]:crop_scaled["x1"],
    ].compute().astype(np.float32)

    pts_local = read_interest_points(point_manifest, rec["setup"], TIMEPOINT, MATCH_LABEL)
    pts_local_crop = filter_points_in_xy_crop(pts_local, crop_full)
    pts_global_scaled = local_fullres_to_global_scaled(pts_local_crop, rec, scale)

    print(f"  local points in crop: {len(pts_local_crop):,}")

    return {
        "vol": vol,
        "scale": scale,
        "crop_full": crop_full,
        "crop_scaled": crop_scaled,
        "vol_crop": vol_crop,
        "pts_local_crop": pts_local_crop,
        "pts_global_scaled": pts_global_scaled,
    }


def main():
    print(f"XML_PATH:        {XML_PATH}")
    print(f"ALIGNMENT_BASE:  {ALIGNMENT_BASE}")
    print(f"SCALE_LEVEL:     {SCALE_LEVEL}")
    print(f"MATCH_LABEL:     {MATCH_LABEL}")
    print(f"FULL_VIEW_ROTATE_90: {FULL_VIEW_ROTATE_90} ({FULL_VIEW_ROTATE_DIRECTION})")
    print(f"FULL_VIEW_FIGSIZE: {full_view_figsize()} max={FULL_VIEW_MAX_FIGSIZE}")
    print(f"SPLIT_VIEW_ROTATE_90: {SPLIT_VIEW_ROTATE_90} ({SPLIT_VIEW_ROTATE_DIRECTION})")
    print(f"SPLIT_VIEW_FIGSIZE: {SPLIT_VIEW_FIGSIZE}")

    root = load_xml_root(XML_PATH)
    point_manifest = load_point_manifest(ALIGNMENT_BASE)
    match_index_df = load_match_index(ALIGNMENT_BASE)

    records = parse_zarr_tile_records(root)
    print(f"Found tile records: {len(records)}")

    if len(records) == 0:
        print("No tile records found.")
        return

    by_setup = {r["setup"]: r for r in records}
    overlaps = list_overlapping_pairs(records)

    if len(overlaps) == 0:
        print("No overlapping pairs found.")
        return

    print("\nOverlapping pairs:")
    for a, b, (ox0, ox1, oy0, oy1) in overlaps:
        print(f"  ({a}, {b}) overlap full-res x[{ox0:.1f}:{ox1:.1f}] y[{oy0:.1f}:{oy1:.1f}]")

    if TARGET_PAIR is None:
        setup_a, setup_b, overlap_box = overlaps[0]
        print(f"\nTARGET_PAIR is None, using first overlap: ({setup_a}, {setup_b})")
    else:
        setup_a, setup_b = TARGET_PAIR

        if setup_a not in by_setup or setup_b not in by_setup:
            raise RuntimeError(f"TARGET_PAIR {TARGET_PAIR} not found in XML records")

        overlap_box = compute_pairwise_overlap_fullres(by_setup[setup_a], by_setup[setup_b])

        if overlap_box is None:
            raise RuntimeError(f"TARGET_PAIR {TARGET_PAIR} does not overlap in XY")

        print(f"\nUsing TARGET_PAIR: ({setup_a}, {setup_b})")

    rec_a = by_setup[setup_a]
    rec_b = by_setup[setup_b]

    data_a = prepare_pair_data(rec_a, overlap_box, point_manifest)
    data_b = prepare_pair_data(rec_b, overlap_box, point_manifest)

    vol_a_crop = data_a["vol_crop"]
    vol_b_crop = data_b["vol_crop"]

    vol_b_in_a_crop = None
    if SHOW_TILE_B_PURPLE_OVERLAY:
        print(f"\nWarping setup {setup_b} image into setup {setup_a} display frame for purple overlay...")
        vol_b_in_a_crop = warp_b_into_a_display_volume(
            vol_a_crop=vol_a_crop,
            rec_a=rec_a,
            scale_a=data_a["scale"],
            crop_a_full=data_a["crop_full"],
            rec_b=rec_b,
            scale_b=data_b["scale"],
            crop_b_full=data_b["crop_full"],
            vol_b_crop=vol_b_crop,
        )

    pts_a_display = global_scaled_to_a_crop_display_filtered(
        data_a["pts_global_scaled"],
        rec_a,
        data_a["crop_full"],
        data_a["scale"],
        vol_a_crop.shape,
    )

    pts_b_display = global_scaled_to_a_crop_display_filtered(
        data_b["pts_global_scaled"],
        rec_a,
        data_a["crop_full"],
        data_a["scale"],
        vol_a_crop.shape,
    )

    print("\nDisplay-frame all points:")
    print(f"  setup {setup_a}: {len(pts_a_display):,}")
    print(f"  setup {setup_b}: {len(pts_b_display):,}")

    match_a_local, match_b_local, match_meta = get_pair_matches_bidirectional(
        point_manifest=point_manifest,
        match_index_df=match_index_df,
        tp_id=TIMEPOINT,
        setup_a=setup_a,
        setup_b=setup_b,
        label=MATCH_LABEL,
    )

    print(f"\nLoaded raw bidirectional matches for pair ({setup_a}, {setup_b}): {len(match_a_local):,}")
    print(f"Match metadata rows before dedupe: {len(match_meta):,}")

    match_a_local, match_b_local = dedupe_match_pairs(match_a_local, match_b_local, decimals=3)
    print(f"Raw matches after A/B pair dedupe: {len(match_a_local):,}")

    match_a_local_crop, match_b_local_crop = filter_paired_points_in_xy_crops(
        match_a_local,
        match_b_local,
        data_a["crop_full"],
        data_b["crop_full"],
    )

    print(f"Matches after paired XY crop filter: {len(match_a_local_crop):,}")

    match_a_global_scaled = local_fullres_to_global_scaled(match_a_local_crop, rec_a, data_a["scale"])
    match_b_global_scaled = local_fullres_to_global_scaled(match_b_local_crop, rec_b, data_b["scale"])

    match_a_display_all = global_scaled_to_a_crop_display_unfiltered(
        match_a_global_scaled,
        rec_a,
        data_a["crop_full"],
        data_a["scale"],
    )
    match_b_display_all = global_scaled_to_a_crop_display_unfiltered(
        match_b_global_scaled,
        rec_a,
        data_a["crop_full"],
        data_a["scale"],
    )

    pair_display_mask = (
        display_bounds_mask(match_a_display_all, vol_a_crop.shape) &
        display_bounds_mask(match_b_display_all, vol_a_crop.shape)
    )

    match_a_display = match_a_display_all[pair_display_mask].astype(np.float32)
    match_b_display = match_b_display_all[pair_display_mask].astype(np.float32)

    print(f"Matches visible in setup-A display frame: {len(match_a_display):,}")

    qc = compute_match_qc(match_a_display, match_b_display, vol_a_crop.shape, data_a["scale"])
    print_match_qc(qc, data_a["scale"])

    view_match_pair_slider(
        vol_a_crop=vol_a_crop,
        vol_b_in_a_crop=vol_b_in_a_crop,
        pts_a_display=pts_a_display,
        pts_b_display=pts_b_display,
        match_a_display=match_a_display,
        match_b_display=match_b_display,
        qc=qc,
        setup_a=setup_a,
        setup_b=setup_b,
    )

    view_match_pair_y_chunk_sliders(
        vol_a_crop=vol_a_crop,
        vol_b_in_a_crop=vol_b_in_a_crop,
        match_a_display=match_a_display,
        match_b_display=match_b_display,
        qc=qc,
        setup_a=setup_a,
        setup_b=setup_b,
        n_y_chunks=ZOOM_Y_CHUNKS,
    )


if __name__ == "__main__":
    main()
