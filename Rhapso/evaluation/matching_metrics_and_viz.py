#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from typing import Optional, Tuple
from urllib.parse import urlparse

import boto3
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import zarr
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider

# Params
# Use detection xml for rigid match metrics - rigid xml for affine match metrics - etc
XML_PATH = "s3://aind-scratch-data/sean.fite/exaSPIM_730904-test/9/rhapso-solver-rigid.xml" 
INTERESTPOINTS_BASE = "s3://aind-scratch-data/sean.fite/exaSPIM_730904-test/9/interestpoints.n5"

SCALE_LEVEL = "4"
TIMEPOINT = 0

# Apply all XML transforms in XML order.

TARGET_PAIR: Optional[Tuple[int, int]] = None
# TARGET_PAIR = (5, 9)

OVERLAP_PAD_XY = 800

POINT_Z_RADIUS_SCALED = 1.5
LINK_Z_RADIUS_SCALED = 2.0

DISPLAY_NORM_MODE = "global"
P_LO = 1
P_HI = 99

IMAGE_ASPECT = "equal"

GOOD_RESIDUAL_PX = 2.0
OK_RESIDUAL_PX = 4.0
BAD_RESIDUAL_PX = 8.0

INLIER_RESIDUAL_PX = 4.0
GRID_BINS = (4, 8, 8)

# XML / image IO

def load_xml_root(xml_path: str) -> ET.Element:
    if xml_path.startswith("s3://"):
        parsed = urlparse(xml_path)
        obj = boto3.client("s3").get_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
        )
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

            # Later XML transforms are applied on top of earlier transforms.
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

    return sorted(records, key=lambda r: r['setup'])

def open_ome_zarr_level(zarr_path: str, scale_level: str):
    if zarr_path.startswith("s3://"):
        s3 = s3fs.S3FileSystem(anon=False)
        store = s3fs.S3Map(root=zarr_path.rstrip("/"), s3=s3, check=False)
        return da.from_zarr(store, component=scale_level)

    return da.from_zarr(zarr_path.rstrip("/"), component=scale_level)

# N5 / matches

def open_n5_root(n5_prefix: str):
    if n5_prefix.startswith("s3://"):
        s3 = s3fs.S3FileSystem(anon=False)
        store = s3fs.S3Map(root=n5_prefix.rstrip("/"), s3=s3, check=False)
        return zarr.open(store, mode="r")

    store = zarr.N5Store(n5_prefix.rstrip("/"))
    return zarr.open(store, mode="r")

def ip_prefix(tp_id: int, setup_id: int) -> str:
    return f"tpId_{tp_id}_viewSetupId_{setup_id}/beads/interestpoints"

def corr_prefix(tp_id: int, setup_id: int) -> str:
    return f"tpId_{tp_id}_viewSetupId_{setup_id}/beads/correspondences"

def require_array(root, rel_path: str):
    if rel_path not in root:
        raise KeyError(f"Missing array: {rel_path}")
    return root[rel_path]

def read_interest_points_from_root(root, setup: int, timepoint: int):
    loc_path = f"{ip_prefix(timepoint, setup)}/loc"
    arr = require_array(root, loc_path)

    pts = np.asarray(arr[:], dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError(f"Expected loc shape Nx3 at {loc_path}, got {pts.shape}")

    return pts

def parse_id_map_entry(key: str):
    parts = str(key).split(",", 2)
    if len(parts) < 2:
        raise ValueError(f"Could not parse idMap key: {key}")

    dst_tp = int(parts[0])
    dst_setup = int(parts[1])
    label = parts[2] if len(parts) > 2 else ""

    return dst_tp, dst_setup, label

def debug_correspondence_group(root, tp_id: int, setup_id: int):
    cpath = corr_prefix(tp_id, setup_id)

    print(f"\nCorrespondence debug for setup {setup_id}")
    print(f"  group path: {cpath}")
    print(f"  exists: {cpath in root}")

    if cpath not in root:
        return

    cg = root[cpath]
    print(f"  children: {list(cg.keys())}")
    print(f"  attrs keys: {list(dict(cg.attrs).keys())}")

    attrs = dict(cg.attrs)
    if "idMap" in attrs:
        print(f"  idMap entries: {len(attrs['idMap'])}")
        print(f"  idMap sample: {list(attrs['idMap'].items())[:5]}")

    if "data" in cg:
        data = cg["data"]
        print(f"  data shape: {data.shape}")
        print(f"  data dtype: {data.dtype}")
        if data.shape[0] > 0:
            print(f"  first rows:\n{np.asarray(data[: min(10, data.shape[0])])}")

def get_pair_matches_one_direction_from_root(
    root,
    tp_id: int,
    setup_a: int,
    setup_b: int,
):
    a_corr_path = corr_prefix(tp_id, setup_a)

    if a_corr_path not in root:
        print(f"No correspondences group for setup {setup_a}: {a_corr_path}")
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            [],
        )

    corr_group = root[a_corr_path]

    if "data" not in corr_group:
        print(f"No correspondences/data under {a_corr_path}")
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            [],
        )

    attrs = dict(corr_group.attrs)
    if "idMap" not in attrs:
        debug_correspondence_group(root, tp_id, setup_a)
        raise KeyError(f"Missing idMap at {a_corr_path}")

    data = np.asarray(corr_group["data"][:])

    if data.size == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            [],
        )

    if data.ndim != 2:
        debug_correspondence_group(root, tp_id, setup_a)
        raise RuntimeError(f"Expected correspondences/data to be 2D, got {data.shape}")

    # attrs["idMap"] maps "tp,setup,label" -> integer group id.
    # Reverse to group id -> "tp,setup,label".
    id_map = {int(v): str(k) for k, v in attrs["idMap"].items()}

    src_loc = read_interest_points_from_root(root, setup_a, tp_id)

    chunks_a = []
    chunks_b = []
    metadata = []

    print(f"\nReading matches {setup_a} -> {setup_b}")
    print(f"  corr data shape: {data.shape}")
    print(f"  idMap entries: {len(id_map)}")

    if data.shape[1] >= 3:
        group_ids = np.unique(data[:, 2].astype(np.int64))

        for group_id in group_ids:
            group_id = int(group_id)

            if group_id not in id_map:
                print(f"WARNING: group id {group_id} not found in idMap")
                continue

            dst_tp, dst_setup, label = parse_id_map_entry(id_map[group_id])

            if dst_tp != tp_id or dst_setup != setup_b:
                continue

            dst_loc = read_interest_points_from_root(root, dst_setup, dst_tp)

            rows = data[data[:, 2].astype(np.int64) == group_id]

            src_idx = rows[:, 0].astype(np.int64)
            dst_idx = rows[:, 1].astype(np.int64)

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

            a_pts = np.asarray(src_loc[src_idx], dtype=np.float32)
            b_pts = np.asarray(dst_loc[dst_idx], dtype=np.float32)

            chunks_a.append(a_pts)
            chunks_b.append(b_pts)

            metadata.extend(
                {
                    "src_setup": setup_a,
                    "src_index": int(si),
                    "dst_setup": setup_b,
                    "dst_index": int(di),
                    "group_id": group_id,
                    "label": label,
                }
                for si, di in zip(src_idx, dst_idx)
            )

    elif data.shape[1] == 2:
        matching_groups = []

        for group_id, key in id_map.items():
            dst_tp, dst_setup, label = parse_id_map_entry(key)
            if dst_tp == tp_id and dst_setup == setup_b:
                matching_groups.append((group_id, dst_tp, dst_setup, label))

        if len(matching_groups) != 1:
            debug_correspondence_group(root, tp_id, setup_a)
            raise RuntimeError(
                f"Correspondence data is Nx2, but idMap does not identify exactly "
                f"one target group for setup {setup_a}->{setup_b}. "
                f"matching_groups={matching_groups}"
            )

        group_id, dst_tp, dst_setup, label = matching_groups[0]
        dst_loc = read_interest_points_from_root(root, dst_setup, dst_tp)

        src_idx = data[:, 0].astype(np.int64)
        dst_idx = data[:, 1].astype(np.int64)

        if len(src_idx):
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

        metadata.extend(
            {
                "src_setup": setup_a,
                "src_index": int(si),
                "dst_setup": setup_b,
                "dst_index": int(di),
                "group_id": int(group_id),
                "label": label,
            }
            for si, di in zip(src_idx, dst_idx)
        )

    else:
        debug_correspondence_group(root, tp_id, setup_a)
        raise RuntimeError(f"Unsupported correspondences/data shape: {data.shape}")

    if not chunks_a:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            [],
        )

    return (
        np.vstack(chunks_a).astype(np.float32),
        np.vstack(chunks_b).astype(np.float32),
        metadata,
    )

def get_pair_matches_bidirectional_from_root(
    root,
    tp_id: int,
    setup_a: int,
    setup_b: int,
):
    a1, b1, meta1 = get_pair_matches_one_direction_from_root(
        root=root,
        tp_id=tp_id,
        setup_a=setup_a,
        setup_b=setup_b,
    )

    b2, a2, meta2_raw = get_pair_matches_one_direction_from_root(
        root=root,
        tp_id=tp_id,
        setup_a=setup_b,
        setup_b=setup_a,
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
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            [],
        )

    return (
        np.vstack(chunks_a).astype(np.float32),
        np.vstack(chunks_b).astype(np.float32),
        metadata,
    )

# Geometry

def open_tile_volume_zyx(rec: dict):
    arr = open_ome_zarr_level(rec['full_path'], SCALE_LEVEL)

    if arr.ndim == 5:
        return arr[0, 0, :, :, :].astype(np.float32)

    if arr.ndim == 3:
        return arr.astype(np.float32)

    raise RuntimeError(f"Unexpected array shape for {rec['full_path']}: {arr.shape}")

def infer_scale_info(rec: dict, vol_zyx: da.Array) -> dict:
    z_scaled, y_scaled, x_scaled = vol_zyx.shape

    return {
        "scale_x": float(rec['size_x']) / float(x_scaled),
        "scale_y": float(rec['size_y']) / float(y_scaled),
        "scale_z": float(rec['size_z']) / float(z_scaled),
    }

def transformed_xy_box_fullres(rec: dict):
    corners = np.asarray(
        [
            [0, 0, 0],
            [rec['size_x'], 0, 0],
            [0, rec['size_y'], 0],
            [rec['size_x'], rec['size_y'], 0],
        ],
        dtype=np.float32,
    )

    out = apply_affine_xyz(corners, rec['transform'])

    return (
        float(np.min(out[:, 0])),
        float(np.max(out[:, 0])),
        float(np.min(out[:, 1])),
        float(np.max(out[:, 1])),
    )

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
                overlaps.append((rec_a['setup'], rec_b['setup'], overlap))

    return overlaps

def nominal_overlap_to_local_padded_crop(rec: dict, overlap_box_fullres, pad_xy: int):
    ox0, ox1, oy0, oy1 = overlap_box_fullres

    inv = np.linalg.inv(rec['transform'])

    aligned_corners = np.asarray(
        [
            [ox0, oy0, 0],
            [ox1, oy0, 0],
            [ox0, oy1, 0],
            [ox1, oy1, 0],
        ],
        dtype=np.float32,
    )

    local = apply_affine_xyz(aligned_corners, inv)

    lx0 = int(np.floor(np.min(local[:, 0]) - pad_xy))
    lx1 = int(np.ceil(np.max(local[:, 0]) + pad_xy))
    ly0 = int(np.floor(np.min(local[:, 1]) - pad_xy))
    ly1 = int(np.ceil(np.max(local[:, 1]) + pad_xy))

    lx0 = max(0, lx0)
    ly0 = max(0, ly0)
    lx1 = min(rec['size_x'], lx1)
    ly1 = min(rec['size_y'], ly1)

    if lx1 <= lx0 or ly1 <= ly0:
        return None

    return {"x0": lx0, "x1": lx1, "y0": ly0, "y1": ly1}

def fullres_crop_to_scaled(crop: dict, scale: dict, vol_zyx: da.Array):
    _, y_max, x_max = vol_zyx.shape

    sx0 = int(np.floor(crop['x0'] / scale['scale_x']))
    sx1 = int(np.ceil(crop['x1'] / scale['scale_x']))
    sy0 = int(np.floor(crop['y0'] / scale['scale_y']))
    sy1 = int(np.ceil(crop['y1'] / scale['scale_y']))

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
        (pts_xyz[:, 0] >= crop['x0']) &
        (pts_xyz[:, 0] < crop['x1']) &
        (pts_xyz[:, 1] >= crop['y0']) &
        (pts_xyz[:, 1] < crop['y1'])
    )

    return pts_xyz[mask]

def filter_paired_points_in_xy_crops(
    pts_a_xyz: np.ndarray,
    pts_b_xyz: np.ndarray,
    crop_a: dict,
    crop_b: dict,
):
    if len(pts_a_xyz) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    mask_a = (
        (pts_a_xyz[:, 0] >= crop_a['x0']) &
        (pts_a_xyz[:, 0] < crop_a['x1']) &
        (pts_a_xyz[:, 1] >= crop_a['y0']) &
        (pts_a_xyz[:, 1] < crop_a['y1'])
    )

    mask_b = (
        (pts_b_xyz[:, 0] >= crop_b['x0']) &
        (pts_b_xyz[:, 0] < crop_b['x1']) &
        (pts_b_xyz[:, 1] >= crop_b['y0']) &
        (pts_b_xyz[:, 1] < crop_b['y1'])
    )

    mask = mask_a & mask_b

    return (
        pts_a_xyz[mask].astype(np.float32),
        pts_b_xyz[mask].astype(np.float32),
    )

def local_fullres_to_global_scaled(
    pts_full_xyz: np.ndarray,
    rec: dict,
    scale: dict,
):
    if pts_full_xyz is None or len(pts_full_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    transformed_full = apply_affine_xyz(pts_full_xyz, rec['transform'])

    out = np.empty_like(transformed_full, dtype=np.float32)
    out[:, 0] = transformed_full[:, 0] / scale['scale_x']
    out[:, 1] = transformed_full[:, 1] / scale['scale_y']
    out[:, 2] = transformed_full[:, 2] / scale['scale_z']

    return out

def global_scaled_to_a_crop_display_unfiltered(
    pts_global_xyz: np.ndarray,
    rec_a: dict,
    crop_a_full: dict,
    scale_a: dict,
):
    if pts_global_xyz is None or len(pts_global_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    local_origin = np.asarray(
        [[crop_a_full['x0'], crop_a_full['y0'], 0.0]],
        dtype=np.float32,
    )
    transformed_origin_full = apply_affine_xyz(local_origin, rec_a['transform'])

    origin_x = transformed_origin_full[0, 0] / scale_a['scale_x']
    origin_y = transformed_origin_full[0, 1] / scale_a['scale_y']
    origin_z = transformed_origin_full[0, 2] / scale_a['scale_z']

    pts = pts_global_xyz.astype(np.float32).copy()
    pts[:, 0] -= origin_x
    pts[:, 1] -= origin_y
    pts[:, 2] -= origin_z

    return pts.astype(np.float32)

def display_bounds_mask(pts_display_xyz: np.ndarray, shape_zyx):
    if len(pts_display_xyz) == 0:
        return np.zeros((0,), dtype=bool)

    Z, Y, X = shape_zyx

    return (
        (pts_display_xyz[:, 0] >= 0) & (pts_display_xyz[:, 0] < X) &
        (pts_display_xyz[:, 1] >= 0) & (pts_display_xyz[:, 1] < Y) &
        (pts_display_xyz[:, 2] >= 0) & (pts_display_xyz[:, 2] < Z)
    )

def global_scaled_to_a_crop_display_filtered(
    pts_global_xyz: np.ndarray,
    rec_a: dict,
    crop_a_full: dict,
    scale_a: dict,
    image_a_crop_shape,
):
    pts = global_scaled_to_a_crop_display_unfiltered(
        pts_global_xyz,
        rec_a,
        crop_a_full,
        scale_a,
    )
    mask = display_bounds_mask(pts, image_a_crop_shape)
    return pts[mask].astype(np.float32)

# Display

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

def normalize_for_display(slice2d: np.ndarray, global_lo: float, global_hi: float):
    if DISPLAY_NORM_MODE == "global":
        return normalize_slice(slice2d, global_lo, global_hi)

    if DISPLAY_NORM_MODE != "per_slice":
        raise ValueError("DISPLAY_NORM_MODE must be 'global' or 'per_slice'")

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

def visible_points_for_slice(pts_xyz: np.ndarray, z: int, z_radius: float):
    if len(pts_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    dz = np.abs(pts_xyz[:, 2] - float(z))
    return pts_xyz[dz <= z_radius]

# Pair prep

def prepare_pair_data(rec: dict, overlap_box_fullres, ip_root):
    print(f"Opening setup {rec['setup']} image: {rec['full_path']}")

    vol = open_tile_volume_zyx(rec)
    scale = infer_scale_info(rec, vol)

    print(
        f"  setup {rec['setup']} scaled image shape zyx={vol.shape}, "
        f"scale xyz=({scale['scale_x']:.3f}, {scale['scale_y']:.3f}, {scale['scale_z']:.3f})"
    )

    crop_full = nominal_overlap_to_local_padded_crop(
        rec,
        overlap_box_fullres,
        OVERLAP_PAD_XY,
    )

    if crop_full is None:
        raise RuntimeError(f"Could not make overlap crop for setup {rec['setup']}")

    crop_scaled = fullres_crop_to_scaled(crop_full, scale, vol)

    if crop_scaled is None:
        raise RuntimeError(f"Could not convert crop to scaled coords for setup {rec['setup']}")

    vol_crop = vol[
        :,
        crop_scaled['y0']:crop_scaled['y1'],
        crop_scaled['x0']:crop_scaled['x1'],
    ].compute().astype(np.float32)

    print(
        f"  setup {rec['setup']} crop full x[{crop_full['x0']}:{crop_full['x1']}] "
        f"y[{crop_full['y0']}:{crop_full['y1']}]"
    )
    print(f"  setup {rec['setup']} crop scaled zyx={vol_crop.shape}")

    print(f"Loading setup {rec['setup']} interest points")
    pts_all = read_interest_points_from_root(ip_root, rec['setup'], TIMEPOINT)
    print(f"  setup {rec['setup']} total points: {len(pts_all):,}")

    pts_xy = filter_points_in_xy_crop(pts_all, crop_full)
    print(f"  setup {rec['setup']} points in padded XY overlap crop: {len(pts_xy):,}")

    pts_global_scaled = local_fullres_to_global_scaled(pts_xy, rec, scale)

    return {
        "vol_crop": vol_crop,
        "scale": scale,
        "crop_full": crop_full,
        "crop_scaled": crop_scaled,
        "pts_local_full": pts_xy,
        "pts_global_scaled": pts_global_scaled,
    }

# QC

def safe_div(n, d):
    return float(n) / float(d) if d else np.nan

def grid_stats(pts_xyz: np.ndarray, shape_zyx, bins=GRID_BINS):
    if len(pts_xyz) == 0:
        return {
            "occupancy": np.nan,
            "max_cell_fraction": np.nan,
        }

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

def compute_match_qc(
    match_a_display: np.ndarray,
    match_b_display: np.ndarray,
    image_shape_zyx,
    scale: dict,
):
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

    good_count = qc["n_inliers"] >= 50
    good_ratio = qc["inlier_ratio"] >= 35.0
    good_error = (
        qc["median_consistency_residual_scaled"] <= 3.0 and
        qc["p90_consistency_residual_scaled"] <= 8.0
    )
    good_spread = (
        np.isfinite(qc["grid_occupancy"]) and
        qc["grid_occupancy"] >= 0.20 and
        np.isfinite(qc["max_cell_fraction"]) and
        qc["max_cell_fraction"] <= 0.25
    )
    good_unique = (
        np.isfinite(qc["duplicate_a_fraction"]) and qc["duplicate_a_fraction"] <= 0.05 and
        np.isfinite(qc["duplicate_b_fraction"]) and qc["duplicate_b_fraction"] <= 0.05
    )

    if all([good_count, good_ratio, good_error, good_spread]):
        qc["verdict"] = "GOOD" if good_unique else "GOOD / DEDUPE CHECK"
    elif good_count and good_error and (good_ratio or good_spread):
        qc["verdict"] = "USABLE / CHECK"
    else:
        qc["verdict"] = "CHECK"

    flags = []
    if not good_count:
        flags.append("low inlier count")
    if not good_ratio:
        flags.append("low inlier ratio")
    if not good_error:
        flags.append("high match-consistency residuals")
    if not good_spread:
        flags.append("weak spatial spread")
    if not good_unique:
        flags.append("duplicate/reused match points")

    qc["flags"] = flags

    return qc

def print_match_qc(qc, scale: dict):
    print("\n" + "=" * 72)
    print("ACTUAL MATCH QC AFTER FULL XML TRANSFORM STACK")
    print("=" * 72)

    if qc is None:
        print("No matches found in display crop.")
        print("VERDICT: CHECK")
        print("=" * 72 + "\n")
        return

    print(f"Matches visible in crop:     {qc['n_matches']:,}")
    print(f"Inliers <= {INLIER_RESIDUAL_PX:g}px consistency: {qc['n_inliers']:,}")
    print(f"Inlier ratio:                {qc['inlier_ratio']:.2f}%")

    print("\nRaw A/B link length after full XML transform stack:")
    print(f"  median scaled px:           {qc['raw_median_link_scaled']:.2f}")
    print(f"  p90 scaled px:              {qc['raw_p90_link_scaled']:.2f}")
    print(f"  p95 scaled px:              {qc['raw_p95_link_scaled']:.2f}")
    print(f"  max scaled px:              {qc['raw_max_link_scaled']:.2f}")
    print(f"  median full-res approx:     {qc['raw_median_link_scaled'] * scale['scale_x']:.1f} px")
    print(f"  p90 full-res approx:        {qc['raw_p90_link_scaled'] * scale['scale_x']:.1f} px")

    print("\nMatch-consistency residual after subtracting median match vector:")
    print(f"  median scaled px:           {qc['median_consistency_residual_scaled']:.2f}")
    print(f"  p90 scaled px:              {qc['p90_consistency_residual_scaled']:.2f}")
    print(f"  p95 scaled px:              {qc['p95_consistency_residual_scaled']:.2f}")
    print(f"  max scaled px:              {qc['max_consistency_residual_scaled']:.2f}")
    print(f"  median full-res approx:     {qc['median_consistency_residual_scaled'] * scale['scale_x']:.1f} px")
    print(f"  p90 full-res approx:        {qc['p90_consistency_residual_scaled'] * scale['scale_x']:.1f} px")

    print("\nConsistency thresholds:")
    print(f"  within 2px:                 {qc['within_2px']:.2f}%")
    print(f"  within 4px:                 {qc['within_4px']:.2f}%")
    print(f"  within 8px:                 {qc['within_8px']:.2f}%")

    print("\nMedian residual vector A→B after full XML transform stack:")
    print(
        f"  scaled xyz:                 "
        f"dx={qc['median_dx']:.2f}, "
        f"dy={qc['median_dy']:.2f}, "
        f"dz={qc['median_dz']:.2f}"
    )
    print(
        f"  full-res approx xyz:        "
        f"dx={qc['median_dx'] * scale['scale_x']:.1f}, "
        f"dy={qc['median_dy'] * scale['scale_y']:.1f}, "
        f"dz={qc['median_dz'] * scale['scale_z']:.1f}"
    )
    print(
        f"  MAD scaled xyz:             "
        f"dx={qc['mad_dx']:.2f}, "
        f"dy={qc['mad_dy']:.2f}, "
        f"dz={qc['mad_dz']:.2f}"
    )

    print("\nSpatial spread of consistency inliers:")
    print(f"  inlier grid occupancy:      {qc['grid_occupancy']:.3f}")
    print(f"  max grid-cell fraction:     {qc['max_cell_fraction']:.3f}")

    print("\nUniqueness:")
    print(f"  duplicate A fraction:       {qc['duplicate_a_fraction']:.3f}")
    print(f"  duplicate B fraction:       {qc['duplicate_b_fraction']:.3f}")

    print("")
    print(f"VERDICT: {qc['verdict']}")

    if qc["flags"]:
        print("Reason flags:")
        for flag in qc["flags"]:
            print(f"  - {flag}")

    print("=" * 72 + "\n")

# Viewer

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

def visible_matches_for_slice(
    match_a: np.ndarray,
    match_b: np.ndarray,
    residuals: np.ndarray,
    z: int,
    z_radius: float,
):
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

    segments = np.stack(
        [
            a[:, :2],
            b[:, :2],
        ],
        axis=1,
    ).astype(np.float32)

    link_colors = [color_for_residual(float(x)) for x in r]
    ring_segments = make_pair_ring_segments(a[:, :2], b[:, :2])

    return a, b, segments, ring_segments, link_colors, r

def view_match_pair_slider(
    vol_a_crop: np.ndarray,
    pts_a_display: np.ndarray,
    pts_b_display: np.ndarray,
    match_a_display: np.ndarray,
    match_b_display: np.ndarray,
    qc,
    setup_a: int,
    setup_b: int,
):
    Z, Y, X = vol_a_crop.shape

    if qc is not None:
        residuals = qc["consistency_residuals"]
    else:
        residuals = np.empty((0,), dtype=np.float32)

    global_lo, global_hi = compute_global_display_range(vol_a_crop)

    def norm(z):
        return normalize_for_display(vol_a_crop[z], global_lo, global_hi)

    z0 = Z // 2

    fig, ax = plt.subplots(figsize=(11, 9))
    plt.subplots_adjust(bottom=0.22, top=0.96, left=0.06, right=0.86)

    im = ax.imshow(
        norm(z0),
        cmap="gray",
        origin="upper",
        aspect=IMAGE_ASPECT,
    )

    a0, b0, seg0, rings0, colors0, r0 = visible_matches_for_slice(
        match_a_display,
        match_b_display,
        residuals,
        z0,
        LINK_Z_RADIUS_SCALED,
    )

    # Pair grouping ring. This is the key visibility cue when red/blue endpoints overlap.
    pair_rings = LineCollection(
        rings0,
        colors="cyan",
        linewidths=1.6,
        alpha=0.95,
        zorder=7,
    )
    ax.add_collection(pair_rings)

    # Link halo + residual-colored foreground line.
    link_halo = LineCollection(
        seg0,
        colors="white",
        linewidths=4.0,
        alpha=0.95,
        zorder=8,
    )
    ax.add_collection(link_halo)

    links = LineCollection(
        seg0,
        colors=colors0,
        linewidths=2.0,
        alpha=1.0,
        zorder=9,
    )
    ax.add_collection(links)

    sc_a_match = ax.scatter(
        a0[:, 0] if len(a0) else [],
        a0[:, 1] if len(a0) else [],
        s=36,
        c="red",
        marker="o",
        alpha=0.98,
        edgecolors="white",
        linewidths=0.8,
        label=f"A setup {setup_a}",
        zorder=10,
    )

    sc_b_match = ax.scatter(
        b0[:, 0] if len(b0) else [],
        b0[:, 1] if len(b0) else [],
        s=36,
        c="blue",
        marker="o",
        alpha=0.98,
        edgecolors="white",
        linewidths=0.8,
        label=f"B setup {setup_b}",
        zorder=10,
    )

    # Legend outside the image so it does not cover data.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        frameon=True,
    )

    ax.set_title(
        f"Match QC after full XML transform stack | red=A setup {setup_a} | blue=B setup {setup_b}",
        fontsize=11,
        pad=8,
    )

    # Info text outside the image, directly above the slider.
    info_text = fig.text(
        0.15,
        0.135,
        "",
        ha="left",
        va="center",
        fontsize=10,
        family="monospace",
    )

    slider_ax = fig.add_axes([0.15, 0.065, 0.70, 0.035])
    slider = Slider(slider_ax, "Z", 0, Z - 1, valinit=z0, valstep=1)

    def update(val):
        z = int(slider.val)

        im.set_data(norm(z))

        a_now, b_now, seg, rings, colors, r = visible_matches_for_slice(
            match_a_display,
            match_b_display,
            residuals,
            z,
            LINK_Z_RADIUS_SCALED,
        )

        sc_a_match.set_offsets(
            a_now[:, :2] if len(a_now) else np.empty((0, 2))
        )
        sc_b_match.set_offsets(
            b_now[:, :2] if len(b_now) else np.empty((0, 2))
        )

        pair_rings.set_segments(rings)
        link_halo.set_segments(seg)
        links.set_segments(seg)
        links.set_color(colors)

        if len(r):
            med = float(np.median(r))
            p90 = float(np.percentile(r, 90))
        else:
            med = np.nan
            p90 = np.nan

        verdict = qc["verdict"] if qc is not None else "NO MATCHES"

        info_text.set_text(
            f"z={z:4d} | visible matched pairs={len(seg):5,d} | "
            f"residual med={med:5.2f}px p90={p90:5.2f}px | {verdict}"
        )

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(z0)
    plt.show()

def dedupe_match_pairs(match_a_local, match_b_local, decimals=3):
    if len(match_a_local) == 0:
        return match_a_local, match_b_local

    if len(match_a_local) != len(match_b_local):
        raise RuntimeError(
            f"Cannot dedupe unpaired match arrays: "
            f"A={len(match_a_local)}, B={len(match_b_local)}"
        )

    key_arr = np.concatenate(
        [
            np.round(match_a_local, decimals=decimals),
            np.round(match_b_local, decimals=decimals),
        ],
        axis=1,
    )

    _, keep_idx = np.unique(key_arr, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)

    return (
        match_a_local[keep_idx].astype(np.float32),
        match_b_local[keep_idx].astype(np.float32),
    )

# Main

def main():
    print("Loading XML...")
    root = load_xml_root(XML_PATH)

    records = parse_zarr_tile_records(root)
    print(f"Found tile records: {len(records)}")

    if len(records) == 0:
        print("No tile records found.")
        return

    by_setup = {r['setup']: r for r in records}

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
        setup_a, setup_b = TARGET_PAIR

        if setup_a not in by_setup or setup_b not in by_setup:
            raise RuntimeError(f"TARGET_PAIR {TARGET_PAIR} not found in XML records")

        overlap_box = compute_pairwise_overlap_fullres(
            by_setup[setup_a],
            by_setup[setup_b],
        )

        if overlap_box is None:
            raise RuntimeError(f"TARGET_PAIR {TARGET_PAIR} does not overlap in XY")

        print(f"\nUsing TARGET_PAIR: ({setup_a}, {setup_b})")

    rec_a = by_setup[setup_a]
    rec_b = by_setup[setup_b]

    print("\nOpening interestpoints N5 root once...")
    ip_root = open_n5_root(INTERESTPOINTS_BASE)

    data_a = prepare_pair_data(rec_a, overlap_box, ip_root)
    data_b = prepare_pair_data(rec_b, overlap_box, ip_root)

    vol_a_crop = data_a["vol_crop"]

    # Convert all detected points into setup-A crop display frame.
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

    print(f"\nDisplay-frame all points:")
    print(f"  setup {setup_a}: {len(pts_a_display):,}")
    print(f"  setup {setup_b}: {len(pts_b_display):,}")

    match_a_local, match_b_local, match_meta = get_pair_matches_bidirectional_from_root(
        root=ip_root,
        tp_id=TIMEPOINT,
        setup_a=setup_a,
        setup_b=setup_b,
    )

    print(f"\nLoaded raw bidirectional matches for pair ({setup_a}, {setup_b}): {len(match_a_local):,}")
    print(f"Match metadata rows before dedupe: {len(match_meta):,}")

    match_a_local, match_b_local = dedupe_match_pairs(
        match_a_local,
        match_b_local,
        decimals=3,
    )

    print(f"Raw matches after A/B pair dedupe: {len(match_a_local):,}")

    match_a_local_crop, match_b_local_crop = filter_paired_points_in_xy_crops(
        match_a_local,
        match_b_local,
        data_a["crop_full"],
        data_b["crop_full"],
    )

    print(f"Matches after paired XY crop filter: {len(match_a_local_crop):,}")

    match_a_global_scaled = local_fullres_to_global_scaled(
        match_a_local_crop,
        rec_a,
        data_a["scale"],
    )

    match_b_global_scaled = local_fullres_to_global_scaled(
        match_b_local_crop,
        rec_b,
        data_b["scale"],
    )

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

    mask_a = display_bounds_mask(match_a_display_all, vol_a_crop.shape)
    mask_b = display_bounds_mask(match_b_display_all, vol_a_crop.shape)
    pair_display_mask = mask_a & mask_b

    match_a_display = match_a_display_all[pair_display_mask].astype(np.float32)
    match_b_display = match_b_display_all[pair_display_mask].astype(np.float32)

    print(f"Matches visible in setup-A display frame: {len(match_a_display):,}")

    qc = compute_match_qc(
        match_a_display,
        match_b_display,
        vol_a_crop.shape,
        data_a["scale"],
    )

    print_match_qc(qc, data_a["scale"])

    view_match_pair_slider(
        vol_a_crop=vol_a_crop,
        pts_a_display=pts_a_display,
        pts_b_display=pts_b_display,
        match_a_display=match_a_display,
        match_b_display=match_b_display,
        qc=qc,
        setup_a=setup_a,
        setup_b=setup_b,
    )

if __name__ == "__main__":
    main()
