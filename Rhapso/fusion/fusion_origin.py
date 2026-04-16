from __future__ import annotations
import os
import logging
import ray
import numpy as np
import s3fs
import tensorstore as ts
from itertools import chain
from collections import OrderedDict
import time
import zarr
import Rhapso.fusion.blend as blend
import Rhapso.fusion.cloud_queue as cq
import Rhapso.fusion.geometry as geometry
import Rhapso.fusion.input_output as input_output
import Rhapso.fusion.fusion_utils as utils

# S3_READ = None
# S3_WRITE = None

# def _get_s3_read():
#     global S3_READ
#     if S3_READ is None:
#         import s3fs
#         S3_READ = s3fs.S3FileSystem(anon=True)
#     return S3_READ

# def _get_s3_write():
#     global S3_WRITE
#     if S3_WRITE is None:
#         import s3fs
#         S3_WRITE = s3fs.S3FileSystem(anon=False)
#     return S3_WRITE

_WORKER_READY = False

_S3_READ = None
_S3_WRITE = None

_OUT_ARR = None
_OUT_KEY = None  # (write_root, write_ds)

_IN_LRU: "OrderedDict[str, zarr.core.Array]" = OrderedDict()
IN_LRU_MAX = 32  # short-lived cache size; try 16/32/64

PRINT_EVERY_CELLS = 1000

_CACHE_LOOKUPS = 0
_CACHE_HITS = 0
_CACHE_MISSES = 0
_CACHE_EVICTS = 0

_CELL_COUNT = 0

_SUM_READ = 0.0
_SUM_FIELD = 0.0
_SUM_INTERP = 0.0
_SUM_BLEND = 0.0
_SUM_WRITE = 0.0
_SUM_TOTAL = 0.0

_MIN_TOTAL = float("inf")
_MAX_TOTAL = 0.0

def _worker_init(write_root: str, write_ds: str) -> None:
    """Initialize S3 clients + output zarr handle once per worker process."""
    global _WORKER_READY, _S3_READ, _S3_WRITE, _OUT_ARR, _OUT_KEY

    if _S3_READ is None:
        _S3_READ = s3fs.S3FileSystem(anon=True)
    if _S3_WRITE is None:
        _S3_WRITE = s3fs.S3FileSystem(anon=False)

    key = (write_root, write_ds)
    if _OUT_ARR is None or _OUT_KEY != key:
        out_store = s3fs.S3Map(root=write_root, s3=_S3_WRITE, check=False)
        # NOTE: you used zarr.open(store=..., mode="a")[write_ds] in your actor
        _OUT_ARR = zarr.open(store=out_store, mode="a")[write_ds]
        _OUT_KEY = key

    _WORKER_READY = True

def _maybe_print_worker_stats():
    global _CELL_COUNT
    global _CACHE_LOOKUPS, _CACHE_HITS, _CACHE_MISSES, _CACHE_EVICTS
    global _SUM_READ, _SUM_FIELD, _SUM_INTERP, _SUM_BLEND, _SUM_WRITE, _SUM_TOTAL
    global _MIN_TOTAL, _MAX_TOTAL

    if _CELL_COUNT == 0 or (_CELL_COUNT % PRINT_EVERY_CELLS) != 0:
        return

    # Cache hit rate
    denom = max(1, _CACHE_HITS + _CACHE_MISSES)
    hit_rate = 100.0 * (_CACHE_HITS / denom)

    # Averages
    n = float(_CELL_COUNT)
    avg_read = _SUM_READ / n
    avg_field = _SUM_FIELD / n
    avg_interp = _SUM_INTERP / n
    avg_blend = _SUM_BLEND / n
    avg_write = _SUM_WRITE / n
    avg_total = _SUM_TOTAL / n

    print(
        f"[worker-stats] pid={os.getpid()} "
        f"cells={_CELL_COUNT} "
        f"cache_hit%={hit_rate:.2f} hits={_CACHE_HITS} misses={_CACHE_MISSES} evicts={_CACHE_EVICTS} lru={len(_IN_LRU)}/{IN_LRU_MAX} "
        f"avg_s(read={avg_read:.3f} field={avg_field:.3f} interp={avg_interp:.3f} blend={avg_blend:.3f} write={avg_write:.3f} total={avg_total:.3f}) "
        f"min_total={_MIN_TOTAL:.3f} max_total={_MAX_TOTAL:.3f}",
        flush=True,
    )

def _get_in_arr(src_path: str):
    """LRU cache for *input zarr array handles* (not image data)."""
    global _IN_LRU, _S3_READ
    global _CACHE_LOOKUPS, _CACHE_HITS, _CACHE_MISSES, _CACHE_EVICTS

    _CACHE_LOOKUPS += 1

    zarr_arr = _IN_LRU.get(src_path)
    if zarr_arr is not None:
        _CACHE_HITS += 1
        _IN_LRU.move_to_end(src_path)
        return zarr_arr

    _CACHE_MISSES += 1
    store = s3fs.S3Map(root=src_path, s3=_S3_READ, check=False)
    zarr_arr = zarr.open(store=store, mode="r")

    _IN_LRU[src_path] = zarr_arr
    _IN_LRU.move_to_end(src_path)

    while len(_IN_LRU) > IN_LRU_MAX:
        _IN_LRU.popitem(last=False)
        _CACHE_EVICTS += 1

    return zarr_arr

def cpu_fusion_cached(
    tile_transforms: dict[int, list[geometry.Transform]],
    tile_sizes_zyx: dict[int, tuple[int, int, int]],
    output_volume_origin: tuple[float, float, float],
    blend_module: blend.BlendingModule,
    cell_aabb: geometry.AABB,
    src_ids: list[int],
    tile_paths,
):
    global _OUT_ARR
    global _CELL_COUNT
    global _SUM_READ, _SUM_FIELD, _SUM_INTERP, _SUM_BLEND, _SUM_WRITE, _SUM_TOTAL
    global _MIN_TOTAL, _MAX_TOTAL

    t_total0 = time.perf_counter()

    # Aggregate per-cell stage times across all src_ids
    cell_read = 0.0
    cell_field = 0.0
    cell_interp = 0.0

    overlap_contributions: list[np.ndarray] = []
    for t_id in src_ids:
        image_slice: tuple[slice, slice, slice, slice, slice] = utils.calculate_image_crop(
            cell_aabb,
            output_volume_origin,
            tile_transforms[t_id],
            tile_sizes_zyx[t_id],
            device="cpu",
        )

        src_path = tile_paths[t_id]
        zarr_arr = _get_in_arr(src_path)  # updates cache stats internally

        t0 = time.perf_counter()
        src_img = zarr_arr[image_slice]
        cell_read += (time.perf_counter() - t0)

        t0 = time.perf_counter()
        sample_field = utils.calculate_sample_field_np(
            cell_aabb,
            output_volume_origin,
            tile_transforms[t_id],
            tile_sizes_zyx[t_id],
        )
        cell_field += (time.perf_counter() - t0)

        t0 = time.perf_counter()
        contribution = utils.interpolate_np(src_img, sample_field)
        cell_interp += (time.perf_counter() - t0)

        overlap_contributions.append(contribution)

    t0 = time.perf_counter()
    blended_cell = blend_module.blend(
        overlap_contributions,
        kwargs={"chunk_tile_ids": src_ids, "cell_box": cell_aabb},
    )
    cell_blend = time.perf_counter() - t0

    output_slice = (
        slice(0, 1),
        slice(0, 1),
        slice(cell_aabb[0], cell_aabb[1]),
        slice(cell_aabb[2], cell_aabb[3]),
        slice(cell_aabb[4], cell_aabb[5]),
    )

    blended_cell = np.nan_to_num(blended_cell)
    blended_cell = np.clip(blended_cell, 0, 65535)
    output_chunk = np.ascontiguousarray(blended_cell.astype(np.uint16))

    t0 = time.perf_counter()
    _OUT_ARR[output_slice] = output_chunk
    cell_write = time.perf_counter() - t0

    cell_total = time.perf_counter() - t_total0

    # ---- update per-worker rolling stats ----
    _CELL_COUNT += 1
    _SUM_READ += cell_read
    _SUM_FIELD += cell_field
    _SUM_INTERP += cell_interp
    _SUM_BLEND += cell_blend
    _SUM_WRITE += cell_write
    _SUM_TOTAL += cell_total

    if cell_total < _MIN_TOTAL:
        _MIN_TOTAL = cell_total
    if cell_total > _MAX_TOTAL:
        _MAX_TOTAL = cell_total

    # Periodic print every PRINT_EVERY_CELLS
    _maybe_print_worker_stats()

# def cpu_fusion_cached(
#     tile_transforms: dict[int, list[geometry.Transform]],
#     tile_sizes_zyx: dict[int, tuple[int, int, int]],
#     output_volume_origin: tuple[float, float, float],
#     blend_module: blend.BlendingModule,
#     cell_aabb: geometry.AABB,
#     src_ids: list[int],
#     tile_paths,
# ):
#     global _OUT_ARR

#     z = int(cell_aabb[1] - cell_aabb[0])
#     y = int(cell_aabb[3] - cell_aabb[2])
#     x = int(cell_aabb[5] - cell_aabb[4])
#     vox = z * y * x
#     n_src = len(src_ids)

#     print(f"[cpu_fusion] pid={os.getpid()} START shape=({z},{y},{x}) vox={vox:,} n_src={n_src} ", flush=True)
#     t_total0 = time.perf_counter()

#     overlap_contributions: list[np.ndarray] = []
#     for t_id in src_ids:
#         image_slice: tuple[slice, slice, slice, slice, slice] = utils.calculate_image_crop(
#             cell_aabb,
#             output_volume_origin,
#             tile_transforms[t_id],
#             tile_sizes_zyx[t_id],
#             device="cpu",
#         )

#         src_path = tile_paths[t_id]

#         # -------- cache: input zarr handle LRU --------
#         zarr_arr = _get_in_arr(src_path)
#         # ---------------------------------------------

#         t_read0 = time.perf_counter()
#         src_img = zarr_arr[image_slice]
#         t_read = time.perf_counter() - t_read0

#         t_field0 = time.perf_counter()
#         sample_field = utils.calculate_sample_field_np(
#             cell_aabb,
#             output_volume_origin,
#             tile_transforms[t_id],
#             tile_sizes_zyx[t_id],
#         )
#         t_field = time.perf_counter() - t_field0

#         t_interp0 = time.perf_counter()
#         contribution = utils.interpolate_np(src_img, sample_field)
#         t_interp = time.perf_counter() - t_interp0

#         print(
#             f"[cpu_fusion] pid={os.getpid()} t_id={t_id} read={t_read:.3f}s field={t_field:.3f}s interp={t_interp:.3f}s",
#             flush=True,
#         )

#         overlap_contributions.append(contribution)

#     t_blend0 = time.perf_counter()
#     blended_cell = blend_module.blend(
#         overlap_contributions,
#         kwargs={"chunk_tile_ids": src_ids, "cell_box": cell_aabb},
#     )
#     t_blend = time.perf_counter() - t_blend0

#     output_slice = (
#         slice(0, 1),
#         slice(0, 1),
#         slice(cell_aabb[0], cell_aabb[1]),
#         slice(cell_aabb[2], cell_aabb[3]),
#         slice(cell_aabb[4], cell_aabb[5]),
#     )

#     blended_cell = np.nan_to_num(blended_cell)
#     blended_cell = np.clip(blended_cell, 0, 65535)
#     output_chunk = np.ascontiguousarray(blended_cell.astype(np.uint16))

#     t_write0 = time.perf_counter()
#     _OUT_ARR[output_slice] = output_chunk
#     t_write = time.perf_counter() - t_write0

#     total = time.perf_counter() - t_total0
#     print(f"[cpu_fusion] pid={os.getpid()} DONE blend={t_blend:.3f}s write={t_write:.3f}s total={total:.3f}s", flush=True)

@ray.remote(num_cpus=1)
def process_batch_task(
    jobs,
    tile_transforms,
    tile_sizes_zyx,
    output_volume_origin,
    blend_module,
    tile_paths,
    write_root,
    write_ds,
):
    # one-time init per worker process
    _worker_init(write_root, write_ds)

    for cell_aabb, src_ids in jobs:
        cpu_fusion_cached(
            tile_transforms=tile_transforms,
            tile_sizes_zyx=tile_sizes_zyx,
            output_volume_origin=output_volume_origin,
            blend_module=blend_module,
            cell_aabb=cell_aabb,
            src_ids=src_ids,
            tile_paths=tile_paths,
        )
    return None

# @ray.remote(num_cpus=1)
# class FusionWorker:
#     """
#     Long-lived Ray actor that keeps:
#       - s3_read client
#       - s3_write client
#       - output zarr array handle
#       - input zarr handles cache
#     alive across many batches.
#     """
#     def __init__(
#         self,
#         tile_transforms,
#         tile_sizes_zyx,
#         output_volume_origin,
#         blend_module,
#         tile_paths,
#         write_root,
#         write_ds,
#     ):
#         self.tile_transforms = tile_transforms
#         self.tile_sizes_zyx = tile_sizes_zyx
#         self.output_volume_origin = output_volume_origin
#         self.blend_module = blend_module
#         self.tile_paths = tile_paths

#         self.s3_read = _get_s3_read()
#         self.s3_write = _get_s3_write()

#         self.out_store = s3fs.S3Map(root=write_root, s3=self.s3_write, check=False)
#         self.arr = zarr.open(store=self.out_store, mode="a")[write_ds]

#         self.in_arr_cache = {}

#     def process_batch(self, jobs):
#         for cell_aabb, src_ids in jobs:
#             cpu_fusion_actor(
#                 self.tile_transforms,
#                 self.tile_sizes_zyx,
#                 self.output_volume_origin,
#                 self.blend_module,
#                 cell_aabb,
#                 src_ids,
#                 self.tile_paths,
#                 self.arr,
#                 self.s3_read,
#                 self.in_arr_cache,
#             )
#         return None


# def cpu_fusion_actor(
#     tile_transforms: dict[int, list[geometry.Transform]],
#     tile_sizes_zyx: dict[int, tuple[int, int, int]],
#     output_volume_origin: tuple[float, float, float],
#     blend_module: blend.BlendingModule,
#     cell_aabb: geometry.AABB,
#     src_ids: list[int],
#     tile_paths,
#     arr,          # actor-owned open output array
#     s3_read,      # actor-owned S3 read client
#     in_arr_cache, # actor-owned input zarr cache
# ):
#     z = int(cell_aabb[1] - cell_aabb[0])
#     y = int(cell_aabb[3] - cell_aabb[2])
#     x = int(cell_aabb[5] - cell_aabb[4])
#     vox = z * y * x
#     n_src = len(src_ids)

#     print(f"[cpu_fusion] pid={os.getpid()} START shape=({z},{y},{x}) vox={vox:,} n_src={n_src} ")
#     t_total0 = time.perf_counter()

#     overlap_contributions: list[np.ndarray] = []
#     for t_id in src_ids:
#         image_slice: tuple[slice, slice, slice, slice, slice] = utils.calculate_image_crop(
#             cell_aabb,
#             output_volume_origin,
#             tile_transforms[t_id],
#             tile_sizes_zyx[t_id],
#             device="cpu",
#         )

#         src_path = tile_paths[t_id]
#         zarr_arr = in_arr_cache.get(src_path)
#         if zarr_arr is None:
#             print(f"[cache] pid={os.getpid()} MISS t_id={t_id}", flush=True)
#             store = s3fs.S3Map(root=src_path, s3=s3_read, check=False)
#             zarr_arr = zarr.open(store=store, mode="r")
#             in_arr_cache[src_path] = zarr_arr
#         else:
#             print(f"[cache] pid={os.getpid()} HIT  t_id={t_id}", flush=True)

#         t_read0 = time.perf_counter()
#         src_img = zarr_arr[image_slice]
#         t_read = time.perf_counter() - t_read0

#         t_field0 = time.perf_counter()
#         sample_field = utils.calculate_sample_field_np(
#             cell_aabb,
#             output_volume_origin,
#             tile_transforms[t_id],
#             tile_sizes_zyx[t_id],
#         )
#         t_field = time.perf_counter() - t_field0

#         t_interp0 = time.perf_counter()
#         contribution = utils.interpolate_np(src_img, sample_field)
#         t_interp = time.perf_counter() - t_interp0

#         print(
#             f"[cpu_fusion] pid={os.getpid()} t_id={t_id} read={t_read:.3f}s field={t_field:.3f}s interp={t_interp:.3f}s",
#             flush=True,
#         )

#         overlap_contributions.append(contribution)

#     t_blend0 = time.perf_counter()
#     blended_cell = blend_module.blend(
#         overlap_contributions,
#         kwargs={"chunk_tile_ids": src_ids, "cell_box": cell_aabb},
#     )
#     t_blend = time.perf_counter() - t_blend0

#     output_slice = (
#         slice(0, 1),
#         slice(0, 1),
#         slice(cell_aabb[0], cell_aabb[1]),
#         slice(cell_aabb[2], cell_aabb[3]),
#         slice(cell_aabb[4], cell_aabb[5]),
#     )

#     blended_cell = np.nan_to_num(blended_cell)
#     blended_cell = np.clip(blended_cell, 0, 65535)
#     output_chunk = blended_cell.astype(np.uint16)

#     t_write0 = time.perf_counter()
#     arr[output_slice] = np.ascontiguousarray(output_chunk)
#     t_write = time.perf_counter() - t_write0

#     total = time.perf_counter() - t_total0
#     print(f"[cpu_fusion] pid={os.getpid()} DONE blend={t_blend:.3f}s write={t_write:.3f}s total={total:.3f}s", flush=True)

def initialize_fusion(
    dataset: input_output.Dataset,
    output_params: input_output.OutputParameters
) -> tuple[dict, dict, dict, dict, dict, tuple, tuple]:
    """
    Creates all core fusion data structures and key algorithm inputs.

    Inputs
    ------
    Dataset, OutputParameters application primitives.

    Returns
    -------
    tile_arrays: Dictionary of input tile arrays
    tile_transforms: Dictionary of (list of) registrations associated with each tile
    tile_sizes_zyx: Dictionary of tile sizes
    tile_aabbs: Dictionary of AABB of each transformed tile
    output_volume_size: Size of output volume
    output_volume_origin: Location of output volume
    """

    # Output Data Structures-- tile_arrays, tile_transforms
    tile_arrays, tile_paths = dataset.tile_volumes_tczyx
    tile_transforms: dict[int, list[geometry.Transform]] = (dataset.tile_transforms_zyx)

    # Output Data Structures-- tile_sizes_zyx, tile_aabbs
    tile_sizes_zyx: dict[int, tuple[int, int, int]] = {}
    tile_aabbs: dict[int, geometry.AABB] = {}
    tile_boundary_point_cloud_zyx = []

    for tile_id, tile_arr in tile_arrays.items():
        tile_sizes_zyx[tile_id] = zyx = tile_arr.shape[2:]

        # z_grid, y_grid, x_grid = torch.meshgrid(
        #     torch.Tensor([0, zyx[0]]),
        #     torch.Tensor([0, zyx[1]]),
        #     torch.Tensor([0, zyx[2]]),
        #     indexing='ij'
        # )
        # tile_boundary_pts = torch.stack([z_grid, y_grid, x_grid], dim=-1)

        zs = np.array([0.0, float(zyx[0])], dtype=np.float32)
        ys = np.array([0.0, float(zyx[1])], dtype=np.float32)
        xs = np.array([0.0, float(zyx[2])], dtype=np.float32)

        # 8 corners, shape (8,3) in zyx
        tile_boundary_pts = np.array([[z, y, x] for z in zs for y in ys for x in xs], dtype=np.float32)

        tfm_list = tile_transforms[tile_id]
        for i, tfm in enumerate(tfm_list):
            tile_boundary_pts = tfm.forward_np(
                tile_boundary_pts
            )

        tile_aabbs[tile_id] = geometry.aabb_3d_np(tile_boundary_pts)
        tile_boundary_point_cloud_zyx.append(tile_boundary_pts)
    
    # tile_boundary_point_cloud_zyx = torch.stack(
    #     tile_boundary_point_cloud_zyx, dim=0
    # )

    tile_boundary_point_cloud_zyx = np.stack(tile_boundary_point_cloud_zyx, axis=0)  # (n_tiles, 8, 3)


    # Output Data Structures-- OUTPUT_VOLUME_SIZE, OUTPUT_VOLUME_ORIGIN
    # Resolve Output Volume Dimensions and Absolute Position
    global_tile_boundaries = geometry.aabb_3d_np(tile_boundary_point_cloud_zyx)
    OUTPUT_VOLUME_SIZE = [
        int(global_tile_boundaries[1] - global_tile_boundaries[0]),
        int(global_tile_boundaries[3] - global_tile_boundaries[2]),
        int(global_tile_boundaries[5] - global_tile_boundaries[4]),
    ]

    # Rounding up the OUTPUT_VOLUME_SIZE to the nearest chunk
    # b/c zarr-python has occasional errors writing at the boundaries.
    # This ensures a multiple of chunksize without losing data.
    remainder_0 = OUTPUT_VOLUME_SIZE[0] % output_params.chunksize[2]
    remainder_1 = OUTPUT_VOLUME_SIZE[1] % output_params.chunksize[3]
    remainder_2 = OUTPUT_VOLUME_SIZE[2] % output_params.chunksize[4]
    if remainder_0 > 0:
        OUTPUT_VOLUME_SIZE[0] -= remainder_0
        OUTPUT_VOLUME_SIZE[0] += output_params.chunksize[2]
    if remainder_1 > 0:
        OUTPUT_VOLUME_SIZE[1] -= remainder_1
        OUTPUT_VOLUME_SIZE[1] += output_params.chunksize[3]
    if remainder_2 > 0:
        OUTPUT_VOLUME_SIZE[2] -= remainder_2
        OUTPUT_VOLUME_SIZE[2] += output_params.chunksize[4]
    OUTPUT_VOLUME_SIZE = tuple(OUTPUT_VOLUME_SIZE)

    OUTPUT_VOLUME_ORIGIN = (global_tile_boundaries[0],
                            global_tile_boundaries[2],
                            global_tile_boundaries[4])

    # Final update to output tile_aabbs.
    # Shift AABB's into OUTPUT_VOLUME.
    for tile_id, t_aabb in tile_aabbs.items():
        updated_aabb = (
            t_aabb[0] - OUTPUT_VOLUME_ORIGIN[0],
            t_aabb[1] - OUTPUT_VOLUME_ORIGIN[0],
            t_aabb[2] - OUTPUT_VOLUME_ORIGIN[1],
            t_aabb[3] - OUTPUT_VOLUME_ORIGIN[1],
            t_aabb[4] - OUTPUT_VOLUME_ORIGIN[2],
            t_aabb[5] - OUTPUT_VOLUME_ORIGIN[2],
        )
        tile_aabbs[tile_id] = updated_aabb

    return (
        tile_arrays,
        tile_transforms,
        tile_sizes_zyx,
        tile_aabbs,
        tile_paths,
        OUTPUT_VOLUME_SIZE,
        OUTPUT_VOLUME_ORIGIN,
    )


def initialize_output_volume_dask(
    output_params: input_output.OutputParameters,
    output_volume_size: tuple[int, int, int],
) -> zarr.core.Array:
    """
    Self-documentation of output store initialization.

    Inputs
    ------
    output_params: OutputParameters application instance.
    output_volume_size: output of initalize_data_structures(...)

    Returns
    -------
    Zarr thread-safe datastore initialized on OutputParameters.
    """

    # Local execution   
    out_group = zarr.open_group(output_params.path, mode="w")

    # Cloud execuion
    if output_params.path.startswith("s3"): 
        s3 = s3fs.S3FileSystem(
            config_kwargs={
                "max_pool_connections": 50,
                "s3": {
                    "multipart_threshold": 64
                    * 1024
                    * 1024,  # 64 MB, avoid multipart upload for small chunks
                    "max_concurrent_requests": 20,  # Increased from 10 -> 20.
                },
                "retries": {
                    "total_max_attempts": 100,
                    "mode": "adaptive",
                },
            }
        )
        store = s3fs.S3Map(root=output_params.path, s3=s3)
        out_group = zarr.open(store=store, mode="a")

    path = "0"
    chunksize = output_params.chunksize
    datatype = output_params.dtype
    dimension_separator = "/"
    compressor = output_params.compressor
    output_volume = out_group.create_dataset(
        path,
        shape=(
            1,
            1,
            output_volume_size[0],
            output_volume_size[1],
            output_volume_size[2],
        ),
        chunks=chunksize,
        dtype=datatype,
        compressor=compressor,
        dimension_separator=dimension_separator,
        overwrite=True,
        fill_value=0,
    )

    return output_volume


def initialize_output_volume_tensorstore(
    output_params: input_output.OutputParameters,
    output_volume_size: tuple[int, int, int],
):
    """
    The output is an async Tensorstore obj that you need
    to call .result() to perform a write.
    """
    parts = output_params.path.split("/")
    bucket = parts[2]
    path = "/".join(parts[3:])
    chunksize = list(output_params.chunksize)
    output_shape = [
        1,
        1,
        output_volume_size[0],
        output_volume_size[1],
        output_volume_size[2],
    ]

    return ts.open(
        {
            "driver": "zarr",
            "dtype": "uint16",
            "kvstore": {
                "driver": "s3",
                "bucket": bucket,
                "path": path,
            },
            "create": True,
            "open": True,
            "metadata": {
                "chunks": chunksize,
                "compressor": {
                    "blocksize": 0,
                    "clevel": 1,
                    "cname": "zstd",
                    "id": "blosc",
                    "shuffle": 1,
                },
                "dimension_separator": "/",
                "dtype": "<u2",
                "fill_value": 0,
                "filters": None,
                "order": "C",
                "shape": output_shape,
                "zarr_format": 2,
            },
        }
    ).result()


def initialize_output_volume(
    output_params: input_output.OutputParameters,
    output_volume_size: tuple[int, int, int],
) -> input_output.OutputArray:

    output = None
    assert output_params.datastore in [
        0,
        1,
    ], "Only 0 = Dask and 1 = Tensorstore supported."
    if output_params.datastore == 0:
        output = initialize_output_volume_dask(
            output_params, output_volume_size
        )
    elif output_params.datastore == 1:
        output = initialize_output_volume_tensorstore(
            output_params, output_volume_size
        )
    return output


def get_cell_count_zyx(
    volume_size: tuple[int, int, int], cell_size: tuple[int, int, int]
) -> tuple[int, int, int]:
    """
    Total amount of z,y, and x cells returned in that order.
    Input sizes are in canonical zyx order.
    """
    z_cnt = int(np.ceil(volume_size[0] / cell_size[0]))
    y_cnt = int(np.ceil(volume_size[1] / cell_size[1]))
    x_cnt = int(np.ceil(volume_size[2] / cell_size[2]))

    return z_cnt, y_cnt, x_cnt

def run_fusion(
    input_s3_path: str,
    xml_path: str,
    channel_num: int,
    output_params: input_output.OutputParameters,
    blend_option: str,
    default_chunk_size,
    cpu_cell_size,
    datastore: int = 0,
    volume_sampler_stride: int = 1,
    volume_sampler_start: int = 0
):
    """
    Fusion algorithm.
    Inputs:
    input_s3_path, xml_path, channel_num: for reading the incoming dataset
    output_params: configurations on output volume
    blend_option: type of blending algorithm

    Optional/Advanced:
    datastore: Option to swap to tensorstore reading.
    cpu/gpu cell_size: size of subvolume in output volume sent to each cpu/gpu worker.
    volume_sampler stride/start: options for partitioning work across capsules.
    """
    if not ray.is_initialized():
        # ray.init(address="auto")  
        ray.init()

    logging.basicConfig(
        format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M"
    )
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)

    # Base Initalization
    dataset = input_output.BigStitcherDatasetChannel(
        xml_path,
        input_s3_path,
        channel_num,
        datastore=datastore
    )
    a, b, c, d, e, f, g = initialize_fusion(dataset, output_params)
    tile_arrays = a
    tile_transforms = b
    tile_sizes_zyx = c
    tile_aabbs = d
    tile_paths = e
    output_volume_size = f
    output_volume_origin = g
    output_volume = initialize_output_volume(output_params, output_volume_size)
    tile_layout = utils.parse_yx_tile_layout(xml_path, channel_num)

    print("================================\n")

    LOGGER.info(f"Number of Tiles: {len(tile_arrays)}")
    LOGGER.info(f"{output_volume_size=}")
    print("Tile layout")
    print(tile_layout)

    # Set Blending
    blending_options = {
        'max_projection': blend.MaxProjection(),
        'weighted_linear_blending': blend.WeightedLinearBlending(tile_aabbs)
    }
    if not (blend_option in blending_options):
        raise ValueError(f"Please choose from the following blending options: {blending_options.keys()}")
    blend_module = blending_options[blend_option]

    # Set CPU/GPU cell_size
    if output_params.chunksize != default_chunk_size:
        if cpu_cell_size is None:
            raise ValueError('Custom CPU/GPU cell sizes must be provided for custom output chunksize.')
        CPU_CELL_SIZE = cpu_cell_size
    if cpu_cell_size:
        CPU_CELL_SIZE = cpu_cell_size

    # Start the CPU Runtime
    overlap_volume_sampler_overlapping_only = FusionVolumeSampler(
        tile_transforms,
        tile_sizes_zyx,
        tile_aabbs,
        output_volume_size,
        output_volume_origin,
        CPU_CELL_SIZE,
        output_params.chunksize[2:],
        tile_layout,
        traverse_overlap=True,
        stride=volume_sampler_stride,
        start=volume_sampler_start
    )

    overlap_volume_sampler_non_overlapping_only = FusionVolumeSampler(
        tile_transforms,
        tile_sizes_zyx,
        tile_aabbs,
        output_volume_size,
        output_volume_origin,
        CPU_CELL_SIZE,
        output_params.chunksize[2:],
        tile_layout,
        traverse_overlap=False,
        stride=volume_sampler_stride,
        start=volume_sampler_start
    )

    # IMPORTANT: recreate chain for submission (safe even if samplers are one-shot iterators)
    overlap_volume_sampler = chain(overlap_volume_sampler_non_overlapping_only, overlap_volume_sampler_overlapping_only)

    print(f"Num jobs: {len(overlap_volume_sampler_non_overlapping_only) + len(overlap_volume_sampler_overlapping_only)}")

    store = output_volume.store
    write_root = getattr(store, "root", None) or getattr(store, "path", None)
    write_ds = output_volume.path 

    # @ray.remote
    # def process_color_cell(curr_cell, src_ids, tile_transforms, tile_sizes_zyx, output_volume_origin, blend_module,
    #                        tile_paths, write_root, write_ds):

    #     cpu_fusion(
    #         tile_transforms, tile_sizes_zyx, output_volume_origin, blend_module, curr_cell, src_ids,
    #         tile_paths, write_root, write_ds
    #     )

    #     return None


    # BATCH_SIZE = 32
    # NUM_ACTORS = 500  # keep this because you want long-lived caches/clients

    # def _batch_iter(iterable, batch_size):
    #     batch = []
    #     for item in iterable:
    #         batch.append(item)
    #         if len(batch) == batch_size:
    #             yield batch
    #             batch = []
    #     if batch:
    #         yield batch

    # actors = [
    #     FusionWorker.remote(
    #         tile_transforms,
    #         tile_sizes_zyx,
    #         output_volume_origin,
    #         blend_module,
    #         tile_paths,
    #         write_root,
    #         write_ds,
    #     )
    #     for _ in range(NUM_ACTORS)
    # ]

    # # submit ALL batches immediately (no gating)
    # pending = [
    #     actors[i % NUM_ACTORS].process_batch.remote(jobs)
    #     for i, jobs in enumerate(_batch_iter(overlap_volume_sampler, BATCH_SIZE))
    # ]

    # # wait for everything
    # ray.get(pending)

    BATCH_SIZE = 1
    MAX_IN_FLIGHT = 2000   # for ~400 CPUs, start ~1600–2400
    DRAIN_N = 200          # drain in waves; 200–400 is good

    def _batch_iter(iterable, batch_size):
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    pending = []
    for jobs in _batch_iter(overlap_volume_sampler, BATCH_SIZE):
        pending.append(
            process_batch_task.remote(
                jobs,
                tile_transforms,
                tile_sizes_zyx,
                output_volume_origin,
                blend_module,
                tile_paths,
                write_root,
                write_ds,
            )
        )

        if len(pending) >= MAX_IN_FLIGHT:
            done, pending = ray.wait(pending, num_returns=DRAIN_N)
            ray.get(done) 

    # final drain
    ray.get(pending)

    ### newest below
    # BATCH_SIZE = 8         
    # NUM_ACTORS = 500                 
    # MAX_IN_FLIGHT = 1000
    # DRAIN_N = 50

    # def _batch_iter(iterable, batch_size):
    #     batch = []
    #     for item in iterable:
    #         batch.append(item)
    #         if len(batch) == batch_size:
    #             yield batch
    #             batch = []
    #     if batch:
    #         yield batch

    # actors = [
    #     FusionWorker.remote(
    #         tile_transforms,
    #         tile_sizes_zyx,
    #         output_volume_origin,
    #         blend_module,
    #         tile_paths,
    #         write_root,
    #         write_ds,
    #     )
    #     for _ in range(NUM_ACTORS)
    # ]

    # pending = []
    # i = 0
    # for jobs in _batch_iter(overlap_volume_sampler, BATCH_SIZE):
    #     pending.append(actors[i % NUM_ACTORS].process_batch.remote(jobs))
    #     i += 1

    #     if len(pending) >= MAX_IN_FLIGHT:
    #         done, pending = ray.wait(pending, num_returns=DRAIN_N)

    # while pending:
    #     done, pending = ray.wait(pending, num_returns=min(DRAIN_N, len(pending)))


    # BATCH_SIZE = 8
    # MAX_IN_FLIGHT = 1000
    # DRAIN_N = 50

    # @ray.remote
    # def process_cell_batch(
    #     jobs,
    #     tile_transforms, tile_sizes_zyx, output_volume_origin, blend_module,
    #     tile_paths, write_root, write_ds
    # ):
    #     in_arr_cache = {}
    #     for curr_cell, src_ids in jobs:
    #         cpu_fusion(
    #             tile_transforms, tile_sizes_zyx, output_volume_origin, blend_module,
    #             curr_cell, src_ids, tile_paths, write_root, write_ds, in_arr_cache
    #         )
    #     return None

    # def _batch_iter(iterable, batch_size):
    #     batch = []
    #     for item in iterable:
    #         batch.append(item)
    #         if len(batch) == batch_size:
    #             yield batch
    #             batch = []
    #     if batch:
    #         yield batch

    # pending = []
    # for jobs in _batch_iter(overlap_volume_sampler, BATCH_SIZE):
    #     pending.append(
    #         process_cell_batch.remote(
    #             jobs, tile_transforms, tile_sizes_zyx, output_volume_origin,
    #             blend_module, tile_paths, write_root, write_ds
    #         )
    #     )

    #     if len(pending) >= MAX_IN_FLIGHT:
    #         done, pending = ray.wait(pending, num_returns=DRAIN_N)

    # while pending:
    #     done, pending = ray.wait(pending, num_returns=min(DRAIN_N, len(pending)))

    ######

    # submit one task per overlap cell (same process: one remote per job, then ray.get)
    # futures = [
    #     process_color_cell.remote(
    #         curr_cell, src_ids, tile_transforms, tile_sizes_zyx, output_volume_origin,
    #         blend_module, tile_paths, write_root, write_ds
    #     )
    #     for i, (curr_cell, src_ids) in enumerate(overlap_volume_sampler)
    # ]

    # ray.get(futures)

    # for i, (curr_cell, src_ids) in enumerate(overlap_volume_sampler):
    #     cpu_fusion(
    #         tile_transforms,
    #         tile_sizes_zyx,
    #         output_volume_origin,
    #         blend_module,
    #         curr_cell,
    #         src_ids,
    #         tile_paths,
    #         write_root,
    #         write_ds
    #     )


# def cpu_fusion(
#     tile_transforms: dict[int, list[geometry.Transform]],
#     tile_sizes_zyx: dict[int, tuple[int, int, int]],
#     output_volume_origin: tuple[float, float, float],
#     blend_module: blend.BlendingModule,
#     cell_aabb: geometry.AABB,
#     src_ids: list[int],
#     tile_paths,
#     write_root,
#     write_ds,
#     in_arr_cache
# ):

#     z = int(cell_aabb[1] - cell_aabb[0])
#     y = int(cell_aabb[3] - cell_aabb[2])
#     x = int(cell_aabb[5] - cell_aabb[4])
#     vox = z * y * x
#     n_src = len(src_ids)

#     # if do_print:
#     print(f"[cpu_fusion] pid={os.getpid()} START shape=({z},{y},{x}) vox={vox:,} n_src={n_src} ")

#     t_total0 = time.perf_counter()

#     # s3_read = s3fs.S3FileSystem(anon=True)
#     # s3_write = s3fs.S3FileSystem(anon=False)
#     s3_read = _get_s3_read()
#     s3_write = _get_s3_write()
#     out_store = s3fs.S3Map(root=write_root, s3=s3_write, check=False)
#     arr = zarr.open(store=out_store, mode="a")[write_ds]

#     # overlap_contributions: list[torch.Tensor] = []
#     overlap_contributions: list[np.ndarray] = []
#     for t_id in src_ids:
#         # Retrieve source image
#         image_slice: tuple[slice, slice, slice, slice, slice] = utils.calculate_image_crop(
#             cell_aabb,
#             output_volume_origin,
#             tile_transforms[t_id],
#             tile_sizes_zyx[t_id],
#             device='cpu'
#         )

#         # s3_read = s3fs.S3FileSystem(anon=True)
#         src_path = tile_paths[t_id]
#         # store = s3fs.S3Map(root=src_path, s3=s3_read, check=False)
#         # zarr_arr = zarr.open(store=store, mode="r")
#         src_path = tile_paths[t_id]
#         zarr_arr = in_arr_cache.get(src_path)
#         if zarr_arr is None:
#             store = s3fs.S3Map(root=src_path, s3=s3_read, check=False)
#             zarr_arr = zarr.open(store=store, mode="r")
#             in_arr_cache[src_path] = zarr_arr

#         t_read0 = time.perf_counter()
#         src_img = zarr_arr[image_slice]
#         t_read = time.perf_counter() - t_read0

#         # src_tensor = torch.Tensor(src_img.astype(np.int16))

#         # Calculate sample field
#         t_field0 = time.perf_counter()
#         sample_field = utils.calculate_sample_field_np(
#             cell_aabb,
#             output_volume_origin,
#             tile_transforms[t_id],
#             tile_sizes_zyx[t_id],
#             # device='cpu'
#         )
#         t_field = time.perf_counter() - t_field0

#         # Perform interpolation
#         t_interp0 = time.perf_counter()
#         contribution = utils.interpolate_np(
#             # src_tensor,
#             src_img,
#             sample_field,
#         )
#         t_interp = time.perf_counter() - t_interp0

#         # if do_print and n_src >= HEAVY_N_SRC:
#         print(f"[cpu_fusion] pid={os.getpid()} t_id={t_id} read={t_read:.3f}s field={t_field:.3f}s interp={t_interp:.3f}s", flush=True)

#         overlap_contributions.append(contribution)

#     # Perform blending
#     t_blend0 = time.perf_counter()
#     # blended_cell = blend_module.blend(
#     #     overlap_contributions,
#     #     device='cpu',
#     #     kwargs={
#     #         "chunk_tile_ids": src_ids,
#     #         "cell_box": cell_aabb
#     #     }
#     # )

#     blended_cell = blend_module.blend(
#         overlap_contributions,
#         kwargs={
#             "chunk_tile_ids": src_ids,
#             "cell_box": cell_aabb,
#         },
#     )

#     t_blend = time.perf_counter() - t_blend0

#     # Write
#     output_slice = (
#         slice(0, 1),
#         slice(0, 1),
#         slice(cell_aabb[0], cell_aabb[1]),
#         slice(cell_aabb[2], cell_aabb[3]),
#         slice(cell_aabb[4], cell_aabb[5]),
#     )

#     # Convert from float32 -> canonical uint16
#     blended_cell = np.nan_to_num(blended_cell)
#     blended_cell = np.clip(blended_cell, 0, 65535)
#     output_chunk = blended_cell.astype(np.uint16)

#     t_write0 = time.perf_counter()
#     # s3_write = s3fs.S3FileSystem(anon=False)
#     # out_store = s3fs.S3Map(root=write_root, s3=s3_write)
#     # arr = zarr.open(store=out_store, mode="a")[write_ds]
#     arr[output_slice] = np.ascontiguousarray(output_chunk)
#     t_write = time.perf_counter() - t_write0

#     # if do_print:
#     total = time.perf_counter() - t_total0
#     print(f"[cpu_fusion] pid={os.getpid()} DONE blend={t_blend:.3f}s write={t_write:.3f}s total={total:.3f}s", flush=True)

### NEWEST STUFF ###


# def run_fusion(
#     input_s3_path: str,
#     xml_path: str,
#     channel_num: int,
#     output_params: input_output.OutputParameters,
#     blend_option: str,
#     default_chunk_size,
#     cpu_cell_size,
#     datastore: int = 0,
#     volume_sampler_stride: int = 1,
#     volume_sampler_start: int = 0
# ):
#     """
#     Fusion algorithm.
#     Inputs:
#     input_s3_path, xml_path, channel_num: for reading the incoming dataset
#     output_params: configurations on output volume
#     blend_option: type of blending algorithm

#     Optional/Advanced:
#     datastore: Option to swap to tensorstore reading.
#     cpu/gpu cell_size: size of subvolume in output volume sent to each cpu/gpu worker.
#     volume_sampler stride/start: options for partitioning work across capsules.
#     """

#     logging.basicConfig(
#         format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M"
#     )
#     LOGGER = logging.getLogger(__name__)
#     LOGGER.setLevel(logging.INFO)

#     # Base Initalization
#     dataset = input_output.BigStitcherDatasetChannel(xml_path, 
#                                            input_s3_path, 
#                                            channel_num, 
#                                            datastore=datastore)
#     a, b, c, d, e, f, g = initialize_fusion(dataset, output_params)
#     tile_arrays = a
#     tile_transforms = b
#     tile_sizes_zyx = c
#     tile_aabbs = d
#     tile_paths = e
#     output_volume_size = f
#     output_volume_origin = g
#     output_volume = initialize_output_volume(output_params, output_volume_size)
#     tile_layout = utils.parse_yx_tile_layout(xml_path, channel_num)

#     print("\n========== RUN CONFIG ==========")
#     print(f"Number of Tiles: {len(tile_arrays)}")
#     print(f"output_volume_size={output_volume_size}")
#     print(f"output_params.chunksize={output_params.chunksize}")
#     print(f"default_chunk_size={default_chunk_size}")
#     print(f"requested cpu_cell_size={cpu_cell_size}")
#     print(f"volume_sampler_stride={volume_sampler_stride}")
#     print(f"volume_sampler_start={volume_sampler_start}")
#     print(f"torch.get_num_threads()={torch.get_num_threads()}")
#     print(f"torch.get_num_interop_threads()={torch.get_num_interop_threads()}")
#     print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
#     print(f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")
#     print(f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
#     print(f"NUMEXPR_NUM_THREADS={os.environ.get('NUMEXPR_NUM_THREADS')}")

#     LOGGER.info(f"Number of Tiles: {len(tile_arrays)}")
#     LOGGER.info(f"{output_volume_size=}")
#     print('Tile layout')
#     print(tile_layout)

#     # Set Blending
#     blending_options = {'max_projection': blend.MaxProjection(),
#                         'weighted_linear_blending': blend.WeightedLinearBlending(tile_aabbs)}
#     if not (blend_option in blending_options):
#         raise ValueError(f"Please choose from the following blending options: {blending_options.keys()}")
#     blend_module = blending_options[blend_option]

#     # Set CPU/GPU cell_size
#     if output_params.chunksize != default_chunk_size:
#         if cpu_cell_size is None:
#             raise ValueError('Custom CPU/GPU cell sizes must be provided for custom output chunksize.')
#         CPU_CELL_SIZE = cpu_cell_size
#     if cpu_cell_size:
#         CPU_CELL_SIZE = cpu_cell_size

#     # Start the CPU Runtime
#     overlap_volume_sampler_overlapping_only = FusionVolumeSampler(tile_transforms,
#                                                 tile_sizes_zyx,
#                                                 tile_aabbs,
#                                                 output_volume_size,
#                                                 output_volume_origin,
#                                                 CPU_CELL_SIZE,
#                                                 output_params.chunksize[2:],
#                                                 tile_layout,
#                                                 traverse_overlap = True,
#                                                 stride=volume_sampler_stride,
#                                                 start=volume_sampler_start)
    
#     overlap_volume_sampler_non_overlapping_only = FusionVolumeSampler(tile_transforms,
#                                             tile_sizes_zyx,
#                                             tile_aabbs,
#                                             output_volume_size,
#                                             output_volume_origin,
#                                             CPU_CELL_SIZE,
#                                             output_params.chunksize[2:],
#                                             tile_layout,
#                                             traverse_overlap = False,
#                                             stride=volume_sampler_stride,
#                                             start=volume_sampler_start)

#     overlap_volume_sampler = chain(overlap_volume_sampler_non_overlapping_only, overlap_volume_sampler_overlapping_only)

#     print(f"Num jobs: {len(overlap_volume_sampler_non_overlapping_only) + len(overlap_volume_sampler_overlapping_only)}")

#     store = output_volume.store
#     write_root = getattr(store, "root", None) or getattr(store, "path", None)
#     write_ds = output_volume.path 

#     @ray.remote(num_cpus=2)
#     def process_color_cell(curr_cell, src_ids, tile_transforms, tile_sizes_zyx, output_volume_origin, blend_module, 
#                            tile_paths, write_root, write_ds):
        
#         cpu_fusion(tile_transforms, tile_sizes_zyx, output_volume_origin, blend_module, curr_cell, src_ids, 
#                    tile_paths, write_root, write_ds)
        
#         return None

#     # submit one task per overlap cell 
#     futures = [
#         process_color_cell.remote(curr_cell, src_ids, tile_transforms, tile_sizes_zyx, output_volume_origin, 
#                                   blend_module, tile_paths, write_root, write_ds)

#         for i, (curr_cell, src_ids) in enumerate(overlap_volume_sampler)
#     ]

#     ray.get(futures)

#     # for i, (curr_cell, src_ids) in enumerate(overlap_volume_sampler):
#     #     cpu_fusion(
#     #         tile_transforms,
#     #         tile_sizes_zyx,
#     #         output_volume_origin,
#     #         blend_module,
#     #         curr_cell,
#     #         src_ids,
#     #         tile_paths,
#     #         write_root,
#     #         write_ds
#     #     )

# def cpu_fusion(
#     tile_transforms: dict[int, list[geometry.Transform]],
#     tile_sizes_zyx: dict[int, tuple[int, int, int]],
#     output_volume_origin: tuple[float, float, float],
#     blend_module: blend.BlendingModule,
#     cell_aabb: geometry.AABB,
#     src_ids: list[int],
#     tile_paths,
#     write_root,
#     write_ds
# ):
#     overlap_contributions: list[torch.Tensor] = []
#     for t_id in src_ids:
#         # Retrieve source image
#         image_slice: tuple[slice, slice, slice, slice, slice] = \
#                 utils.calculate_image_crop(cell_aabb,
#                                             output_volume_origin,
#                                             tile_transforms[t_id],
#                                             tile_sizes_zyx[t_id],
#                                             device='cpu')

#         s3_read = s3fs.S3FileSystem(anon=True)
#         src_path = tile_paths[t_id]
#         store = s3fs.S3Map(root=src_path, s3=s3_read)
#         zarr_arr = zarr.open(store=store, mode="r")
#         src_img = zarr_arr[image_slice]

#         # src_bytes = src_img.nbytes
#         # src_mb = src_bytes / (1024 ** 2)
#         # src_gb = src_bytes / (1024 ** 3)
#         # print(
#         #     f"[fusion] src_img slice shape={src_img.shape}, "
#         #     f"size={src_mb:.2f} MiB ({src_gb:.4f} GiB)"
#         # )
        
#         src_tensor = torch.Tensor(src_img.astype(np.int16))

#         # Calculate sample field
#         sample_field = \
#             utils.calculate_sample_field(cell_aabb,
#                                         output_volume_origin,
#                                         tile_transforms[t_id],
#                                         tile_sizes_zyx[t_id],
#                                         device='cpu')

#         # Perform interpolation
#         contribution = utils.interpolate(src_tensor,
#                                          sample_field,
#                                          device="cpu")

#         overlap_contributions.append(contribution)

#     # Perform blending
#     blended_cell = blend_module.blend(overlap_contributions,
#                                     device='cpu',
#                                     kwargs={
#                                     "chunk_tile_ids": src_ids,
#                                     "cell_box": cell_aabb
#                                     })

#     # Write
#     output_slice = (
#         slice(0, 1),
#         slice(0, 1),
#         slice(cell_aabb[0], cell_aabb[1]),
#         slice(cell_aabb[2], cell_aabb[3]),
#         slice(cell_aabb[4], cell_aabb[5]),
#     )

#     # Convert from float32 -> canonical uint16
#     blended_cell = np.nan_to_num(blended_cell)
#     blended_cell = np.clip(blended_cell, 0, 65535)
#     output_chunk = blended_cell.astype(np.uint16)

#     # chunk_bytes = output_chunk.nbytes
#     # chunk_mb = chunk_bytes / (1024 ** 2)
#     # chunk_gb = chunk_bytes / (1024 ** 3)
#     # print(f"[fusion] output_chunk shape={output_chunk.shape}, size={chunk_mb:.2f} MiB ({chunk_gb:.4f} GiB)")

#     s3_write = s3fs.S3FileSystem(anon=False)
#     out_store = s3fs.S3Map(root=write_root, s3=s3_write)
#     arr = zarr.open(store=out_store, mode="a")[write_ds]

#     arr[output_slice] = np.ascontiguousarray(output_chunk)

# class CloudDataset(Dataset):
#     def __init__(
#         self,
#         tile_arrays: dict[int, input_output.InputArray],
#         tile_transforms: dict[int, list[geometry.Transform]],
#         tile_sizes_zyx: dict[int, tuple[int, int, int]],
#         tile_aabbs: dict[int, geometry.AABB],
#         output_volume_size: tuple[int, int, int],
#         output_volume_origin: tuple[float, float, float],
#         cell_size: tuple[int, int, int],
#         pin_memory: bool=True
#         ) -> None:
#         """
#         Input fields are produced from
#         fusion.initalize_fusion(..)

#         Following codebase convention,
#         input 3-ples are expected in zyx order.
#         """

#         # Store input arguments
#         self.tile_arrays: dict[int, input_output.InputArray] = tile_arrays
#         self.tile_transforms: dict[int, list[geometry.Transform]] = tile_transforms
#         self.tile_sizes_zyx: dict[int, tuple[int, int, int]] = tile_sizes_zyx
#         self.tile_aabbs: dict[int, geometry.AABB] = tile_aabbs
#         self.output_volume_size: tuple[int, int, int] = output_volume_size
#         self.output_volume_origin: tuple[float, float, float] = output_volume_origin
#         self.cell_size: tuple[int, int, int] = cell_size
#         self.pin_memory: bool = pin_memory

#     def __getitem__(self, input_bundle):
#         """
#         Return src_tensor associated with the
#         input cell_aabb/t_id.
#         """

#         cell_aabb, src_ids = input_bundle

#         src_tensors: list[torch.Tensor] = []
#         for t_id in src_ids:
#             image_slice: tuple[slice, slice, slice, slice, slice] = \
#             utils.calculate_image_crop(cell_aabb,
#                                         self.output_volume_origin,
#                                         self.tile_transforms[t_id],
#                                         self.tile_sizes_zyx[t_id],
#                                         device='cpu')

#             result = self.tile_arrays[t_id][image_slice]

#             # uint16 -> int16 for pytorch compatibility.
#             # Max intensity values of original data are close to 1000,
#             # no where near 1/2 uint16 (32,767), so this is safe.
#             if self.pin_memory:
#                 result = torch.Tensor(result.astype(np.int16)).pin_memory()
#             else:
#                 result = torch.Tensor(result.astype(np.int16))

#             src_tensors.append(result)

#         return cell_aabb, src_ids, src_tensors

#     def __len__(self):
#         z_cnt, y_cnt, x_cnt = \
#             get_cell_count_zyx(self.output_volume_size, self.cell_size)
#         total_cells = z_cnt * y_cnt * x_cnt
#         return total_cells


class FusionVolumeSampler(cq.VolumeSampler):
    def __init__(
        self,
        tile_transforms: dict[int, list[geometry.Transform]],
        tile_sizes_zyx: dict[int, tuple[int, int, int]],
        tile_aabbs: dict[int, geometry.AABB],
        output_volume_size: tuple[int, int, int],
        output_volume_origin: tuple[float, float, float],
        cell_size: tuple[int, int, int],
        chunk_size: tuple[int, int, int],
        tile_layout: list[list[int]],
        traverse_overlap: bool = False,
        stride: int = 1,
        start: int = 0,
    ):
        """
        NOTE:
        Stride/start define cell positions within
        user's choice of region.

        Work within user's choice of region can be distributed
        among workers by setting stride = N and start = {0 -> N - 1}
        Ex: stride = 3, start = {0, 1, 2}
        """
        super().__init__(output_volume_size, cell_size)

        if ((cell_size[0] % chunk_size[0] != 0) or
            (cell_size[1] % chunk_size[1] != 0) or
            (cell_size[2] % chunk_size[2] != 0)):
            raise ValueError(f"""Cell_size: {cell_size}
                                 Chunk_size: {chunk_size}
                                 Please make cell_size a multiple of chunk_size
                                 to prevent race conditions.""")

        if start >= stride:
            raise ValueError('Start index must be strictly less than stride length.')

        # Store fields
        self.tile_transforms = tile_transforms
        self.tile_sizes_zyx = tile_sizes_zyx
        self.tile_aabbs = tile_aabbs
        self.output_volume_size = output_volume_size
        self.output_volume_origin = output_volume_origin
        self.cell_size = cell_size
        self.chunk_size = chunk_size
        self.tile_layout = tile_layout
        self.traverse_overlap = traverse_overlap
        self.stride = stride
        self.start = start

        # Calculate the non/overlap regions
        self.overlap_regions: list[geometry.AABB] = []
        self.non_overlap_regions: list[geometry.AABB] = []

        # Overlap regions = true overlap AABB extended in z to output vol size
        # Rounded to the nearest chunk to prevent race conditions.
        tile_to_overlap_ids, overlaps = \
            utils.get_overlap_regions(tile_layout, tile_aabbs)

        modified_overlaps: dict[int, geometry.AABB]= {}
        cz, cy, cx = chunk_size
        for o_id, o_aabb in overlaps.items():
            modified_o_aabb = (0,
                            output_volume_size[0],
                            np.floor(o_aabb[2] / cy) * cy,
                            np.ceil(o_aabb[3] / cy) * cy,
                            np.floor(o_aabb[4] / cx) * cx,
                            np.ceil(o_aabb[5] / cx) * cx)
            self.overlap_regions.append(modified_o_aabb)
            modified_overlaps[o_id] = modified_o_aabb

        # Non-overlap regions = z-extended tile AABB's - respective overlap AABB's.
        for t_id, o_ids in tile_to_overlap_ids.items():
            # This is the base nullspace
            t_aabb = list(self.tile_aabbs[t_id])
            t_aabb[0] = 0
            t_aabb[1] = output_volume_size[0]

            for o_id in o_ids:
                o_aabb = modified_overlaps[o_id]
                oy_length = o_aabb[3] - o_aabb[2]
                ox_length = o_aabb[5] - o_aabb[4]

                # y_min is inside overlap y-boundaries
                # o_aabb is long and flat
                if ((o_aabb[2] <= t_aabb[2] <= o_aabb[3]) and
                     ox_length > oy_length):
                    t_aabb[2] = o_aabb[3]

                # y_max is inside overlap y-boundaries
                # o_aabb is long and flat
                if ((o_aabb[2] <= t_aabb[3] <= o_aabb[3]) and
                     ox_length > oy_length):
                    t_aabb[3] = o_aabb[2]

                # x_min is inside overlap x-boundaries
                # o_aabb is tall and skinny
                if ((o_aabb[4] <= t_aabb[4] <= o_aabb[5]) and
                     oy_length > ox_length):
                    t_aabb[4] = o_aabb[5]

                # x_max is inside overlap x-boundaries
                # o_aabb is tall and skinny
                if ((o_aabb[4] <= t_aabb[5] <= o_aabb[5]) and
                     oy_length > ox_length):
                    t_aabb[5] = o_aabb[4]

            self.non_overlap_regions.append(tuple(t_aabb))

        # For border non-overlap regions,
        # round to output_volume min/max
        # such that all cells generated from
        # inside are chunk aligned.
        # Rounding are simply extensions to the y/x region boundaries.
        cz, cy, cx = chunk_size
        oz, oy, ox = output_volume_size
        updated_regions: list[geometry.AABB] = []
        for o_aabb in self.non_overlap_regions:
            updated_aabb = list(o_aabb)
            if o_aabb[2] < cy:
                updated_aabb[2] = 0
            if (oy - cy) < o_aabb[3] < oy:
                updated_aabb[3] = oy
            if o_aabb[4] < cx:
                updated_aabb[4] = 0
            if (ox - cx) < o_aabb[5] < ox:
                updated_aabb[5] = ox
            updated_regions.append(tuple(updated_aabb))
        self.non_overlap_regions = updated_regions

        # Rounding all regions appropriately to integers
        self.overlap_regions = [(int(np.floor(o_aabb[0])),
                                 int(np.ceil(o_aabb[1])),
                                 int(np.floor(o_aabb[2])),
                                 int(np.ceil(o_aabb[3])),
                                 int(np.floor(o_aabb[4])),
                                 int(np.ceil(o_aabb[5])))
                                for o_aabb in self.overlap_regions]

        self.non_overlap_regions = [(int(np.floor(o_aabb[0])),
                                    int(np.ceil(o_aabb[1])),
                                    int(np.floor(o_aabb[2])),
                                    int(np.ceil(o_aabb[3])),
                                    int(np.floor(o_aabb[4])),
                                    int(np.ceil(o_aabb[5])))
                                    for o_aabb in self.non_overlap_regions]

    import numpy as np

    def _check_true_collision(
        self,
        cell_box: geometry.AABB,
        transform_list: list[geometry.Transform],
        src_vol_shape_zyx: tuple[int, int, int],
    ) -> bool:
        # Build the 8 corners (zyx) with +/-0.5 offsets
        z_min, z_max, y_min, y_max, x_min, x_max = cell_box
        zs = np.array([z_min + 0.5, z_max - 0.5], dtype=np.float32)
        ys = np.array([y_min + 0.5, y_max - 0.5], dtype=np.float32)
        xs = np.array([x_min + 0.5, x_max - 0.5], dtype=np.float32)

        cell_box_pts = np.array(
            [[z, y, x] for z in zs for y in ys for x in xs],
            dtype=np.float32
        )  # (8, 3)

        # Apply origin
        cell_box_pts += np.asarray(self.output_volume_origin, dtype=np.float32).reshape(1, 3)

        # Apply inverse transforms (NumPy)
        for tfm in reversed(transform_list):
            cell_box_pts = tfm.backward_np(cell_box_pts)

        # AABB of transformed points (zyx)
        z0, z1 = float(cell_box_pts[:, 0].min()), float(cell_box_pts[:, 0].max())
        y0, y1 = float(cell_box_pts[:, 1].min()), float(cell_box_pts[:, 1].max())
        x0, x1 = float(cell_box_pts[:, 2].min()), float(cell_box_pts[:, 2].max())
        cell_box_src: geometry.AABB = (z0, z1, y0, y1, x0, x1)

        sv_z, sv_y, sv_x = src_vol_shape_zyx
        aabb_src: geometry.AABB = (0, sv_z, 0, sv_y, 0, sv_x)

        return utils.check_collision(cell_box_src, aabb_src)

    # def _check_true_collision(
    #     self,
    #     cell_box: geometry.AABB,
    #     transform_list: list[geometry.Transform],
    #     src_vol_shape_zyx: tuple[int, int, int]
    # ) -> bool:
    #     # Build the box
    #     z_min, z_max, y_min, y_max, x_min, x_max = cell_box
    #     z_grid, y_grid, x_grid = torch.meshgrid(
    #         torch.Tensor([z_min + 0.5, z_max - 0.5]),
    #         torch.Tensor([y_min + 0.5, y_max - 0.5]),
    #         torch.Tensor([x_min + 0.5, x_max - 0.5]),
    #         indexing='ij'
    #     )
    #     cell_box_pts = torch.stack([z_grid, y_grid, x_grid], dim=-1)

    #     # Apply inverse transform
    #     cell_box_pts = cell_box_pts + torch.Tensor(self.output_volume_origin)
    #     for tfm in reversed(transform_list):
    #         # cell_box_pts = tfm.backward(cell_box_pts, device='cpu')
    #         cell_box_pts = tfm.backward_np(cell_box_pts)

    #     # Check collision
    #     cell_box_src: geometry.AABB = geometry.aabb_3d(cell_box_pts)
    #     sv_z, sv_y, sv_x = src_vol_shape_zyx
    #     aabb_src: geometry.AABB = (0, sv_z, 0, sv_y, 0, sv_x)

    #     return utils.check_collision(cell_box_src, aabb_src)

    def __len__(self):
        cz, cy, cx = self.cell_size
        regions = self.non_overlap_regions
        if self.traverse_overlap:
            regions = self.overlap_regions

        total_count = 0
        for region in regions:
            rz_min, rz_max, ry_min, ry_max, rx_min, rx_max = region
            rz_length = rz_max - rz_min
            ry_length = ry_max - ry_min
            rx_length = rx_max - rx_min

            z_cnt = int(np.ceil(rz_length / cz))
            y_cnt = int(np.ceil(ry_length / cy))
            x_cnt = int(np.ceil(rx_length / cx))

            total_count += (z_cnt * y_cnt * x_cnt)
        
        stride_count = int(total_count / self.stride)

        return stride_count

    def __iter__(self):
        """
        Modified metadata generator.
        Iterates through cells and intersecting tile ids.
        """
        cz, cy, cx = self.cell_size

        regions = self.non_overlap_regions
        if self.traverse_overlap:
            regions = self.overlap_regions

        cell_num = 0
        for region in regions:
            rz_min, rz_max, ry_min, ry_max, rx_min, rx_max = region
            for z in range(rz_min, rz_max, cz):
                for y in range(ry_min, ry_max, cy):
                    for x in range(rx_min, rx_max, cx):
                        cell_num += 1

                        curr_cell: geometry.AABB = \
                            (z, min(z + cz, rz_max),
                            y, min(y + cy, ry_max),
                            x, min(x + cx, rx_max))

                        src_ids: list[int] = \
                        [t_id
                        for (t_id, t_aabb) in self.tile_aabbs.items()
                        if self._check_true_collision(curr_cell,
                                                      self.tile_transforms[t_id],
                                                      self.tile_sizes_zyx[t_id])
                        ]

                        true_overlap_condition = (len(src_ids) != 0)
                        stride_condition = (cell_num % self.stride == self.start)

                        if true_overlap_condition and stride_condition:
                            yield curr_cell, src_ids
