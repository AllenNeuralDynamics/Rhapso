from __future__ import annotations
import logging
import ray
import numpy as np
import s3fs
import os
from itertools import chain
import time
import zarr
from collections import OrderedDict
import Rhapso.translation_fusion.blend as blend
import Rhapso.translation_fusion.cloud_queue as cq
import Rhapso.translation_fusion.geometry as geometry
import Rhapso.translation_fusion.input_output as input_output
import Rhapso.translation_fusion.fusion_utils as utils

S3_READ = None
S3_WRITE = None

# Per-worker-process caches (Ray reuses worker processes for tasks)
_OUT_ARR_CACHE: dict[tuple[str, str], zarr.core.Array] = {}
_TILE_ARR_LRU: "OrderedDict[str, zarr.core.Array]" = OrderedDict()
_TILE_LRU_MAX = 32  

def get_s3_read(max_pool_connections: int = 25):
    global S3_READ
    if S3_READ is None:
        S3_READ = s3fs.S3FileSystem(
            anon=True,
            config_kwargs={"max_pool_connections": max_pool_connections},
        )
    return S3_READ

def get_s3_write(max_pool_connections: int = 25, retries_total: int = 10):
    global S3_WRITE
    if S3_WRITE is None:
        S3_WRITE = s3fs.S3FileSystem(
            anon=False,
            config_kwargs={
                "max_pool_connections": max_pool_connections,
                "retries": {"total_max_attempts": retries_total, "mode": "adaptive"},
            },
        )
    return S3_WRITE

def get_out_arr(write_root: str, write_ds: str) -> zarr.core.Array:
    """
    Cache output zarr array handle per worker process.
    Avoids S3Map + zarr.open per cell.
    """
    key = (write_root, write_ds)
    arr = _OUT_ARR_CACHE.get(key)
    if arr is not None:
        return arr

    out_store = s3fs.S3Map(root=write_root, s3=get_s3_write(), check=False)
    arr = zarr.open(store=out_store, mode="a")[write_ds]
    _OUT_ARR_CACHE[key] = arr
    return arr

def get_tile_arr(tile_path: str) -> zarr.core.Array:
    """
    Cache input tile zarr handle per worker process (bounded LRU).
    Avoids S3Map + zarr.open per tile per cell.
    """
    arr = _TILE_ARR_LRU.get(tile_path)
    if arr is not None:
        _TILE_ARR_LRU.move_to_end(tile_path)
        return arr

    store = s3fs.S3Map(root=tile_path, s3=get_s3_read(), check=False)
    arr = zarr.open(store=store, mode="r")

    _TILE_ARR_LRU[tile_path] = arr
    _TILE_ARR_LRU.move_to_end(tile_path)

    while len(_TILE_ARR_LRU) > _TILE_LRU_MAX:
        _TILE_ARR_LRU.popitem(last=False)

    return arr

# WE GO IN THIS
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

def initialize_output_volume(   
    output_params: input_output.OutputParameters,
    output_volume_size: tuple[int, int, int],
) -> input_output.OutputArray:
    output = initialize_output_volume_dask(
        output_params, output_volume_size
    )
    return output

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
    ray.init() 

    logging.basicConfig(
        format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M"
    )

    # Base Initalization
    dataset = input_output.BigStitcherDatasetChannel(
        xml_path,
        input_s3_path,
        channel_num,
        datastore=datastore
    )
    _, b, c, d, e, f, g = initialize_fusion(dataset, output_params)
    tile_transforms = b
    tile_sizes_zyx = c
    tile_aabbs = d
    tile_paths = e
    output_volume_size = f
    output_volume_origin = g
    output_volume = initialize_output_volume(output_params, output_volume_size)
    tile_layout = utils.parse_yx_tile_layout(xml_path, channel_num)

    print(f"Tile layout {tile_layout}")

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

    # recreate chain for submission 
    overlap_volume_sampler = chain(overlap_volume_sampler_non_overlapping_only, overlap_volume_sampler_overlapping_only)

    print(f"Num jobs non overlapping: {len(overlap_volume_sampler_non_overlapping_only)}") 
    print(f"Num jobs, overlapping: {len(overlap_volume_sampler_overlapping_only)}")

    store = output_volume.store
    write_root = getattr(store, "root", None) or getattr(store, "path", None)
    write_ds = output_volume.path 

    @ray.remote
    def process_color_cell(curr_cell, src_ids, tile_transforms, tile_sizes_zyx, output_volume_origin, blend_module,
                           tile_paths, write_root, write_ds):

        cpu_fusion(tile_transforms, tile_sizes_zyx, output_volume_origin, blend_module, curr_cell, src_ids,
                   tile_paths, write_root, write_ds)

        return None
  
    futures = []
    total_cells = len(overlap_volume_sampler_non_overlapping_only) + len(overlap_volume_sampler_overlapping_only)
    completed = 0
    failed = 0

    t_run0 = time.perf_counter()
    last_pct_printed = -1

    for i, (curr_cell, src_ids) in enumerate(overlap_volume_sampler, start=1):
        futures.append(
            process_color_cell.remote(
                curr_cell, src_ids, tile_transforms, tile_sizes_zyx, output_volume_origin,
                blend_module, tile_paths, write_root, write_ds
            )
        )

        # drain completions while we are submitting so progress prints during the run
        done, futures = ray.wait(futures, num_returns=1, timeout=0)
        while done:
            try:
                ray.get(done)
                completed += len(done)
            except Exception as e:
                failed += len(done)
                print(f"[fusion][ERROR] {len(done)} task(s) failed: {type(e).__name__}: {e}", flush=True)

            progress_pct = (completed / total_cells) * 100.0
              
            pct_int = int(progress_pct)
            if pct_int > last_pct_printed:
                last_pct_printed = pct_int
                elapsed = time.perf_counter() - t_run0
                rate = completed / max(elapsed, 1e-9)
                eta_s = (total_cells - completed) / max(rate, 1e-9)
                print(
                    f"[fusion] Progress: ok={completed} failed={failed} total={total_cells} ({pct_int}%) "
                    f"elapsed={elapsed/60:.1f}m rate={rate:.2f} cells/s eta={eta_s/60:.1f}m",
                    flush=True,
                )
            
            done, futures = ray.wait(futures, num_returns=1, timeout=0)

    # finish remaining tasks and keep printing progress
    while futures:
        done, futures = ray.wait(futures, num_returns=1, timeout=1.0)
        if done:
            try:
                ray.get(done)
                completed += len(done)
            except Exception as e:
                failed += len(done)
                print(f"[fusion][ERROR] {len(done)} task(s) failed: {type(e).__name__}: {e}", flush=True)

            progress_pct = (completed / total_cells) * 100.0
            
            pct_int = int(progress_pct)
            if pct_int > last_pct_printed:
                last_pct_printed = pct_int
                elapsed = time.perf_counter() - t_run0
                rate = completed / max(elapsed, 1e-9)
                eta_s = (total_cells - completed) / max(rate, 1e-9)
                print(
                    f"[fusion] Progress: ok={completed} failed={failed} total={total_cells} ({pct_int}%) "
                    f"elapsed={elapsed/60:.1f}m rate={rate:.2f} cells/s eta={eta_s/60:.1f}m",
                    flush=True,
                )
    
    # DEBUG ONLY
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

def cpu_fusion(
    tile_transforms: dict[int, list[geometry.Transform]],
    tile_sizes_zyx: dict[int, tuple[int, int, int]],
    output_volume_origin: tuple[float, float, float],
    blend_module: blend.BlendingModule,
    cell_aabb: geometry.AABB,
    src_ids: list[int],
    tile_paths,
    write_root,
    write_ds,
):

    z = int(cell_aabb[1] - cell_aabb[0])
    y = int(cell_aabb[3] - cell_aabb[2])
    x = int(cell_aabb[5] - cell_aabb[4])
    vox = z * y * x
    n_src = len(src_ids)

    # print(f"[cpu_fusion] pid={os.getpid()} START shape=({z},{y},{x}) vox={vox:,} n_src={n_src} ")

    t_total0 = time.perf_counter()

    arr = get_out_arr(write_root, write_ds)

    overlap_contributions: list[np.ndarray] = []
    for t_id in src_ids:
        image_slice: tuple[slice, slice, slice, slice, slice] = utils.calculate_image_crop(
            cell_aabb,
            output_volume_origin,
            tile_transforms[t_id],
            tile_sizes_zyx[t_id],
        )
        
        zarr_arr = get_tile_arr(tile_paths[t_id])

        t_read0 = time.perf_counter()
        src_img = zarr_arr[image_slice]
        t_read = time.perf_counter() - t_read0

        # Calculate sample field
        t_field0 = time.perf_counter()
        sample_field = utils.calculate_sample_field_np(
            cell_aabb,
            output_volume_origin,
            tile_transforms[t_id],
            tile_sizes_zyx[t_id],
        )
        t_field = time.perf_counter() - t_field0

        # Perform interpolation
        t_interp0 = time.perf_counter()
        contribution = utils.interpolate_np(
            src_img,
            sample_field,
        )
        t_interp = time.perf_counter() - t_interp0

        # print(f"[cpu_fusion] pid={os.getpid()} t_id={t_id} read={t_read:.3f}s field={t_field:.3f}s interp={t_interp:.3f}s", flush=True)

        overlap_contributions.append(contribution)

    # Perform blending
    t_blend0 = time.perf_counter()
    blended_cell = blend_module.blend(
        overlap_contributions,
        kwargs={
            "chunk_tile_ids": src_ids,
            "cell_box": cell_aabb,
        },
    )
    t_blend = time.perf_counter() - t_blend0

    # Write
    output_slice = (
        slice(0, 1),
        slice(0, 1),
        slice(cell_aabb[0], cell_aabb[1]),
        slice(cell_aabb[2], cell_aabb[3]),
        slice(cell_aabb[4], cell_aabb[5]),
    )

    # Convert from float32 -> canonical uint16
    blended_cell = np.nan_to_num(blended_cell)
    blended_cell = np.clip(blended_cell, 0, 65535)
    output_chunk = blended_cell.astype(np.uint16)

    t_write0 = time.perf_counter()
    arr[output_slice] = np.ascontiguousarray(output_chunk)
    t_write = time.perf_counter() - t_write0

    total = time.perf_counter() - t_total0
    # print(f"[cpu_fusion] pid={os.getpid()} DONE blend={t_blend:.3f}s write={t_write:.3f}s total={total:.3f}s", flush=True)

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
            # t_aabb[0] = 0
            # t_aabb[1] = output_volume_size[0]

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
