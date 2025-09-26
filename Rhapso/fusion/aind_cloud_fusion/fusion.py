"""Core fusion algorithm."""

from __future__ import annotations
import logging
import time
import os
import gc
from typing import Optional
import dask.array as da
from dask import delayed
import numpy as np
import s3fs
import tensorstore as ts
import torch
from torch.utils.data import Dataset
import zarr
import ray
import Rhapso.fusion.aind_cloud_fusion.blend as blend
import Rhapso.fusion.aind_cloud_fusion.cloud_queue as cq
import Rhapso.fusion.aind_cloud_fusion.geometry as geometry
import Rhapso.fusion.aind_cloud_fusion.input_output as input_output
import Rhapso.fusion.aind_cloud_fusion.fusion_utils as utils

# Set up logging
LOGGER = logging.getLogger(__name__)


def initialize_fusion(
    dataset: input_output.Dataset,
    output_params: input_output.OutputParameters
) -> tuple[dict, dict, dict, dict, tuple, tuple, torch.Tensor]:
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
    tile_arrays: dict[int, input_output.InputArray] = dataset.tile_volumes_tczyx
    tile_transforms: dict[int, list[geometry.Transform]] = (
        dataset.tile_transforms_zyx
    )

    # Output Data Structures-- tile_sizes_zyx, tile_aabbs
    tile_sizes_zyx: dict[int, tuple[int, int, int]] = {}
    tile_aabbs: dict[int, geometry.AABB] = {}
    tile_boundary_point_cloud_zyx = []

    for tile_id, tile_arr in tile_arrays.items():
        tile_sizes_zyx[tile_id] = zyx = tile_arr.shape[2:]

        z_grid, y_grid, x_grid = torch.meshgrid(
            torch.Tensor([0, zyx[0]]),
            torch.Tensor([0, zyx[1]]),
            torch.Tensor([0, zyx[2]]),
            indexing='ij'
        )
        tile_boundary_pts = torch.stack([z_grid, y_grid, x_grid], dim=-1)

        tfm_list = tile_transforms[tile_id]
        for i, tfm in enumerate(tfm_list):
            tile_boundary_pts = tfm.forward(
                tile_boundary_pts, device=torch.device("cpu")
            )

        tile_aabbs[tile_id] = geometry.aabb_3d(tile_boundary_pts)
        tile_boundary_point_cloud_zyx.append(tile_boundary_pts)
    tile_boundary_point_cloud_zyx = torch.stack(
        tile_boundary_point_cloud_zyx, dim=0
    )

    # Output Data Structures-- OUTPUT_VOLUME_SIZE, OUTPUT_VOLUME_ORIGIN
    # Resolve Output Volume Dimensions and Absolute Position
    global_tile_boundaries = geometry.aabb_3d(tile_boundary_point_cloud_zyx)
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


def run_fusion(  # noqa: C901
    input_s3_path: str,
    xml_path: str,
    channel_num: int,
    output_params: input_output.OutputParameters,
    blend_option: str,
    datastore: int = 0,
    cpu_cell_size: Optional[tuple[int, int, int]] = None,
    gpu_cell_size: Optional[tuple[int, int, int]] = None,
    volume_sampler_stride: int = 1,
    volume_sampler_start: int = 0,
    batch_size: int = 10
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
    batch_size: number of cells to process in each batch (default: 10).
    """

    logging.basicConfig(
        format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M"
    )
    LOGGER.setLevel(logging.INFO)

    # Base Initalization
    dataset = input_output.BigStitcherDatasetChannel(xml_path, 
                                           input_s3_path, 
                                           channel_num, 
                                           datastore=datastore)
    a, b, c, d, e, f = initialize_fusion(dataset, output_params)
    tile_arrays = a
    tile_transforms = b
    tile_sizes_zyx = c
    tile_aabbs = d
    output_volume_size = e
    output_volume_origin = f
    output_volume = initialize_output_volume(output_params, output_volume_size)
    tile_layout = utils.parse_yx_tile_layout(xml_path, channel_num)

    LOGGER.info(f"Number of Tiles: {len(tile_arrays)}")
    LOGGER.info(f"{output_volume_size=}")
    print('Tile layout')
    print(tile_layout)

    # Set Blending
    blending_options = {'max_projection': blend.MaxProjection(),
                        'weighted_linear_blending': blend.WeightedLinearBlending(tile_aabbs)}
    if not (blend_option in blending_options):
        raise ValueError(f"Please choose from the following blending options: {blending_options.keys()}")
    blend_module = blending_options[blend_option]

    # Set CPU/GPU cell_size
    DEFAULT_CHUNKSIZE = (1, 1, 128, 128, 128)
    CPU_CELL_SIZE = (256, 128, 128)  # Reduced cell size to use less memory
    # GPU_CELL_SIZE = calculate_gpu_cell_size(output_volume_size)  # COMMENTED OUT: GPU processing disabled
    
    if output_params.chunksize != DEFAULT_CHUNKSIZE:
        if cpu_cell_size is None:  # Removed gpu_cell_size requirement
            raise ValueError('Custom CPU cell size must be provided for custom output chunksize.')
        CPU_CELL_SIZE = cpu_cell_size
        # GPU_CELL_SIZE = gpu_cell_size  # COMMENTED OUT: GPU processing disabled
    if cpu_cell_size:
        CPU_CELL_SIZE = cpu_cell_size
    # if gpu_cell_size:  # COMMENTED OUT: GPU processing disabled
    #     GPU_CELL_SIZE = gpu_cell_size

    # ============================================================================
    # GPU PROCESSING SECTION - COMMENTED OUT (CPU-ONLY MODE)
    # ============================================================================
    # NOTE: GPU processing has been disabled. Only CPU processing with Ray is used.
    # The following code creates a GPU process but never starts it, causing the
    # "can only join a started process" error. Commenting out to prevent errors.
    # ============================================================================
    
    # # Start GPU Runtime
    # p = torch.multiprocessing.Process(
    #     target=gpu_fusion,
    #     args=(input_s3_path,
    #         xml_path,
    #         channel_num,
    #         output_params,
    #         tile_layout,
    #         output_volume,
    #         GPU_CELL_SIZE,
    #         volume_sampler_stride,
    #         volume_sampler_start,
    #         datastore)
    # )
    # # p.daemon = True
    # # p.start()

    # Start the CPU Runtime
    overlap_volume_sampler = FusionVolumeSampler(tile_transforms,
                                                tile_sizes_zyx,
                                                tile_aabbs,
                                                output_volume_size,
                                                output_volume_origin,
                                                CPU_CELL_SIZE,
                                                output_params.chunksize[2:],
                                                tile_layout,
                                                traverse_overlap = True,
                                                stride=volume_sampler_stride,
                                                start=volume_sampler_start)

    batch_start = time.time()
    total_cells = len(overlap_volume_sampler)
    # Reduce batch size to prevent OOM issues
    batch_size = min(batch_size, 10)  # Cap at 10 to prevent memory issues
    LOGGER.info(f"CPU Cell Size: {CPU_CELL_SIZE}")
    LOGGER.info(f"CPU Total Cells: {total_cells}")
    LOGGER.info(f"CPU Batch Size: {batch_size} (reduced to prevent OOM)")

    """
    RAY INITIALIZATION START
    """
    # Initialize Ray locally
    try:
        # View dashboard at http://localhost:8265/
        ray.init(
            ignore_reinit_error=True,
            dashboard_host='0.0.0.0',  # Allow external access
            dashboard_port=8265
        )
        # ray.init(ignore_reinit_error=True)
        LOGGER.info("Ray initialized successfully")
    except Exception as e:
        LOGGER.error(f"Failed to initialize Ray: {e}")
        raise
    """
    RAY INITIALIZATION END
    """

    """
    RAY START
    """
    try:
        # Create list of Ray remote tasks
        ray_jobs = []
        for i, (curr_cell, src_ids) in enumerate(overlap_volume_sampler):
            LOGGER.info(f"Processing cell {i+1}/{total_cells} - Cell AABB: {curr_cell}, Source IDs: {src_ids}")
            
            ray_job = cpu_fusion_ray.remote(tile_arrays,
                                           tile_transforms,
                                           tile_sizes_zyx,
                                           tile_aabbs,
                                           output_volume_size,
                                           output_volume_origin,
                                           output_volume,
                                           blend_module,
                                           curr_cell,
                                           src_ids
                                           )
            ray_jobs.append(ray_job)
            
            # Log Ray job status
            LOGGER.info(f"Created Ray job {len(ray_jobs)} for cell {i+1}/{total_cells} - Job ID: {ray_job}")

            # Process in batches
            if len(ray_jobs) == batch_size:
                LOGGER.info(f"CPU: Starting batch processing for cells {i-batch_size+2} to {i+1}/{total_cells} ({len(ray_jobs)} jobs)")
                try:
                    # Wait for batch completion
                    LOGGER.info(f"Waiting for {len(ray_jobs)} Ray jobs to complete...")
                    results = ray.get(ray_jobs)
                    successful_jobs = sum(1 for result in results if result is True)
                    failed_jobs = len(results) - successful_jobs
                    LOGGER.info(
                        f"CPU: Batch completed - {successful_jobs} successful, {failed_jobs} failed. "
                        f"Cells {i-batch_size+2} to {i+1}/{total_cells}. Batch time: {time.time() - batch_start:.2f}s"
                    )
                except Exception as e:
                    LOGGER.error(f"Error processing batch ending at {i+1}/{total_cells}: {e}")
                    # Continue with next batch even if this one failed
                finally:
                    ray_jobs = []
                    # Force garbage collection after each batch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    batch_start = time.time()

        # Process remaining cells
        if ray_jobs:
            LOGGER.info(f"CPU: Processing final batch of {len(ray_jobs)} remaining cells...")
            try:
                LOGGER.info(f"Waiting for final {len(ray_jobs)} Ray jobs to complete...")
                results = ray.get(ray_jobs)
                successful_jobs = sum(1 for result in results if result is True)
                failed_jobs = len(results) - successful_jobs
                LOGGER.info(
                    f"CPU: Final batch completed - {successful_jobs} successful, {failed_jobs} failed. "
                    f"Final batch time: {time.time() - batch_start:.2f}s"
                )
            except Exception as e:
                LOGGER.error(f"Error processing final batch: {e}")
            finally:
                ray_jobs = []
                # Force garbage collection after final batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        LOGGER.info("✅ CPU fusion processing complete.")

    except Exception as e:
        LOGGER.error(f"❌ CPU fusion error: {e}")
        raise

    finally:
        # Cleanup Ray
        try:
            ray.shutdown()
            LOGGER.info("Ray shutdown complete")
        except Exception as e:
            LOGGER.error(f"Error during Ray shutdown: {e}")
    """
    RAY END
    """

    # ============================================================================
    # GPU PROCESS CLEANUP - COMMENTED OUT (CPU-ONLY MODE)
    # ============================================================================
    # NOTE: GPU process was never started, so these lines cause errors.
    # Commenting out to prevent "can only join a started process" error.
    # ============================================================================
    
    # p.join()  # COMMENTED OUT: GPU process never started
    # p.close()  # COMMENTED OUT: GPU process never started


def cpu_fusion(
    tile_arrays: dict[int, input_output.InputArray],
    tile_transforms: dict[int, list[geometry.Transform]],
    tile_sizes_zyx: dict[int, tuple[int, int, int]],
    tile_aabbs: dict[int, geometry.AABB],
    output_volume_size: tuple[int, int, int],
    output_volume_origin: tuple[float, float, float],
    output_volume: input_output.OutputArray,
    blend_module: blend.BlendingModule,
    cell_aabb: geometry.AABB,
    src_ids: list[int]
):
    overlap_contributions: list[torch.Tensor] = []
    for t_id in src_ids:
        # Retrieve source image
        image_slice: tuple[slice, slice, slice, slice, slice] = \
                utils.calculate_image_crop(cell_aabb,
                                            output_volume_origin,
                                            tile_transforms[t_id],
                                            tile_sizes_zyx[t_id],
                                            device='cpu')
        src_img = tile_arrays[t_id][image_slice]
        src_tensor = torch.Tensor(src_img.astype(np.int16))

        # Calculate sample field
        sample_field = \
            utils.calculate_sample_field(cell_aabb,
                                        output_volume_origin,
                                        tile_transforms[t_id],
                                        tile_sizes_zyx[t_id],
                                        device='cpu')

        # Perform interpolation
        contribution = utils.interpolate(src_tensor,
                                         sample_field,
                                         device="cpu")

        overlap_contributions.append(contribution)

    # Perform blending
    blended_cell = blend_module.blend(overlap_contributions,
                                    device='cpu',
                                    kwargs={
                                    "chunk_tile_ids": src_ids,
                                    "cell_box": cell_aabb
                                    })

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
    output_volume[output_slice] = output_chunk


"""
RAY REMOTE FUNCTION START
"""
@ray.remote(num_cpus=1, memory=2 * 1024 * 1024 * 1024)  # 2GB memory limit per task
def cpu_fusion_ray(
    tile_arrays: dict[int, input_output.InputArray],
    tile_transforms: dict[int, list[geometry.Transform]],
    tile_sizes_zyx: dict[int, tuple[int, int, int]],
    tile_aabbs: dict[int, geometry.AABB],
    output_volume_size: tuple[int, int, int],
    output_volume_origin: tuple[float, float, float],
    output_volume: input_output.OutputArray,
    blend_module: blend.BlendingModule,
    cell_aabb: geometry.AABB,
    src_ids: list[int]
):
    """Ray remote version of cpu_fusion function."""
    try:
        overlap_contributions: list[torch.Tensor] = []
        for t_id in src_ids:
            # Retrieve source image
            image_slice: tuple[slice, slice, slice, slice, slice] = \
                    utils.calculate_image_crop(cell_aabb,
                                                output_volume_origin,
                                                tile_transforms[t_id],
                                                tile_sizes_zyx[t_id],
                                                device='cpu')
            src_img = tile_arrays[t_id][image_slice]
            src_tensor = torch.Tensor(src_img.astype(np.int16))

            # Calculate sample field
            sample_field = \
                utils.calculate_sample_field(cell_aabb,
                                            output_volume_origin,
                                            tile_transforms[t_id],
                                            tile_sizes_zyx[t_id],
                                            device='cpu')

            # Perform interpolation
            contribution = utils.interpolate(src_tensor,
                                             sample_field,
                                             device="cpu")

            overlap_contributions.append(contribution)

        # Perform blending
        blended_cell = blend_module.blend(overlap_contributions,
                                        device='cpu',
                                        kwargs={
                                        "chunk_tile_ids": src_ids,
                                        "cell_box": cell_aabb
                                        })

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
        output_volume[output_slice] = output_chunk
        
        # Clean up memory
        del overlap_contributions
        del blended_cell
        del output_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True  # Success indicator
        
    except Exception as e:
        print(f"Error in cpu_fusion_ray for cell {cell_aabb}: {e}")
        return False  # Failure indicator
"""
RAY REMOTE FUNCTION END
"""


# ============================================================================
# GPU FUSION FUNCTION - COMMENTED OUT (CPU-ONLY MODE)
# ============================================================================
# NOTE: This function is not used in CPU-only mode. Commenting out to avoid
# confusion and reduce code complexity.
# ============================================================================

# # No blending
# def gpu_fusion(
#     input_s3_path: str,
#     xml_path: str,
#     channel_num: int,
#     output_params: input_output.OutputParameters,
#     tile_layout: list[list[int]],
#     output_volume: input_output.OutputArray,
#     cell_size: tuple[int, int, int],
#     volume_sampler_stride: int,
#     volume_sampler_start: int,
#     datastore: int = 0,
# ):
#     """
#     NOTE:
#     ONLY INTERPOLATION, NO BLENDING.
#     Only intended to be used on non-overlap regions
#     for ultra-fast interpolation.
#     """
# 
#     dataset = input_output.BigStitcherDatasetChannel(xml_path, 
#                                            input_s3_path, 
#                                            channel_num, 
#                                            datastore=datastore)
#     a, b, c, d, e, f = initialize_fusion(dataset, output_params)
#     tile_arrays = a
#     tile_transforms = b
#     tile_sizes_zyx = c
#     tile_aabbs = d
#     output_volume_size = e
#     output_volume_origin = f
# 
#     dataset = CloudDataset(tile_arrays,
#                             tile_transforms,
#                             tile_sizes_zyx,
#                             tile_aabbs,
#                             output_volume_size,
#                             output_volume_origin,
#                             cell_size)
# 
#     volume_sampler = FusionVolumeSampler(tile_transforms,
#                                         tile_sizes_zyx,
#                                         tile_aabbs,
#                                         output_volume_size,
#                                         output_volume_origin,
#                                         cell_size,
#                                         output_params.chunksize[2:],
#                                         tile_layout,
#                                         traverse_overlap = False,
#                                         stride = volume_sampler_stride,
#                                         start = volume_sampler_start)
# 
#     cloud_dataloader = cq.CloudDataloader(dataset,
#                                           volume_sampler,
#                                           num_workers=3)
# 
#     batch_start = time.time()
#     total_cells = len(volume_sampler)
#     batch_size = 40
#     print(f'GPU Cell Size: {cell_size}')
#     print(f'GPU Total cells: {total_cells}')
#     print(f'GPU Batch size: {batch_size}')
# 
#     for i, (cell_aabb, src_ids, src_tensors) in enumerate(cloud_dataloader):
#         # Extract only tensor in src tensors
#         t_id = src_ids[0]
#         src_cell = src_tensors[0]
# 
#         # Interpolation on first GPU
#         sample_field = \
#         utils.calculate_sample_field(cell_aabb,
#                                     output_volume_origin,
#                                     tile_transforms[t_id],
#                                     tile_sizes_zyx[t_id],
#                                     device='cuda:0')
#         interpolated_cell = utils.interpolate(src_cell,
#                                         sample_field,
#                                         device='cuda:0')
# 
#         # Write
#         output_slice = (
#                 slice(0, 1),
#                 slice(0, 1),
#                 slice(cell_aabb[0], cell_aabb[1]),
#                 slice(cell_aabb[2], cell_aabb[3]),
#                 slice(cell_aabb[4], cell_aabb[5]),
#             )
# 
#         # Convert from float16 -> canonical uint16
#         output_chunk = np.array(interpolated_cell.cpu()).astype(np.uint16)
#         output_volume[output_slice] = output_chunk
# 
#         if i % batch_size == 0:
#             print(f"GPU: Finished up to {i}/{total_cells}. Batch time: {time.time() - batch_start}")
#             batch_start = time.time()
# 
#     print(f"GPU: Finished up to {i}/{total_cells}. Batch time: {time.time() - batch_start}")



# ============================================================================
# GPU CELL SIZE CALCULATION - COMMENTED OUT (CPU-ONLY MODE)
# ============================================================================
# NOTE: This function is not used in CPU-only mode. Commenting out to avoid
# confusion and reduce code complexity.
# ============================================================================

# def calculate_gpu_cell_size(
#     output_volume_size: tuple[int, int, int]
# ) -> tuple[int, int, int]:
#     """
#     Heuristic lookup table for 16 GB GPU.
#     Cell sizes are fit to canonical (128, 128, 128) chunk size.
#     """
# 
#     gpu_cell_sizes = {1024: (1024, 512, 512),  # Good for exaspim
#                       512: (512, 640, 640),    # Good smartspim
#                       384: (384, 768, 768)}    # Good for dispim
#     closest_key = min(gpu_cell_sizes.keys(), key=lambda k: abs(k - output_volume_size[0]))
# 
#     return gpu_cell_sizes[closest_key]


class CloudDataset(Dataset):
    def __init__(
        self,
        tile_arrays: dict[int, input_output.InputArray],
        tile_transforms: dict[int, list[geometry.Transform]],
        tile_sizes_zyx: dict[int, tuple[int, int, int]],
        tile_aabbs: dict[int, geometry.AABB],
        output_volume_size: tuple[int, int, int],
        output_volume_origin: tuple[float, float, float],
        cell_size: tuple[int, int, int],
        pin_memory: bool=True
        ) -> None:
        """
        Input fields are produced from
        fusion.initalize_fusion(..)

        Following codebase convention,
        input 3-ples are expected in zyx order.
        """

        # Store input arguments
        self.tile_arrays: dict[int, input_output.InputArray] = tile_arrays
        self.tile_transforms: dict[int, list[geometry.Transform]] = tile_transforms
        self.tile_sizes_zyx: dict[int, tuple[int, int, int]] = tile_sizes_zyx
        self.tile_aabbs: dict[int, geometry.AABB] = tile_aabbs
        self.output_volume_size: tuple[int, int, int] = output_volume_size
        self.output_volume_origin: tuple[float, float, float] = output_volume_origin
        self.cell_size: tuple[int, int, int] = cell_size
        self.pin_memory: bool = pin_memory

    def __getitem__(self, input_bundle):
        """
        Return src_tensor associated with the
        input cell_aabb/t_id.
        """

        cell_aabb, src_ids = input_bundle

        src_tensors: list[torch.Tensor] = []
        for t_id in src_ids:
            image_slice: tuple[slice, slice, slice, slice, slice] = \
            utils.calculate_image_crop(cell_aabb,
                                        self.output_volume_origin,
                                        self.tile_transforms[t_id],
                                        self.tile_sizes_zyx[t_id],
                                        device='cpu')

            result = self.tile_arrays[t_id][image_slice]

            # uint16 -> int16 for pytorch compatibility.
            # Max intensity values of original data are close to 1000,
            # no where near 1/2 uint16 (32,767), so this is safe.
            if self.pin_memory:
                result = torch.Tensor(result.astype(np.int16)).pin_memory()
            else:
                result = torch.Tensor(result.astype(np.int16))

            src_tensors.append(result)

        return cell_aabb, src_ids, src_tensors

    def __len__(self):
        z_cnt, y_cnt, x_cnt = \
            get_cell_count_zyx(self.output_volume_size, self.cell_size)
        total_cells = z_cnt * y_cnt * x_cnt
        return total_cells


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

    def _check_true_collision(
        self,
        cell_box: geometry.AABB,
        transform_list: list[geometry.Transform],
        src_vol_shape_zyx: tuple[int, int, int]
    ) -> bool:
        # Build the box
        z_min, z_max, y_min, y_max, x_min, x_max = cell_box
        z_grid, y_grid, x_grid = torch.meshgrid(
            torch.Tensor([z_min + 0.5, z_max - 0.5]),
            torch.Tensor([y_min + 0.5, y_max - 0.5]),
            torch.Tensor([x_min + 0.5, x_max - 0.5]),
            indexing='ij'
        )
        cell_box_pts = torch.stack([z_grid, y_grid, x_grid], dim=-1)

        # Apply inverse transform
        cell_box_pts = cell_box_pts + torch.Tensor(self.output_volume_origin)
        for tfm in reversed(transform_list):
            cell_box_pts = tfm.backward(cell_box_pts, device='cpu')

        # Check collision
        cell_box_src: geometry.AABB = geometry.aabb_3d(cell_box_pts)
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
