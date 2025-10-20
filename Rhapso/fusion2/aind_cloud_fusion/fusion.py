"""Core fusion algorithm."""
import os
import logging
import time

import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
import numpy as np
import torch
import s3fs
import zarr
import tensorstore as ts

import aind_cloud_fusion.blend as blend
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.io as io


def initialize_fusion(
    dataset: io.Dataset,
    post_reg_tfms: list[geometry.Transform],
    output_params: io.OutputParameters,
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
    tile_sizes: Dictionary of tile sizes
    tile_aabbs: Dictionary of AABB of each transformed tile
    output_volume_size: Size of output volume
    output_volume_origin: Location of output volume
    """

    tile_arrays: dict[int, io.InputArray] = dataset.tile_volumes_tczyx

    tile_transforms: dict[
        int, list[geometry.Transform]
    ] = dataset.tile_transforms_zyx
    input_resolution_zyx: tuple[
        float, float, float
    ] = dataset.tile_resolution_zyx
    iz, iy, ix = input_resolution_zyx
    scale_input_zyx = geometry.Affine(
        np.array([[iz, 0, 0, 0], [0, iy, 0, 0], [0, 0, ix, 0]])
    )

    output_resolution_zyx: tuple[
        float, float, float
    ] = output_params.resolution_zyx
    oz, oy, ox = output_resolution_zyx
    sample_output_zyx = geometry.Affine(
        np.array([[1 / oz, 0, 0, 0], [0, 1 / oy, 0, 0], [0, 0, 1 / ox, 0]])
    )
    for tile_id, tfm_list in tile_transforms.items():
        tile_transforms[tile_id] = [
            *tfm_list,
            scale_input_zyx,
            *post_reg_tfms,
            sample_output_zyx,
        ]

    tile_sizes_zyx: dict[int, tuple[int, int, int]] = {}
    tile_aabbs: dict[int, geometry.AABB] = {}
    tile_boundary_point_cloud_zyx = []

    for tile_id, tile_arr in tile_arrays.items():
        tile_sizes_zyx[tile_id] = zyx = tile_arr.shape[2:]
        tile_sizes_zyx[tile_id] = zyx = tile_arr.shape[2:]
        tile_boundaries = torch.Tensor(
            [
                [0.0, 0.0, 0.0],
                [zyx[0], 0.0, 0.0],
                [0.0, zyx[1], 0.0],
                [0.0, 0.0, zyx[2]],
                [zyx[0], zyx[1], 0.0],
                [zyx[0], 0.0, zyx[2]],
                [0.0, zyx[1], zyx[2]],
                [zyx[0], zyx[1], zyx[2]],
            ]
        )

        tfm_list = tile_transforms[tile_id]
        for i, tfm in enumerate(tfm_list):
            tile_boundaries = tfm.forward(
                tile_boundaries, device=torch.device("cpu")
            )

        tile_aabbs[tile_id] = geometry.aabb_3d(tile_boundaries)
        tile_boundary_point_cloud_zyx.extend(tile_boundaries)
    tile_boundary_point_cloud_zyx = torch.stack(
        tile_boundary_point_cloud_zyx, dim=0
    )

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

    OUTPUT_VOLUME_ORIGIN = (
        torch.min(tile_boundary_point_cloud_zyx[:, 0]).item(),
        torch.min(tile_boundary_point_cloud_zyx[:, 1]).item(),
        torch.min(tile_boundary_point_cloud_zyx[:, 2]).item(),
    )

    # Shift AABB's into Output Volume where
    # absolute position of output volume is moved to (0, 0, 0)
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
    output_params: io.OutputParameters,
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
    if output_params.path.startswith('s3'):
        s3 = s3fs.S3FileSystem(
            config_kwargs={
                'max_pool_connections': 50,
                's3': {
                'multipart_threshold': 64 * 1024 * 1024,  # 64 MB, avoid multipart upload for small chunks
                'max_concurrent_requests': 20  # Increased from 10 -> 20.
                },
                'retries': {
                'total_max_attempts': 100,
                'mode': 'adaptive',
                }
            }
        )
        store = s3fs.S3Map(root=output_params.path, s3=s3)
        out_group = zarr.open(store=store, mode='a')

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
    output_params: io.OutputParameters,
    output_volume_size: tuple[int, int, int],
):
    """
    The output is an async Tensorstore obj that you need
    to call .result() to perform a write.
    """
    parts = output_params.path.split('/')
    bucket = parts[2]
    path = '/'.join(parts[3:])
    chunksize = list(output_params.chunksize)
    output_shape = [1,
                    1,
                    output_volume_size[0],
                    output_volume_size[1],
                    output_volume_size[2]]

    return ts.open({
        'driver': 'zarr',
        'dtype': 'uint16',
        'kvstore' : {
            'driver': 's3',
            'bucket': bucket,
            'path': path,
        },
        'create': True,
        'open': True,
        'metadata': {
            'chunks': chunksize,
            'compressor': {
                'blocksize': 0,
                'clevel': 1,
                'cname': 'zstd',
                'id': 'blosc',
                'shuffle': 1,
            },
            'dimension_separator': '/',
            'dtype': '<u2',
            'fill_value': 0,
            'filters': None,
            'order': 'C',
            'shape': output_shape,
            'zarr_format': 2
        }
    }).result()


def initialize_output_volume(
    output_params: io.OutputParameters,
    output_volume_size: tuple[int, int, int],
) -> io.OutputArray:
    output = None
    
    assert output_params.datastore in [0, 1], \
        f"Only 0 = Dask and 1 = Tensorstore supported."
    if output_params.datastore == 0:
        output = initialize_output_volume_dask(output_params, output_volume_size)
    elif output_params.datastore == 1:
        output = initialize_output_volume_tensorstore(output_params, output_volume_size)
    return output


def get_cell_count_zyx(
    output_volume_size: tuple[int, int, int], cell_size: tuple[int, int, int]
) -> tuple[int, int, int]:
    """
    Total amount of z,y, and x cells returned in that order.
    Input sizes are in canonical zyx order.
    """
    z_cnt = int(np.ceil(output_volume_size[0] / cell_size[0]))
    y_cnt = int(np.ceil(output_volume_size[1] / cell_size[1]))
    x_cnt = int(np.ceil(output_volume_size[2] / cell_size[2]))

    return z_cnt, y_cnt, x_cnt


def run_fusion(
    # client,    # Uncomment for testing in jupyterlab
    dataset: io.Dataset,
    output_params: io.OutputParameters,
    runtime_params: io.RuntimeParameters,
    cell_size: tuple[int, int, int],
    post_reg_tfms: list[geometry.Affine],
    blend_module: blend.BlendingModule,
):
    """
    Fusion algorithm.
    Inputs: Application objs initalized from input configurations.
    Output: Writes to location in output params.
    """

    logging.basicConfig(
        format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M"
    )
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)

    a, b, c, d, e, f = initialize_fusion(dataset, post_reg_tfms, output_params)
    tile_arrays = a
    tile_transforms = b
    tile_sizes_zyx = c
    tile_aabbs = d
    output_volume_size = e
    output_volume_origin = f  # Temp variables to meet line character maximum.

    output_volume = initialize_output_volume(output_params, output_volume_size)

    LOGGER.info(f"Number of Tiles: {len(tile_arrays)}")
    LOGGER.info(f"{output_volume_size=}")

    # Vanilla Processing
    if runtime_params.option == 0:
        start_run = time.time()

        process_args: list[dict] = []
        num_cells = len(runtime_params.worker_cells)
        for cell_num, cell in enumerate(runtime_params.worker_cells):
            z, y, x = cell
            process_args.append({"cell_num": cell_num, "z": z, "y": y, "x": x})

        for p_args in process_args:
            LOGGER.info(f'Starting Cell {p_args["cell_num"]}/{num_cells}')
            start_time = time.time()
            color_cell(tile_arrays,
                    tile_transforms,
                    tile_sizes_zyx,
                    tile_aabbs,
                    output_volume,
                    output_volume_origin,
                    cell_size,
                    blend_module,
                    p_args["z"],
                    p_args["y"],
                    p_args["x"],
                    torch.device('cpu'),
                    LOGGER,
            )
            LOGGER.info(
                f"Finished Cell {p_args['cell_num']}/{num_cells}: {time.time() - start_time}"
            )
        LOGGER.info(f"Runtime: {time.time() - start_run}")

    # Multi Processing
    elif runtime_params.option == 1:
        start_run = time.time()

        process_args: list[dict] = []
        num_cells = len(runtime_params.worker_cells)
        for cell_num, cell in enumerate(runtime_params.worker_cells):
            z, y, x = cell
            process_args.append({"cell_num": cell_num, "z": z, "y": y, "x": x})

        # Important for prevent running out of resources
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        # Run Fusion: Fill work queue (active processes) with inital tasks
        # Task-specific info includes process_args and device.
        start_run = time.time()
        active_processes: list[
            tuple
        ] = []  # (process, device, cell_num, start_time)
        for i in range(runtime_params.pool_size):
            p_args = process_args.pop(0)
            p = torch.multiprocessing.Process(
                target=color_cell,
                args=(
                    tile_arrays,
                    tile_transforms,
                    tile_sizes_zyx,
                    tile_aabbs,
                    output_volume,
                    output_volume_origin,
                    cell_size,
                    blend_module,
                    p_args["z"],
                    p_args["y"],
                    p_args["x"],
                    torch.device('cpu'),
                    LOGGER,
                ),
            )
            active_processes.append(
                (
                    p,
                    torch.device('cpu'),
                    p_args["cell_num"],
                    time.time(),
                )
            )
            LOGGER.info(f'Starting Cell {p_args["cell_num"]}/{num_cells}')
            p.start()

        # Run Fusion: Exhaust all the tasks
        # Tasks all implictly defined in process_args buffer
        while len(active_processes) != 0:
            tmp = []
            for p, device, cell_num, start_time in active_processes:
                p.join(
                    timeout=0
                )  # timeout indicates do not wait until the process is explicitly finished

                if p.is_alive():
                    tmp.append((p, device, cell_num, start_time))
                else:
                    p.close()
                    del p
                    LOGGER.info(
                        f"Finished Cell {cell_num}/{num_cells}: {time.time() - start_time}"
                    )

                    if len(process_args) != 0:
                        p_args = process_args.pop(0)
                        new_p = torch.multiprocessing.Process(
                            target=color_cell,
                            args=(
                                tile_arrays,
                                tile_transforms,
                                tile_sizes_zyx,
                                tile_aabbs,
                                output_volume,
                                output_volume_origin,
                                cell_size,
                                blend_module,
                                p_args["z"],
                                p_args["y"],
                                p_args["x"],
                                device,
                                LOGGER,
                            ),
                        )
                        tmp.append(
                            (new_p, device, p_args["cell_num"], time.time())
                        )
                        LOGGER.info(
                            f'Starting Cell {p_args["cell_num"]}/{num_cells}'
                        )
                        new_p.start()

                active_processes = tmp

        LOGGER.info(f"Runtime: {time.time() - start_run}")

    # Dask
    elif runtime_params.option == 2:
        start_run = time.time()

        batch_start = start_run
        client = Client(LocalCluster(n_workers=runtime_params.pool_size, threads_per_worker=1, processes=True))

        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        num_cells = len(runtime_params.worker_cells)
        batch_size = 1000   # Good batch size for 128^3.
        LOGGER.info(f'Coloring {num_cells} cells')
        LOGGER.info(f'Batch Size: {batch_size}')

        delayed_color_cells = []
        for cell_num, cell in enumerate(runtime_params.worker_cells):
            z, y, x = cell
            delayed_color_cells.append(
                dask.delayed(color_cell, pure=True)(
                    tile_arrays,
                    tile_transforms,
                    tile_sizes_zyx,
                    tile_aabbs,
                    output_volume,
                    output_volume_origin,
                    cell_size,
                    blend_module,
                    z,
                    y,
                    x,
                    torch.device("cpu"),  # Hardcoding CPU cluster
                    LOGGER
                )
            )
            # Batching
            if len(delayed_color_cells) == batch_size:
                LOGGER.info(f'Calculating up to {cell_num}/{num_cells}...')
                da.compute(*delayed_color_cells)

                # Clear memory for runtime stability
                delayed_color_cells = []
                # client.restart()
                # ^ Remove restart command, dask bug

                LOGGER.info(f'Finished up to {cell_num}/{num_cells}. Batch time: {time.time() - batch_start}')
                batch_start = time.time()

        # Compute remaining cells
        LOGGER.info(f'Calculating up to {num_cells}/{num_cells}...')
        da.compute(*delayed_color_cells)
        delayed_color_cells = []
        LOGGER.info(f'Finished up to {num_cells}/{num_cells}. Batch time: {time.time() - batch_start}')
        batch_start = time.time()

        LOGGER.info(f"Runtime: {time.time() - start_run}")


def color_cell(
    tile_arrays: dict[int, io.InputArray],
    tile_transforms: dict[int, list[geometry.Transform]],
    tile_sizes_zyx: dict[int, tuple[int, int, int]],
    tile_aabbs: dict[int, geometry.AABB],
    output_volume: io.OutputArray,
    output_volume_origin: tuple[float, float, float],
    cell_size: tuple[int, int, int],
    blend_module: blend.BlendingModule,
    z: int,
    y: int,
    x: int,
    device: torch.device,
    logger: logging.Logger  # Depreciated
):
    """
    Parallelized function called in fusion.

    Inputs
    -------
    tile_arrays: Dictionary of input tile arrays
    tile_transforms: Dictionary of (list of) registrations associated with each tile
    tile_sizes_zyx: Dictionary of tile sizes
    tile_aabbs_zyx: Dictionary of AABB of each transformed tile
    output_volume: Zarr store parallel functions write to
    output_volume_origin: Location of output volume
    cell_size: operating volume of this function
    blend_module: application blending obj
    z, y, x: location of cell in terms of output volume indices
    """

    # LOGGER = logger

    # Cell Boundaries, exclusive stop index
    output_volume_size = output_volume.shape
    cell_box = np.array(
        [
            [z * cell_size[0], z * cell_size[0] + cell_size[0]],
            [y * cell_size[1], y * cell_size[1] + cell_size[1]],
            [x * cell_size[2], x * cell_size[2] + cell_size[2]],
        ]
    )
    cell_box[:, 1] = np.minimum(
        cell_box[:, 1], np.array(output_volume_size[2:])
    )

    cell_box = cell_box.flatten()

    # Collision Detection
    # Collision defined by overlapping intervals in all 3 dimensions.
    # Two intervals (A, B) collide if A_max is not <= B_min and A_min is not >= B_max.
    overlapping_tiles: list[int] = []
    for tile_id, t_aabb in tile_aabbs.items():
        if (
            (cell_box[1] > t_aabb[0] and cell_box[0] < t_aabb[1])
            and (cell_box[3] > t_aabb[2] and cell_box[2] < t_aabb[3])
            and (cell_box[5] > t_aabb[4] and cell_box[4] < t_aabb[5])
        ):
            overlapping_tiles.append(tile_id)

    # LOGGER.info(f"{cell_box=}")
    # LOGGER.info(f"{overlapping_tiles=}")

    # Interpolation for cell_contributions
    cell_contributions: list[torch.Tensor] = []
    cell_contribution_tile_ids: list[int] = []
    for tile_id in overlapping_tiles:
        # Init tile coords, arange end-exclusive, +0.5 to represent voxel center
        z_indices = torch.arange(cell_box[0], cell_box[1], step=1) + 0.5
        y_indices = torch.arange(cell_box[2], cell_box[3], step=1) + 0.5
        x_indices = torch.arange(cell_box[4], cell_box[5], step=1) + 0.5
        z_indices = z_indices.to(device)
        y_indices = y_indices.to(device)
        x_indices = x_indices.to(device)

        z_grid, y_grid, x_grid = torch.meshgrid(
            z_indices, y_indices, x_indices, indexing="ij"
        )
        z_grid = torch.unsqueeze(z_grid, 0)
        y_grid = torch.unsqueeze(y_grid, 0)
        x_grid = torch.unsqueeze(x_grid, 0)

        tile_coords = torch.concatenate((z_grid, y_grid, x_grid), axis=0)
        # (3, z, y, x) -> (z, y, x, 3)
        tile_coords = torch.movedim(tile_coords, source=0, destination=3)

        # Define tile coords wrt output vol origin
        tile_coords = tile_coords + torch.Tensor(output_volume_origin).to(
            device
        )

        # Send tile_coords through inverse transforms
        # NOTE: tile_transforms list must be iterated thru in reverse
        # (z, y, x, 3) -> (z, y, x, 3)
        for tfm in reversed(tile_transforms[tile_id]):
            tile_coords = tfm.backward(tile_coords, device=device)

        # Calculate AABB of transformed coords
        z_min, z_max, y_min, y_max, x_min, x_max = geometry.aabb_3d(
            tile_coords
        )

        # Mini Optimization: Check true collision before executing interpolation/fusion
        # That is, aabb of transformed coordinates into imagespace actually overlap the image.
        t_size_zyx = tile_sizes_zyx[tile_id]
        if not (
            (z_max > 0 and z_min < t_size_zyx[0])
            and (y_max > 0 and y_min < t_size_zyx[1])
            and (x_max > 0 and x_min < t_size_zyx[2])
        ):
            continue

        # Calculate overlapping region between transformed coords and image boundary
        # For intervals (A, B):
        # The lower bound of overlapping region = max(A_min, B_min)
        # The upper bound of overlapping region = min(A_max, B_max)
        crop_min_z = torch.max(torch.Tensor([0, z_min]))
        crop_max_z = torch.min(torch.Tensor([t_size_zyx[0], z_max]))

        crop_min_y = torch.max(torch.Tensor([0, y_min]))
        crop_max_y = torch.min(torch.Tensor([t_size_zyx[1], y_max]))

        crop_min_x = torch.max(torch.Tensor([0, x_min]))
        crop_max_x = torch.min(torch.Tensor([t_size_zyx[2], x_max]))

        # Make sure crop_{min, max}_{z, y, x} are integers to be used as indices.
        # Minimum values are rounded down to nearest integer.
        # Maximum values are rounded up to nearest integer.
        crop_min_z = int(torch.floor(crop_min_z))
        crop_min_y = int(torch.floor(crop_min_y))
        crop_min_x = int(torch.floor(crop_min_x))

        crop_max_z = int(torch.ceil(crop_max_z))
        crop_max_y = int(torch.ceil(crop_max_y))
        crop_max_x = int(torch.ceil(crop_max_x))

        # LOGGER.info(
        #     f"""AABB of Crop belonging to image {tile_id}:
        # {crop_min_z}, {crop_max_z}, {crop_min_y}, {crop_max_y}, {crop_min_x}, {crop_max_x}"""
        # )

        # Define tile coords wrt base image crop coordinates
        image_crop_offset = torch.Tensor(
            [crop_min_z, crop_min_y, crop_min_x]
        ).to(device)
        tile_coords = tile_coords - image_crop_offset

        # Prep inputs to interpolation
        image_crop_slice = (
            0,
            0,
            slice(crop_min_z, crop_max_z),
            slice(crop_min_y, crop_max_y),
            slice(crop_min_x, crop_max_x),
        )

        image_crop = tile_arrays[tile_id][image_crop_slice]
        if isinstance(image_crop, da.Array):
            image_crop = image_crop.compute()

        image_crop = image_crop.astype(
            np.int32
        )  # Promote uint16 -> Pytorch compatible int32
        image_crop = torch.Tensor(image_crop).to(device)

        # Pytorch flow field follows a different basis than the image numpy basis.
        # Change of basis to interpolation basis, which preserves relative distances/angles/positions.
        # (z, y, x, 3) -> (z, y, x, 3)
        interp_cob_matrix = torch.Tensor(
            [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
        )
        interp_cob = geometry.Affine(interp_cob_matrix)
        tile_coords = interp_cob.forward(tile_coords, device=device)

        # Interpolation expects 'grid' parameter/sample locations to be normalized [-1, 1].
        # Very specific per-dimension normalization according to CoB
        crop_z_length = crop_max_z - crop_min_z
        crop_y_length = crop_max_y - crop_min_y
        crop_x_length = crop_max_x - crop_min_x
        tile_coords[:, :, :, 0] = (
            tile_coords[:, :, :, 0] - (crop_x_length / 2)
        ) / (crop_x_length / 2)
        tile_coords[:, :, :, 1] = (
            tile_coords[:, :, :, 1] - (crop_y_length / 2)
        ) / (crop_y_length / 2)
        tile_coords[:, :, :, 2] = (
            tile_coords[:, :, :, 2] - (crop_z_length / 2)
        ) / (crop_z_length / 2)

        # Final reshaping
        # image_crop: (z_in, y_in, x_in) -> (1, 1, z_in, y_in, x_in)
        # tile_coords: (z_out, y_out, x_out, 3) -> (1, z_out, y_out, x_out, 3)
        # => tile_contribution: (1, 1, z_out, y_out, x_out)
        image_crop = image_crop[(None,) * 2]
        tile_coords = torch.unsqueeze(tile_coords, 0)

        # Interpolate and Store
        tile_contribution = torch.nn.functional.grid_sample(
            image_crop, tile_coords, padding_mode="zeros", mode="nearest", align_corners=False
        )

        cell_contributions.append(tile_contribution)
        cell_contribution_tile_ids.append(tile_id)

        del tile_coords

    # Fuse all cell contributions together with specified blend module
    fused_cell = torch.zeros((1,
                              1,
                              cell_box[1] - cell_box[0],
                              cell_box[3] - cell_box[2],
                              cell_box[5] - cell_box[4]))
    if len(cell_contributions) != 0:
        fused_cell = blend_module.blend(
            cell_contributions,
            device,
            kwargs={'chunk_tile_ids': cell_contribution_tile_ids,
                    'cell_box': cell_box}
        )
        cell_contributions = []

    # Write
    output_slice = (
        slice(0, 1),
        slice(0, 1),
        slice(cell_box[0], cell_box[1]),
        slice(cell_box[2], cell_box[3]),
        slice(cell_box[4], cell_box[5]),
    )
    # Convert from float32 -> canonical uint16
    output_chunk = np.array(fused_cell.cpu()).astype(np.uint16)
    output_volume[output_slice] = output_chunk

    del fused_cell
    del output_chunk
