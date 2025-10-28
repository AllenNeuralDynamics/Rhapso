"""
Computes radial correction in microscopic data
"""

import asyncio
import logging
import multiprocessing as mp
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import dask.array as da
import numba as nb
import numpy as np
import tensorstore as ts
from aind_data_schema.core.processing import DataProcess, ProcessName
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from natsort import natsorted
from scipy.ndimage import map_coordinates

from . import __maintainers__, __pipeline_version__, __url__, __version__
from .array_to_zarr import convert_array_to_zarr
from .utils import utils

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def calculate_corner_shift_from_pixel_size(XY_pixel_size: float) -> float:
    """
    Compute the corner shift value based on pixel size.

    Parameters
    ----------
    XY_pixel_size : float
        The size of a pixel in microns (not currently used).

    Returns
    -------
    float
        A constant value of 4.3.
    """
    return 4.3


def calculate_frac_cutoff_from_pixel_size(XY_pixel_size: float) -> float:
    """
    Compute the fractional cutoff for radial correction.

    Parameters
    ----------
    XY_pixel_size : float
        The size of a pixel in microns (not currently used).

    Returns
    -------
    float
        A constant value of 0.5.
    """
    return 0.5


# enables multithreading with prange
@nb.njit(parallel=True)
def _compute_coordinates(
    pixels: int, cutoff: float, corner_shift: float, edge: int
) -> tuple:
    """
    Computes the coordinates where the
    pixels will be moved.

    Parameters
    ----------
    pixels: int
        Width or height of the image. It is assumed that the image
        has the same resolution in XY.

    cutoff: float
        Radius beyond which distortion is applied.

    corner_shift: float
        How much to "pull in" corners beyond the cutoff.

    edge: int
        Number of pixels to crop from each side (e.g., due to interpolation instability).
    """
    # coords[0] -> y-coordinates relative to center
    # coords[1] -> x-coordinates relative to center
    coords = np.zeros((2, pixels, pixels), dtype=np.float32)

    # stores the radius of each pixel from the center.
    r = np.zeros((pixels, pixels), dtype=np.float32)

    # First pass: calculate centered coordinates and radius for
    # every pixel in the image, parallelized with prange
    for i in nb.prange(pixels):
        for j in range(pixels):
            # Shifts (i, j) so the origin is at the center.
            y = i - pixels // 2
            x = j - pixels // 2
            coords[0, i, j] = y
            coords[1, i, j] = x

            # Calculates the radius from the center.
            r[i, j] = np.sqrt(x * x + y * y)

    # Finds the maximum radius,
    # rmax, used to normalize the distortion.
    rmax = r.max()

    # Second pass: apply radial distortion
    r_piece = np.zeros_like(r)
    angles = np.zeros_like(r)

    for i in nb.prange(pixels):
        for j in range(pixels):
            r_val = r[i, j]
            # Y X angle, careful with x y
            # coords 0 is y, coords 1 is x
            # Uses arctan2(y, x) to get the angle of the pixel from the center
            angle = np.arctan2(coords[0, i, j], coords[1, i, j])

            # pixels farther from center than cutoff are pulled outward/inward
            if r_val > cutoff:
                r_val += (r_val - cutoff) * corner_shift / (rmax - cutoff)

            r_piece[i, j] = r_val
            angles[i, j] = angle
            coords[0, i, j] = r_val * np.sin(angle)
            coords[1, i, j] = r_val * np.cos(angle)

    # Crop edges and shift to image space
    cropped = coords[:, edge:-edge, edge:-edge]

    cropped += pixels // 2

    return cropped


def _process_plane(args):
    """Helper function to process a single z-plane for parallel execution"""
    z, plane, coords, order = args
    warp_coords = np.zeros((2, *coords[0].shape), dtype=np.float32)
    warp_coords[0] = coords[0]
    warp_coords[1] = coords[1]
    return z, map_coordinates(plane, warp_coords, order=order, mode="constant")


def radial_correction(
    tile_data: np.ndarray,
    corner_shift: Optional[float] = 5.5,
    frac_cutoff: Optional[float] = 0.5,
    mode: Union[Literal["2d"], Literal["3d"]] = "3d",
    order: int = 1,
    max_workers: Optional[int] = None,
) -> np.ndarray:
    """
    Apply radial correction to a tile with optimized performance.

    Parameters
    ----------
    tile_data : np.ndarray
        The 3D tile data (Z, Y, X) to be corrected.
    corner_shift : Optional[float]
        The amount of radial shift to apply (default is 5.5).
    frac_cutoff : Optional[float]
        Fraction of the radius to begin applying correction (default is 0.5).
    mode : Union[Literal["2d"], Literal["3d"]]
        Processing mode - "2d" for plane-wise processing or "3d" for full volume (default is "3d").
    order : int
        Interpolation order for map_coordinates (default is 1).
    max_workers : Optional[int]
        Maximum number of worker threads for parallel processing (default is None, which uses CPU count).

    Returns
    -------
    np.ndarray
        The corrected tile.
    """
    edge = ceil(corner_shift / np.sqrt(2)) + 1
    shape = tile_data.shape
    pixels = shape[1]  # Assume square XY plane
    cutoff = pixels * frac_cutoff

    # Compute the warp to transform coordinates using numba
    coords = _compute_coordinates(pixels, cutoff, corner_shift, edge)

    # Calculate new shape after edge cropping
    new_shape = np.array(shape) - [0, edge * 2, edge * 2]
    LOGGER.info(f"New shape: {new_shape} - Mode {mode} - Cutoff: {cutoff}")

    # Different processing methods based on mode
    if mode == "2d":
        # Process each z-plane separately in parallel
        result = np.zeros(new_shape, dtype=tile_data.dtype)  # dtype=np.uint16)

        # Use ThreadPoolExecutor for parallel processing of z-planes
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [(z, tile_data[z], coords, order) for z in range(shape[0])]
            for z, processed_plane in executor.map(
                lambda args: _process_plane(args), tasks
            ):
                # print(f"Tile data dtype: {tile_data[z].dtype} - Processed: {processed_plane.dtype}")
                result[z] = processed_plane

        return result

    else:  # 3D mode
        # Create full 3D warping coordinates array
        warp_coords = np.zeros((3, *new_shape), dtype=np.float32)

        # Z coordinates remain unchanged
        for z in range(new_shape[0]):
            warp_coords[0, z] = z

        # Apply pre-computed X-Y coordinate transformations to each z-plane
        warp_coords[1] = np.repeat(
            coords[0][np.newaxis, :, :], new_shape[0], axis=0
        )
        warp_coords[2] = np.repeat(
            coords[1][np.newaxis, :, :], new_shape[0], axis=0
        )

        # Process the entire volume at once
        return map_coordinates(
            tile_data, warp_coords, order=order, mode="constant"
        )


def read_zarr(
    dataset_path: str,
    compute: Optional[bool] = True,
) -> Tuple:
    """
    Reads a zarr dataset

    Parameters
    ----------
    dataset_path: str
        Path where the dataset is stored.

    compute: Optional[bool]
        Computes the lazy dask graph.
        Default: True

    Returns
    -------
    Tuple[ArrayLike, da.Array]
        ArrayLike or None if compute is false
        Lazy dask array
    """
    tile = None

    cluster = LocalCluster(
        n_workers=mp.cpu_count(), threads_per_worker=1, memory_limit="auto"
    )
    client = Client(cluster)

    # Explicitly setting threads to do reading (way faster)
    try:
        tile_lazy = da.from_zarr(dataset_path).squeeze()

        if compute:
            with ProgressBar():
                tile = tile_lazy.compute(scheduler="threads")
    finally:
        client.close()
        cluster.close()

    return tile, tile_lazy


async def read_zarr_tensorstore(
    dataset_path: str, scale: str, driver: Optional[str] = "zarr"
) -> Tuple:
    """
    Reads a zarr dataset from local filesystem or S3 bucket
    Parameters
    ----------
    dataset_path: str
        Path where the dataset is stored. Can be a local path or an S3 path (s3://...)
    scale: str
        Multiscale to load
    driver: Optional[str]
        Tensorstore driver
        Default: zarr
    Returns
    -------
    Tuple[ArrayLike, da.Array]
        ArrayLike or None if compute is false
        Lazy dask array
    """
    # Parse the URL properly using urllib
    parsed_url = urllib.parse.urlparse(dataset_path)

    if parsed_url.scheme == "s3":
        # Handle S3 path
        bucket = parsed_url.netloc
        # Remove leading slash if present
        key = parsed_url.path.lstrip("/")
        print(parsed_url, bucket, key)

        ts_spec = {
            "driver": str(driver),
            "kvstore": {
                "driver": "s3",
                "bucket": bucket,
                "path": key,
            },
            "path": str(scale),
        }
    else:
        # Original local file handling
        ts_spec = {
            "driver": str(driver),
            "kvstore": {
                "driver": "file",
                "path": str(dataset_path),
            },
            "path": str(scale),
        }

    tile_lazy = await ts.open(ts_spec)
    tile = await tile_lazy.read()
    return tile, tile_lazy


def apply_corr_to_zarr_tile(
    dataset_path: str,
    scale: str,
    corner_shift: Optional[float] = 5.5,
    frac_cutoff: Optional[float] = 0.5,
    z_size_threshold: Optional[int] = 400,
    order: Optional[int] = 1,
    max_workers: Optional[int] = None,
    driver: Optional[str] = "zarr",
) -> np.ndarray:
    """
    Load a Zarr tile, apply radial correction, and return corrected tile.

    Parameters
    ----------
    dataset_path : str
        Path to the Zarr file containing the tile.

    scale: str
        Multiscale to load the data

    corner_shift : Optional[float]
        The amount of shift to apply to corners (default is 5.5).

    frac_cutoff : Optional[float]
        The fractional radius where correction starts (default is 0.5).

    z_size_threshold: Optional[int]
        Threshold in which 3D radial correction is applied.

    order: Optional[int]
        Interpolation order.
        Default: 1

    max_workers: Optional[int]
        Max number of workers.
        Default: None

    driver: Optional[str]
        Zarr driver to read the data.
        Default: zarr

    Returns
    -------
    np.ndarray
        The corrected tile.
    """
    if z_size_threshold < 0:
        raise ValueError(
            f"Please, provide a correct threshold: {z_size_threshold}"
        )

    # Reading zarr dataset
    data_in_memory, lazy_array = asyncio.run(
        read_zarr_tensorstore(dataset_path, scale=scale, driver=driver)
    )
    # data_in_memory, lazy_array = read_zarr(f"{dataset_path}/{scale}", compute=True)
    data_in_memory = data_in_memory.squeeze()
    z_size = data_in_memory.shape[-3]

    output_radial = None

    LOGGER.info(f"Dataset shape {data_in_memory.shape}")

    mode = "2d"

    if z_size < z_size_threshold:
        mode = "3d"

    output_radial = radial_correction(
        tile_data=data_in_memory,
        corner_shift=corner_shift,
        frac_cutoff=frac_cutoff,
        mode=mode,
        order=order,
        max_workers=max_workers,
    )

    # print(f"input radial correction: {data_in_memory.shape} - {data_in_memory.dtype}")
    # print(f"Output radial correction: {output_radial.shape} - {output_radial.dtype}")

    return output_radial


def correct_and_save_tile(
    dataset_loc: str,
    output_path: str,
    resolution_zyx: List[float],
    scale: str = "0",
    n_lvls: Optional[int] = 4,
    driver: Optional[str] = "zarr",
):
    """
    Corrects and saves a single tile.

    Parameters
    ----------
    dataset_loc: str
        Path to the dataset to be corrected.
    output_path: str
        Path to save the corrected dataset.
    resolution_zyx: List[float]
        Voxel size in the format [z, y, x].
    scale: str
        Multiscale to load the data.
        Default: 0
    n_lvls: Optional[int]
        Number of downsampled levels to write.
        Default: 4
    s3_output_path: str
        Dataset name in S3.
        Default: None
    cloud_write: bool
        If True, write to S3.
        Default: True
    driver: Optional[str]
        Driver to read the data with tensorstore.
        Default: "zarr"
    """

    corner_shift = calculate_corner_shift_from_pixel_size(resolution_zyx[1])
    frac_cutoff = calculate_frac_cutoff_from_pixel_size(resolution_zyx[1])

    LOGGER.info(f"Input: {dataset_loc} - Output: {output_path}")
    LOGGER.info(f"Corner Shift: {corner_shift} pixels")
    LOGGER.info(f"Fraction Cutoff: {frac_cutoff}")

    start_time = time.time()
    corrected_tile = apply_corr_to_zarr_tile(
        dataset_loc, scale, corner_shift, frac_cutoff, driver=driver
    )
    end_time = time.time()
    LOGGER.info(
        f"Time to correct: {end_time - start_time} seconds -> New shape {corrected_tile.shape}"
    )

    convert_array_to_zarr(
        array=corrected_tile,
        voxel_size=resolution_zyx,
        chunk_size=[128] * 3,
        output_path=str(output_path),
        n_lvls=n_lvls,
    )

    data_process = None
    # TODO: activate this when aind-data-schema 2.0 is out
    # DataProcess(
    #     name=ProcessName.IMAGE_RADIAL_CORRECTION,
    #     software_version=__version__,
    #     start_date_time=start_time,
    #     end_date_time=end_time,
    #     input_location=dataset_loc,
    #     output_location=output_path,
    #     code_version=__version__,
    #     code_url=__url__,
    #     parameters={
    #         'corner_shift': corner_shift,
    #         'frac_cutoff': frac_cutoff
    #     },
    # )

    return data_process


def main(
    data_folder: str,
    results_folder: str,
    acquisition_path: str,
    tilenames: List[str],
    driver: Optional[str] = "zarr",
):
    """
    Radial correction to multiple tiles
    based on provided YMLs.

    Parameters
    ----------
    data_folder: str
        Folder where the data is stored.

    results_folder: str
        Results folder. It could be a local path or
        a S3 bucket.

    acquisition_path: str
        Path where the acquisition.json is.

    tilenames: List[str]
        Tiles to process. E.g.,
        [Tile_X_000...ome.zarr, ..., ]

    driver: Optional[str]
        Driver to read the data with tensorstore
        Default: "zarr"

    """
    zyx_voxel_size = utils.get_voxel_resolution(
        acquisition_path=acquisition_path
    )
    LOGGER.info(f"Voxel ZYX resolution: {zyx_voxel_size}")

    data_processes = []
    for tilename in tilenames:
        curr_tilename = tilename
        zarr_path = f"{data_folder}/{tilename}"
        output_path = f"{results_folder}/{curr_tilename}"
        data_process = correct_and_save_tile(
            dataset_loc=zarr_path,
            output_path=output_path,
            resolution_zyx=zyx_voxel_size,
            n_lvls = 6,
            driver=driver,
        )

    # utils.generate_processing(
    #     data_processes=data_processes,
    #     dest_processing=results_folder,
    #     processor_full_name=__maintainers__[0],
    #     pipeline_version=__pipeline_version__,
    #     prefix='radial_correction'
    # )


if __name__ == "__main__":
    main()
