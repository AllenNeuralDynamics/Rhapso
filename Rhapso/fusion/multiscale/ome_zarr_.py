from fusion.multiscale.multiscale_ import downscale, multiscale
from fusion.multiscale.reducers import windowed_mean, WindowedReducer
from fusion.multiscale.chunk_utils import Union, Tuple
from fusion.multiscale.blocked_array_writer import BlockedArrayWriter

from ome_zarr.format import CurrentFormat
from ome_zarr.writer import write_multiscales_metadata
from typing import List, Optional, Dict, cast, Any
from numcodecs.abc import Codec
from numpy.typing import NDArray
import xarray as xr
import math
import dask.array as da
import numpy as np
import logging
import zarr


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

class DimensionsError(Exception):
    pass

def _get_bytes(data: Union[List[NDArray], NDArray]):
    if isinstance(data, list):
        total_bytes = 0
        for arr in data:
            total_bytes += arr.nbytes
        return total_bytes
    return data.nbytes


def _build_ome(
    data_shape: Tuple[int, ...],
    image_name: str,
    channel_names: Optional[List[str]] = None,
    channel_colors: Optional[List[int]] = None,
    channel_minmax: Optional[List[Tuple[float, float]]] = None,
) -> Dict:
    """
    Create the necessary metadata for an OME tiff image

    Parameters
    ----------
    data_shape: A 5-d tuple, assumed to be TCZYX order
    image_name: The name of the image
    channel_names: The names for each channel
    channel_colors: List of all channel colors
    channel_minmax: List of all (min, max) pairs of channel intensities

    Returns
    -------
    Dict: An "omero" metadata object suitable for writing to ome-zarr
    """
    if channel_names is None:
        channel_names = [
            f"Channel:{image_name}:{i}" for i in range(data_shape[1])
        ]
    if channel_colors is None:
        channel_colors = [i for i in range(data_shape[1])]
    if channel_minmax is None:
        channel_minmax = [(0.0, 1.0) for _ in range(data_shape[1])]
    ch = []
    for i in range(data_shape[1]):
        ch.append(
            {
                "active": True,
                "coefficient": 1,
                "color": f"{channel_colors[i]:06x}",
                "family": "linear",
                "inverted": False,
                "label": channel_names[i],
                "window": {
                    "end": float(channel_minmax[i][1]),
                    "max": float(channel_minmax[i][1]),
                    "min": float(channel_minmax[i][0]),
                    "start": float(channel_minmax[i][0]),
                },
            }
        )

    omero = {
        "id": 1,  # ID in OMERO
        "name": image_name,  # Name as shown in the UI
        "version": "0.4",  # Current version
        "channels": ch,
        "rdefs": {
            "defaultT": 0,  # First timepoint to show the user
            "defaultZ": data_shape[2] // 2,  # First Z section to show the user
            "model": "color",  # "color" or "greyscale"
        },
    }
    return omero


def _compute_scales(
    scale_num_levels: int,
    scale_factor: Tuple[float, float, float],
    pixelsizes: Tuple[float, float, float],
    chunks: Tuple[int, int, int, int, int],
    data_shape: Tuple[int, int, int, int, int],
    translations: Optional[List[List[float]]] = None,
) -> Tuple[List, List]:
    """Generate the list of coordinate transformations and associated chunk options.

    Parameters
    ----------
    scale_num_levels: the number of downsampling levels
    scale_factor: a tuple of scale factors in each spatial dimension (Z, Y, X)
    pixelsizes: a list of pixel sizes in each spatial dimension (Z, Y, X)
    chunks: a 5D tuple of integers with size of each chunk dimension (T, C, Z, Y, X)
    data_shape: a 5D tuple of the full resolution image's shape
    translations: a list of 3 element lists specifying the offset in physical units in Z, Y, X

    Returns
    -------
    A tuple of the coordinate transforms and chunk options
    """
    transforms = [
        [
            # the voxel size for the first scale level
            {
                "type": "scale",
                "scale": [
                    1.0,
                    1.0,
                    pixelsizes[0],
                    pixelsizes[1],
                    pixelsizes[2],
                ],
            }
        ]
    ]
    if translations is not None:
        transforms[0].append(
            {"type": "translation", "translation": [0, 0, *translations[0]]}
        )
    chunk_sizes = []
    lastz = data_shape[2]
    lasty = data_shape[3]
    lastx = data_shape[4]
    opts = dict(
        chunks=(
            1,
            1,
            min(lastz, chunks[2]),
            min(lasty, chunks[3]),
            min(lastx, chunks[4]),
        )
    )
    chunk_sizes.append(opts)
    if scale_num_levels > 1:
        for i in range(scale_num_levels - 1):
            last_transform = transforms[-1][0]
            last_scale = cast(List, last_transform["scale"])
            transforms.append(
                [
                    {
                        "type": "scale",
                        "scale": [
                            1.0,
                            1.0,
                            last_scale[2] * scale_factor[0],
                            last_scale[3] * scale_factor[1],
                            last_scale[4] * scale_factor[2],
                        ],
                    }
                ]
            )
            if translations is not None:
                transforms[-1].append(
                    {"type": "translation", "translation": [0, 0, *translations[i+1]]}
                )
            lastz = int(math.ceil(lastz / scale_factor[0]))
            lasty = int(math.ceil(lasty / scale_factor[1]))
            lastx = int(math.ceil(lastx / scale_factor[2]))
            opts = dict(
                chunks=(
                    1,
                    1,
                    min(lastz, chunks[2]),
                    min(lasty, chunks[3]),
                    min(lastx, chunks[4]),
                )
            )
            chunk_sizes.append(opts)

    return transforms, chunk_sizes


def _get_axes_5d(
    time_unit: str = "millisecond", space_unit: str = "micrometer"
) -> List[Dict]:
    """Generate the list of axes.

    Parameters
    ----------
    time_unit: the time unit string, e.g., "millisecond"
    space_unit: the space unit string, e.g., "micrometer"

    Returns
    -------
    A list of dictionaries for each axis
    """
    axes_5d = [
        {"name": "t", "type": "time", "unit": f"{time_unit}"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": f"{space_unit}"},
        {"name": "y", "type": "space", "unit": f"{space_unit}"},
        {"name": "x", "type": "space", "unit": f"{space_unit}"},
    ]
    return axes_5d


def store_array(
    arr: da.Array,
    group: zarr.Group,
    path: str,
    block_shape: tuple,
    compressor: Codec = None,
    dimension_separator: str = "/",
) -> zarr.Array:
    """
    Store the full resolution layer of a Dask pyramid into a Zarr group.

    Parameters
    ----------
    arr : da.Array
        The input Dask array.
    group : zarr.Group
        The output Zarr group.
    block_shape : Tuple
        The shape of blocks to use for partitioning the array.
    compressor : numcodecs.abc.Codec, optional
        The compression codec to use for the output Zarr array. Default is Blosc with "zstd" method and compression
        level 1.

    Returns
    -------
    zarr.Array
        The output Zarr array.
    """
    ds = group.create_dataset(
        path,
        shape=arr.shape,
        chunks=arr.chunksize,
        dtype=arr.dtype,
        compressor=compressor,
        dimension_separator=dimension_separator,
        overwrite=True,
    )

    BlockedArrayWriter.store(arr, ds, block_shape)

    return ds

def create_pyramid(
    arr: Union[np.ndarray, da.Array],
    n_lvls: int,
    scale_factors: tuple,
    chunks: Union[str, tuple] = "preserve",
    reducer: WindowedReducer = windowed_mean,
) -> list:
    """
    Create a multiscale pyramid of the input data.

    Parameters
    ----------
    arr : array-like
        The input data to be downsampled.
    n_lvls : int
        The number of pyramid levels to generate.
    scale_factors : tuple
        The scale factors for downsampling along each dimension.
    chunks : tuple
        The chunk size to use for the output arrays.

    Returns
    -------
    list
        A list of Dask arrays representing the pyramid levels.
    """
    pyramid = multiscale(
        array=arr,
        reduction=reducer,  # func
        scale_factors=scale_factors,
        preserve_dtype=True,
        chunks=chunks,
    )[:n_lvls]

    return [arr.data for arr in pyramid]

def ensure_array_5d(
    arr: Union[np.ndarray, da.Array]
) -> Union[np.ndarray, da.Array]:
    """
    Checks that the array is 5D, adding singleton dimensions to the
    start of the array if less, throwing a DimensionsError if more
    Args:
        arr: the arraylike object
    Returns:
        the 5D array
    Raises:
        DimensionsError: if the array has more than 5 dimensions
    """
    if arr.ndim > 5:
        raise DimensionsError("Only arrays up to 5D are supported")
    while arr.ndim < 5:
        arr = arr[np.newaxis, ...]
    return arr

def _get_first_mipmap_level(
    arr: da.Array,
    scale_factors: tuple,
    reducer: WindowedReducer = windowed_mean,
) -> da.Array:
    """
    Generate a mipmap pyramid from the input array and return the first mipmap level.

    Parameters:
    - arr: dask.array.Array
        The input array for which the mipmap pyramid is to be generated.
    - scale_factors: tuple
        A 5D tuple of scale factors for each dimension used to generate the mipmap pyramid.

    Returns:
    - dask.array.Array
        The first mipmap level of the input array.
    """
    n_lvls = 2
    pyramid = create_pyramid(
        arr, n_lvls, scale_factors, arr.chunksize, reducer
    )
    return ensure_array_5d(pyramid[1])


def downsample_and_store(
    arr: da.Array,
    group: zarr.Group,
    n_lvls: int,
    scale_factors: Tuple,
    block_shape: Tuple,
    compressor: Codec = None,
    reducer: WindowedReducer = windowed_mean,
) -> list:
    """
    Progressively downsample the input array and store the results as separate arrays in a Zarr group.

    Parameters
    ----------
    arr : da.Array
        The full-resolution Dask array.
    group : zarr.Group
        The output Zarr group.
    n_lvls : int
        The number of pyramid levels.
    scale_factors : Tuple
        The scale factors for downsampling along each dimension.
    block_shape : Tuple
        The shape of blocks to use for partitioning the array.
    compressor : numcodecs.abc.Codec, optional
        The compression codec to use for the output Zarr array. Default is Blosc with "zstd" method and compression
        level 1.
    """
    pyramid = [arr]

    for arr_index in range(1, n_lvls):
        first_mipmap = _get_first_mipmap_level(arr, scale_factors, reducer)

        ds = group.create_dataset(
            str(arr_index),
            shape=first_mipmap.shape,
            chunks=first_mipmap.chunksize,
            dtype=first_mipmap.dtype,
            compressor=compressor,
            dimension_separator="/",
            overwrite=True,
        )

        BlockedArrayWriter.store(first_mipmap, ds, block_shape)

        arr = da.from_array(ds, ds.chunks)
        pyramid.append(arr)

    return pyramid


def _downscale_origin(
    arr: Any,
    origin: List[float],
    voxel_size: List[float],
    scale_factors: List[int],
    n_levels: int
):
    """
    Calculate new origins for downscaled coordinate grids.

    Parameters
    ----------
    arr : Any
       Input 5D array representing the data volume.
    origin : list or tuple of float
       The initial origin coordinates (z, y, x) of the array.
    voxel_size : list or tuple of float
       The size of each voxel along the (z, y, x) dimensions.
    scale_factors : list or tuple of float
       The factors by which to downscale the coordinates along each axis
       (z, y, x).
    n_levels : int
       The number of downscaling levels to calculate.

    Returns
    -------
    new_origins : list of list of float
       A list of new origin coordinates for each downscaled level.
    """
    arr = arr.squeeze()
    z_coords = origin[0] + voxel_size[0] * np.arange(arr.shape[0])
    y_coords = origin[1] + voxel_size[1] * np.arange(arr.shape[1])
    x_coords = origin[2] + voxel_size[2] * np.arange(arr.shape[2])
    coords = xr.Coordinates({'z': z_coords, 'y': y_coords, 'x': x_coords})
    ds = xr.DataArray(arr, coords=coords)
    new_origins = [list(origin)]
    for i in range(n_levels - 1):
        ds = downscale(ds, windowed_mean, scale_factors)
        new_origins.append(
            [float(ds.coords['z'][0]), float(ds.coords['y'][0]),
             float(ds.coords['x'][0])]
        )
    return new_origins


def write_ome_ngff_metadata(
    group: zarr.Group,
    arr: da.Array,
    image_name: str,
    n_lvls: int,
    scale_factors: tuple,
    voxel_size: tuple,
    origin: list = None,
    metadata: dict = None,
) -> None:
    """
    Write OME-NGFF metadata to a Zarr group.

    Parameters
    ----------
    group : zarr.Group
        The output Zarr group.
    arr : array-like
        The input array.
    image_name : str
        The name of the image.
    n_lvls : int
        The number of pyramid levels.
    scale_factors : tuple
        The scale factors for downsampling along each dimension.
    voxel_size : tuple
        The voxel size along each dimension.
    """
    if metadata is None:
        metadata = {}
    fmt = CurrentFormat()
    ome_json = _build_ome(
        arr.shape,
        image_name,
        channel_names=None,
        channel_colors=None,
        channel_minmax=None,
    )
    group.attrs["omero"] = ome_json
    axes_5d = _get_axes_5d()

    if origin is not None:
        origin = _downscale_origin(
            arr,
            origin[-3:],
            voxel_size[-3:],
            scale_factors[-3:],
            n_lvls
        )

    coordinate_transformations, chunk_opts = _compute_scales(
        n_lvls, scale_factors, voxel_size, arr.chunksize, arr.shape, origin
    )
    fmt.validate_coordinate_transformations(
        arr.ndim, n_lvls, coordinate_transformations
    )
    datasets = [{"path": str(i)} for i in range(n_lvls)]
    if coordinate_transformations is not None:
        for dataset, transform in zip(datasets, coordinate_transformations):
            dataset["coordinateTransformations"] = transform

    write_multiscales_metadata(group, datasets, fmt, axes_5d, **metadata)