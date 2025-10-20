"""
CZI to Zarr writer. It takes an input path
where 3D stacks are located, then these
stacks are loaded into memory and written
to zarr.
"""

import logging
import time
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union, cast
import czifile
import dask
import dask.array as da
import numpy as np
import xarray_multiscale
import zarr
from numcodecs import blosc
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscales_metadata
from .zarr_writer import BlockedArrayWriter
import json
import multiprocessing
import os
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor
from czifile.czifile import create_output
from natsort import natsorted
from pathlib import Path

ArrayLike = Union[da.Array, np.ndarray]
PathLike = Union[str, Path]

def add_leading_dim(data: ArrayLike) -> ArrayLike:
    """
    Adds a new dimension to existing data.
    Parameters
    ------------------------
    arr: ArrayLike
        Dask/numpy array that contains image data.

    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """

    return data[None, ...]


def pad_array_n_d(arr: ArrayLike, dim: int = 5) -> ArrayLike:
    """
    Pads a daks array to be in a 5D shape.

    Parameters
    ------------------------

    arr: ArrayLike
        Dask/numpy array that contains image data.
    dim: int
        Number of dimensions that the array will be padded

    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """
    if dim > 5:
        raise ValueError("Padding more than 5 dimensions is not supported.")

    while arr.ndim < dim:
        arr = arr[np.newaxis, ...]
    return arr


def extract_data(
    arr: ArrayLike, last_dimensions: Optional[int] = None
) -> ArrayLike:
    """
    Extracts n dimensional data (numpy array or dask array)
    given expanded dimensions.
    e.g., (1, 1, 1, 1600, 2000) -> (1600, 2000)
    e.g., (1, 1600, 2000) -> (1600, 2000)
    e.g., (1, 1, 2, 1600, 2000) -> (2, 1600, 2000)

    Parameters
    ------------------------
    arr: ArrayLike
        Numpy or dask array with image data. It is assumed
        that the last dimensions of the array contain
        the information about the image.

    last_dimensions: Optional[int]
        If given, it selects the number of dimensions given
        stating from the end
        of the array
        e.g., arr=(1, 1, 1600, 2000) last_dimensions=3 -> (1, 1600, 2000)
        e.g., arr=(1, 1, 1600, 2000) last_dimensions=1 -> (2000)

    Raises
    ------------------------
    ValueError:
        Whenever the last dimensions value is higher
        than the array dimensions.

    Returns
    ------------------------
    ArrayLike:
        Reshaped array with the selected indices.
    """

    if last_dimensions is not None:
        if last_dimensions > arr.ndim:
            raise ValueError(
                "Last dimensions should be lower than array dimensions"
            )

    else:
        last_dimensions = len(arr.shape) - arr.shape.count(1)

    dynamic_indices = [slice(None)] * arr.ndim

    for idx in range(arr.ndim - last_dimensions):
        dynamic_indices[idx] = 0

    return arr[tuple(dynamic_indices)]


def read_json_as_dict(filepath: PathLike) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def sync_dir_to_s3(directory_to_upload: PathLike, s3_location: str) -> None:
    """
    Syncs a local directory to an s3 location by running aws cli in a
    subprocess.

    Parameters
    ----------
    directory_to_upload : PathLike
    s3_location : str

    Returns
    -------
    None

    """
    # Upload to s3
    if platform.system() == "Windows":
        shell = True
    else:
        shell = False

    base_command = [
        "aws",
        "s3",
        "sync",
        str(directory_to_upload),
        s3_location,
        "--only-show-errors",
    ]

    subprocess.run(base_command, shell=shell, check=True)


def copy_file_to_s3(file_to_upload: PathLike, s3_location: str) -> None:
    """
    Syncs a local directory to an s3 location by running aws cli in a
    subprocess.

    Parameters
    ----------
    file_to_upload : PathLike
    s3_location : str

    Returns
    -------
    None

    """
    # Upload to s3
    if platform.system() == "Windows":
        shell = True
    else:
        shell = False

    base_command = [
        "aws",
        "s3",
        "cp",
        str(file_to_upload),
        s3_location,
        "--only-show-errors",
    ]

    subprocess.run(base_command, shell=shell, check=True)


def validate_slices(start_slice: int, end_slice: int, len_dir: int):
    """
    Validates that the slice indices are within bounds

    Parameters
    ----------
    start_slice: int
        Start slice integer

    end_slice: int
        End slice integer

    len_dir: int
        Len of czi directory
    """
    if not (0 <= start_slice < end_slice <= len_dir):
        msg = (
            f"Slices out of bounds. Total: {len_dir}"
            f"Start: {start_slice}, End: {end_slice}"
        )
        raise ValueError(msg)


def parallel_reader(
    args: tuple,
    out: np.ndarray,
    nominal_start: np.ndarray,
    start_slice: int,
    ax_index: int,
    resize: bool,
    order: int,
):
    """
    Reads a single subblock and places it in the output array.

    Parameters
    ----------
    args: tuple
        Index and directory entry of the czi file.

    out: np.ndarray
        Placeholder array for the data

    nominal_start: np.ndarray
        Nominal start of the dataset when it was acquired.

    start_slice: int
        Start slice.

    ax_index: int
        Axis index.

    resize: bool
        True if resizing is needed when reading CZI data.

    order: int
        Interpolation in resizing.
    """
    idx, directory_entry = args
    subblock = directory_entry.data_segment()
    tile = subblock.data(resize=resize, order=order)
    dir_start = np.array(directory_entry.start) - nominal_start

    # Calculate index placement
    index = tuple(slice(i, i + k) for i, k in zip(dir_start, tile.shape))
    index = list(index)
    index[ax_index] = slice(
        index[ax_index].start - start_slice, index[ax_index].stop - start_slice
    )

    try:
        out[tuple(index)] = tile
    except ValueError as e:
        raise ValueError(f"Error writing subblock {idx + start_slice}: {e}")


def read_slices_czi(
    czi_stream,
    subblock_directory: List,
    start_slice: int,
    end_slice: int,
    slice_axis: Optional[str] = "z",
    resize: Optional[bool] = True,
    order: Optional[int] = 0,
    out: Optional[List[int]] = None,
    max_workers: Optional[int] = None,
):
    """
    Reads chunked data from CZI files. From AIND-Zeiss
    the data is being chunked in a slice basis. Therefore,
    we assume the slice axis to be 'z'.

    Parameters
    ----------
    czi_stream
        Opened CZI file decriptor.

    subblock_directory: List
        List of subblock directories. These must be ordered.

    start_slice: int
        Start slice from where the data will be pulled.

    end_slice: int
        End slice from where the data will be pulled.

    slice_axis: Optional[str] = 'z'
        Axis in which start and end slice parameters will
        be applied.
        Default: 'z'

    resize: Optional[bool] = True
        If we want to resize the tile from the CZI file.
        Default: True

    order: Optional[int] = 0
        Interpolation order
        Default: 0

    out: Optional[List[int]] = None
        Out shape of the final array
        Default: None

    max_workers: Optional[int] = None
        Number of workers that will be pulling data.
        Default: None

    Returns
    -------
    np.ndarray
        Numpy array with the pulled data
    """

    shape, dtype, axes = (
        czi_stream.shape,
        czi_stream.dtype,
        list(czi_stream.axes.lower()),
    )
    nominal_start = np.array(czi_stream.start)

    len_dir = len(subblock_directory)

    validate_slices(start_slice, end_slice, len_dir)

    ax_index = axes.index(slice_axis.lower())
    new_shape = list(shape)
    new_shape[ax_index] = end_slice - start_slice
    new_shape[axes.index("c")] = 1  # Assume 1 channel per CZI

    out = create_output(out, new_shape, dtype)
    max_workers = max_workers or min(
        multiprocessing.cpu_count() // 2, end_slice - start_slice
    )

    selected_entries = subblock_directory[start_slice:end_slice]

    if max_workers > 1 and end_slice - start_slice > 1:
        czi_stream._fh.lock = True
        with ThreadPoolExecutor(max_workers) as executor:
            executor.map(
                lambda args: parallel_reader(
                    args,
                    out,
                    nominal_start,
                    start_slice,
                    ax_index,
                    resize,
                    order,
                ),
                enumerate(selected_entries),
            )
        czi_stream._fh.lock = None
    else:
        for idx, entry in enumerate(selected_entries):
            parallel_reader(
                (idx, entry),
                out,
                nominal_start,
                start_slice,
                ax_index,
                resize,
                order,
            )

    if hasattr(out, "flush"):
        out.flush()

    return np.squeeze(out)


def generate_jumps(n: int, jump_size: Optional[int] = 128):
    """
    Generates jumps for indexing.

    Parameters
    ----------
    n: int
        Final number for indexing.
        It is exclusive in the final number.

    jump_size: Optional[int] = 128
        Jump size.
    """
    jumps = list(range(0, n, jump_size))
    # if jumps[-1] + jump_size >= n:
    #     jumps.append(n)

    return jumps


def get_axis_index(czi_shape: List[int], czi_axis: int, axis_name: str):
    """
    Gets the axis index from the CZI natural shape.

    Parameters
    ----------
    czi_shape: List[int]
        List of ints of the CZI shape. CZI files come
        with many more axis than traditional file formats.
        Please, check its documentation.

    czi_axis: int
        Axis from which we will pull the index.

    axis_name: str
        Axis name. Allowed axis names are:
        ['b', 'v', 'i', 'h', 'r', 's', 'c', 't', 'z', 'y', 'x', '0']
    """
    czi_axis = list(str(czi_axis).lower())
    axis_name = axis_name.lower()
    ALLOWED_AXIS_NAMES = [
        "b",
        "v",
        "i",
        "h",
        "r",
        "s",
        "c",
        "t",
        "z",
        "y",
        "x",
        "0",
    ]

    if axis_name not in ALLOWED_AXIS_NAMES:
        raise ValueError(f"Axis {axis_name} not valid!")

    czi_shape = list(czi_shape)
    ax_index = czi_axis.index(axis_name)

    return ax_index, czi_shape[ax_index]


def czi_block_generator(
    czi_decriptor,
    axis_jumps: Optional[int] = 128,
    slice_axis: Optional[str] = "z",
):
    """
    CZI data block generator.

    Parameters
    ----------
    czi_decriptor
        Opened CZI file.

    axis_jumps: int
        Number of jumps in a given axis.
        Default: 128

    slice_axis: str
        Axis in which the jumps will be
        generated.
        Default: 'z'

    Yields
    ------
    np.ndarray
        Numpy array with the data
        of the picked block.

    slice
        Slice of start and end positions
        in a given axis.
    """

    axis_index, axis_shape = get_axis_index(
        czi_decriptor.shape, czi_decriptor.axes, slice_axis
    )

    subblock_directory = czi_decriptor.filtered_subblock_directory

    # Sorting indices so planes are ordered
    ordered_subblock_directory = natsorted(subblock_directory, key=lambda sb: sb.start[axis_index])

    jumps = generate_jumps(axis_shape, axis_jumps)
    n_jumps = len(jumps)
    for i, start_slice in enumerate(jumps):
        if i + 1 < n_jumps:
            end_slice = jumps[i + 1]

        else:
            end_slice = axis_shape

        block = read_slices_czi(
            czi_decriptor,
            subblock_directory=ordered_subblock_directory,
            start_slice=start_slice,
            end_slice=end_slice,
            slice_axis=slice_axis,
            resize=True,
            order=0,
            out=None,
            max_workers=None,
        )
        yield block, slice(start_slice, end_slice)


def _build_ome(
    data_shape: Tuple[int, ...],
    image_name: str,
    channel_names: Optional[List[str]] = None,
    channel_colors: Optional[List[int]] = None,
    channel_minmax: Optional[List[Tuple[float, float]]] = None,
    channel_startend: Optional[List[Tuple[float, float]]] = None,
) -> Dict:
    """
    Create the necessary metadata for an OME tiff image

    Parameters
    ----------
    data_shape: A 5-d tuple, assumed to be TCZYX order
    image_name: The name of the image
    channel_names: The names for each channel
    channel_colors: List of all channel colors
    channel_minmax: List of all (min, max) pairs of channel pixel
    ranges (min value of darkest pixel, max value of brightest)
    channel_startend: List of all pairs for rendering where start is
    a pixel value of darkness and end where a pixel value is
    saturated

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
    if channel_startend is None:
        channel_startend = channel_minmax

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
                    "end": float(channel_startend[i][1]),
                    "max": float(channel_minmax[i][1]),
                    "min": float(channel_minmax[i][0]),
                    "start": float(channel_startend[i][0]),
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
    translation: Optional[List[float]] = None,
) -> Tuple[List, List]:
    """
    Generate the list of coordinate transformations
    and associated chunk options.

    Parameters
    ----------
    scale_num_levels: the number of downsampling levels
    scale_factor: a tuple of scale factors in each spatial dimension (Z, Y, X)
    pixelsizes: a list of pixel sizes in each spatial dimension (Z, Y, X)
    chunks: a 5D tuple of integers with size of each
    chunk dimension (T, C, Z, Y, X)
    data_shape: a 5D tuple of the full resolution image's shape
    translation: a 5 element list specifying the offset
    in physical units in each dimension

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
    if translation is not None:
        transforms[0].append(
            {"type": "translation", "translation": translation}
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
            if translation is not None:
                transforms[-1].append(
                    {"type": "translation", "translation": translation}
                )
            lastz = int(np.ceil(lastz / scale_factor[0]))
            lasty = int(np.ceil(lasty / scale_factor[1]))
            lastx = int(np.ceil(lastx / scale_factor[2]))
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


def write_ome_ngff_metadata(
    group: zarr.Group,
    arr_shape: List[int],
    final_chunksize: List[int],
    image_name: str,
    n_lvls: int,
    scale_factors: tuple,
    voxel_size: tuple,
    channel_names: List[str] = None,
    channel_colors: List[str] = None,
    channel_minmax: List[float] = None,
    channel_startend: List[float] = None,
    metadata: dict = None,
):
    """
    Write OME-NGFF metadata to a Zarr group.

    Parameters
    ----------
    group : zarr.Group
        The output Zarr group.
    arr_shape : List[int]
        List of ints with the dataset shape.
    image_name : str
        The name of the image.
    n_lvls : int
        The number of pyramid levels.
    scale_factors : tuple
        The scale factors for downsampling along each dimension.
    voxel_size : tuple
        The voxel size along each dimension.
    channel_names: List[str]
        List of channel names to add to the OMENGFF metadata
    channel_colors: List[str]
        List of channel colors to visualize the data
    chanel_minmax: List[float]
        List of channel min and max values based on the
        image dtype
    channel_startend: List[float]
        List of the channel start and end metadata. This is
        used for visualization. The start and end range might be
        different from the min max and it is usually inside the
        range
    metadata: dict
        Extra metadata to write in the OME-NGFF metadata
    """
    if metadata is None:
        metadata = {}
    fmt = CurrentFormat()

    # Building the OMERO metadata
    ome_json = _build_ome(
        arr_shape,
        image_name,
        channel_names=channel_names,
        channel_colors=channel_colors,
        channel_minmax=channel_minmax,
        channel_startend=channel_startend,
    )
    group.attrs["omero"] = ome_json
    axes_5d = _get_axes_5d()
    coordinate_transformations, chunk_opts = _compute_scales(
        n_lvls, scale_factors, voxel_size, final_chunksize, arr_shape, None
    )
    fmt.validate_coordinate_transformations(
        len(arr_shape), n_lvls, coordinate_transformations
    )
    # Setting coordinate transfomations
    datasets = [{"path": str(i)} for i in range(n_lvls)]
    if coordinate_transformations is not None:
        for dataset, transform in zip(datasets, coordinate_transformations):
            dataset["coordinateTransformations"] = transform

    # Writing the multiscale metadata
    write_multiscales_metadata(group, datasets, fmt, axes_5d, **metadata)


def create_czi_opts(codec: str, compression_level: int) -> dict:
    """
    Creates CZI options for writing
    the OMEZarr.

    Parameters
    ----------
    codec: str
        Image codec used to write the image

    compression_level: int
        Compression level for the image

    Returns
    -------
    dict
        Dictionary with the blosc compression
        to write the CZI image
    """
    return {
        "compressor": blosc.Blosc(
            cname=codec, clevel=compression_level, shuffle=blosc.SHUFFLE
        )
    }


def _get_pyramid_metadata():
    """
    Gets the image pyramid metadata
    using xarray_multiscale package
    """
    return {
        "metadata": {
            "description": "Downscaling using the windowed mean",
            "method": "xarray_multiscale.reducers.windowed_mean",
            "version": str(xarray_multiscale.__version__),
            "args": "[false]",
            # No extra parameters were used different
            # from the orig. array and scales
            "kwargs": {},
        }
    }


def compute_pyramid(
    data: dask.array.core.Array,
    n_lvls: int,
    scale_axis: Tuple[int],
    chunks: Union[str, Sequence[int], Dict[Hashable, int]] = "auto",
) -> List[dask.array.core.Array]:
    """
    Computes the pyramid levels given an input full resolution image data

    Parameters
    ------------------------

    data: dask.array.core.Array
        Dask array of the image data

    n_lvls: int
        Number of downsampling levels
        that will be applied to the original image

    scale_axis: Tuple[int]
        Scaling applied to each axis

    chunks: Union[str, Sequence[int], Dict[Hashable, int]]
        chunksize that will be applied to the multiscales
        Default: "auto"

    Returns
    ------------------------

    Tuple[List[dask.array.core.Array], Dict]:
        List with the downsampled image(s) and dictionary
        with image metadata
    """

    metadata = _get_pyramid_metadata()

    pyramid = xarray_multiscale.multiscale(
        array=data,
        reduction=xarray_multiscale.reducers.windowed_mean,  # func
        scale_factors=scale_axis,  # scale factors
        preserve_dtype=True,
        chunks=chunks,
    )[:n_lvls]

    return [pyramid_level.data for pyramid_level in pyramid], metadata


def czi_stack_zarr_writer(
    czi_path: str,
    output_path: str,
    voxel_size: List[float],
    final_chunksize: List[int],
    scale_factor: List[int],
    n_lvls: int,
    channel_name: str,
    logger: logging.Logger,
    stack_name: str,
    writing_options,
    target_size_mb: Optional[int] = 24000,
):
    """
    Writes a fused Zeiss channel in OMEZarr
    format. This channel was read as a lazy array.

    Parameters
    ----------
    czi_path: str
        Path where the CZI file is stored.

    output_path: PathLike
        Path where we want to write the OMEZarr
        channel

    voxel_size: List[float]
        Voxel size representing the dataset

    final_chunksize: List[int]
        Final chunksize we want to use to write
        the final dataset

    codec: str
        Image codec for writing the Zarr

    compression_level: int
        Compression level

    scale_factor: List[int]
        Scale factor per axis. The dimensionality
        is organized as ZYX.

    n_lvls: int
        Number of levels on the pyramid (multiresolution)
        for better visualization

    channel_name: str
        Channel name we are currently writing

    logger: logging.Logger
        Logger object

    target_size_mb: Optional[int]
        Target size to pull from the CZI array.

    """
    written_pyramid = []
    start_time = time.time()

    with czifile.CziFile(str(czi_path)) as czi:
        dataset_shape = tuple(i for i in czi.shape if i != 1)
        extra_axes = (1,) * (5 - len(dataset_shape))
        dataset_shape = extra_axes + dataset_shape

        final_chunksize = ([1] * (5 - len(final_chunksize))) + final_chunksize
        # Getting channel color
        channel_colors = None

        print(f"Writing {dataset_shape} from {stack_name} to {output_path}")

        # Creating Zarr dataset
        store = parse_url(path=output_path, mode="w").store
        root_group = zarr.group(store=store)

        # Using 1 thread since is in single machine.
        # Avoiding the use of multithreaded due to GIL

        if np.issubdtype(czi.dtype, np.integer):
            np_info_func = np.iinfo

        else:
            # Floating point
            np_info_func = np.finfo

        # Getting min max metadata for the dtype
        channel_minmax = [
            (
                np_info_func(czi.dtype).min,
                np_info_func(czi.dtype).max,
            )
            for _ in range(dataset_shape[1])
        ]

        # Setting values for CZI
        # Ideally we would use da.percentile(image_data, (0.1, 95))
        # However, it would take so much time and resources and it is
        # not used that much on neuroglancer
        channel_startend = [(0.0, 550.0) for _ in range(dataset_shape[1])]

        new_channel_group = root_group.create_group(
            name=stack_name, overwrite=True
        )

        # Writing OME-NGFF metadata
        write_ome_ngff_metadata(
            group=new_channel_group,
            arr_shape=dataset_shape,
            image_name=stack_name,
            n_lvls=n_lvls,
            scale_factors=scale_factor,
            voxel_size=voxel_size,
            channel_names=[channel_name],
            channel_colors=channel_colors,
            channel_minmax=channel_minmax,
            channel_startend=channel_startend,
            metadata=_get_pyramid_metadata(),
            final_chunksize=final_chunksize,
        )

        # performance_report_path = f"{output_path}/report_{stack_name}.html"

        # Writing zarr and performance report
        # with performance_report(filename=performance_report_path):
        logger.info(f"Writing channel {channel_name}/{stack_name}")

        # Writing first multiscale by default
        pyramid_group = new_channel_group.create_dataset(
            name="0",
            shape=dataset_shape,
            chunks=final_chunksize,
            dtype=czi.dtype,
            compressor=writing_options,
            dimension_separator="/",
            overwrite=True,
        )

        # final_chunksize must be TCZYX order
        for block, axis_area in czi_block_generator(
            czi,
            axis_jumps=final_chunksize[-3],
            slice_axis="z",
        ):
            region = (
                slice(None),
                slice(None),
                axis_area,
                slice(0, dataset_shape[-2]),
                slice(0, dataset_shape[-1]),
            )
            pyramid_group[region] = pad_array_n_d(block)

        # Writing multiscales
        previous_scale = da.from_zarr(pyramid_group, pyramid_group.chunks)
        written_pyramid.append(previous_scale)

        block_shape = list(
            BlockedArrayWriter.get_block_shape(
                arr=previous_scale,
                target_size_mb=target_size_mb,
                chunks=final_chunksize,
            )
        )
        block_shape = extra_axes + tuple(block_shape)

        for level in range(1, n_lvls):
            previous_scale = da.from_zarr(pyramid_group, pyramid_group.chunks)
            new_scale_factor = (
                [1] * (len(previous_scale.shape) - len(scale_factor))
            ) + scale_factor

            previous_scale_pyramid, _ = compute_pyramid(
                data=previous_scale,
                scale_axis=new_scale_factor,
                chunks=final_chunksize,
                n_lvls=2,
            )
            array_to_write = previous_scale_pyramid[-1]

            logger.info(
                f"[level {level}]: pyramid level: {array_to_write.shape}"
            )

            pyramid_group = new_channel_group.create_dataset(
                name=str(level),
                shape=array_to_write.shape,
                chunks=final_chunksize,
                dtype=array_to_write.dtype,
                compressor=writing_options,
                dimension_separator="/",
                overwrite=True,
            )
            BlockedArrayWriter.store(
                array_to_write, pyramid_group, block_shape
            )
            written_pyramid.append(array_to_write)

    end_time = time.time()
    logger.info(f"Time to write the dataset: {end_time - start_time}")
    print(f"Time to write the dataset: {end_time - start_time}")
    logger.info(f"Written pyramid: {written_pyramid}")


def example():
    """
    Conversion example
    """
    from pathlib import Path

    czi_test_stack = Path("path/to/data/tiles_test/SPIM/488_large.czi")

    if czi_test_stack.exists():
        writing_opts = create_czi_opts(codec="zstd", compression_level=3)

        # for channel_name in
        # for i, chn_name in enumerate(czi_file_reader.channel_names):
        czi_stack_zarr_writer(
            czi_path=str(czi_test_stack),
            output_path=f"./{czi_test_stack.stem}",
            voxel_size=[1.0, 1.0, 1.0],
            final_chunksize=[128, 128, 128],
            scale_factor=[2, 2, 2],
            n_lvls=4,
            channel_name=czi_test_stack.stem,
            logger=logging.Logger(name="test"),
            stack_name="test_conversion_czi_package.zarr",
            writing_options=writing_opts["compressor"],
        )

    else:
        print(f"File does not exist: {czi_test_stack}")


if __name__ == "__main__":
    example()