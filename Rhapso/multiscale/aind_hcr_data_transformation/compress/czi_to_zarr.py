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

from .zarr_writer import (
    BlockedArrayWriter,
)
from ..utils.utils import (
    czi_block_generator,
    pad_array_n_d,
)


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
    translations: Optional[List[float]] = None,
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
    if translations is not None:
        transforms[0].append(
            {"type": "translation", "translation": translations[0]}
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
                    {"type": "translation", "translation": translations[i + 1]}
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


def _downscale_origin(
    array_shape: List[int],
    origin: List[float],
    voxel_size: List[float],
    scale_factors: List[int],
    n_levels: int,
):
    """
    Calculate new origins for downscaled coordinate grids.

    Parameters
    ----------
    array_shape : List[int]
        Shape of the array in [t, c, z, y, x] order.
    origin : list or tuple of float
        The initial origin coordinates (z, y, x) of the array.
    voxel_size : list or tuple of float
        The size of each voxel along the (z, y, x) dimensions.
    scale_factors : list or tuple of int
        The factors by which to downscale the coordinates along each axis
        (z, y, x).
    n_levels : int
        The number of downscaling levels to calculate.

    Returns
    -------
    new_origins : list of list of float
        A list of new origin coordinates for each downscaled level.
    """
    current_shape = np.array(array_shape[-3:], dtype=np.int32)
    current_origin = np.array(origin[-3:], dtype=np.float64)
    current_voxel_size = np.array(voxel_size[-3:], dtype=np.float64)
    scale_factors = np.array(scale_factors[-3:], dtype=np.int32)

    new_origins = [current_origin.tolist()]

    for _ in range(n_levels - 1):

        # Calculate the center shift for the new origin
        center_shift = (current_voxel_size * (scale_factors - 1)) / 2
        current_origin += center_shift

        current_shape = np.ceil(current_shape / scale_factors).astype(int)
        next_voxel_size = current_voxel_size * scale_factors
        current_voxel_size = next_voxel_size

        # Append the new origin
        new_origins.append([0.0, 0.0] + current_origin.tolist())

    # Ensure the initial origin is 5D
    if len(new_origins[0]) < 5:
        new_origins[0] = [0.0, 0.0] + new_origins[0]

    return new_origins


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
    origin: list = None,
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

    if origin is not None:
        origin = _downscale_origin(
            arr_shape, origin[-3:], voxel_size[-3:], scale_factors[-3:], n_lvls
        )

    coordinate_transformations, chunk_opts = _compute_scales(
        n_lvls, scale_factors, voxel_size, final_chunksize, arr_shape, origin
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
    # Try to get version, fallback to 'unknown' if not available
    try:
        version = str(xarray_multiscale.__version__)
    except AttributeError:
        version = "unknown"
    
    return {
        "metadata": {
            "description": "Downscaling using the windowed mean",
            "method": "xarray_multiscale.reducers.windowed_mean",
            "version": version,
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
            origin=[0, 0, 0],
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
