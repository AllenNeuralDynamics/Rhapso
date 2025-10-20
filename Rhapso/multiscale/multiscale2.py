from datetime import datetime
from pathlib import Path 
import glob
import logging
import time
import uuid
import yaml

import dask
from dask.distributed import Client, LocalCluster, performance_report
from ome_zarr.io import parse_url
import numpy as np
import s3fs
import zarr

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def ensure_array_5d(arr):
    """
    Ensures the array is 5D by adding singleton dimensions if necessary.
    """
    if arr.ndim > 5:
        raise ValueError("Only arrays up to 5D are supported")
    while arr.ndim < 5:
        arr = arr[np.newaxis, ...]
    return arr

def ensure_shape_5d(shape):
    """
    Ensures the shape is 5D by adding singleton dimensions if necessary.
    """
    if len(shape) > 5:
        raise ValueError("Only shapes up to 5D are supported")
    while len(shape) < 5:
        shape = (1, *shape)
    return shape

def expand_chunks(
    chunks: tuple,
    data_shape: tuple,
    target_size: int,
    itemsize: int,
    mode: str = "iso",
) -> tuple:
    """
    Given the shape and chunk size of a pre-chunked 3D array, determine the optimal chunk shape
    closest to target_size. Expanded chunk dimensions are an integer multiple of the base chunk dimension,
    to ensure optimal access patterns.

    Args:
        chunks: the shape of the input array chunks
        data_shape: the shape of the input array
        target_size: target chunk size in bytes
        itemsize: the number of bytes per array element
        mode: chunking strategy. Must be one of "cycle", or "iso"

    Returns:
        the optimal chunk shape
    """
    if any(c < 1 for c in chunks):
        raise ValueError("chunks must be >= 1 for all dimensions")
    if any(s < 1 for s in data_shape):
        raise ValueError("data_shape must be >= 1 for all dimensions")
    if any(c > s for c, s in zip(chunks, data_shape)):
        raise ValueError(
            "chunks cannot be larger than data_shape in any dimension"
        )
    if target_size <= 0:
        raise ValueError("target_size must be > 0")
    if itemsize <= 0:
        raise ValueError("itemsize must be > 0")

    if mode == "cycle":
        current = np.array(chunks, dtype=np.uint64)
        prev = current.copy()
        idx = 0
        ndims = len(current)
        while _get_size(current, itemsize) < target_size:
            prev = current.copy()
            current[idx % ndims] = min(data_shape[idx % ndims], current[idx % ndims] * 2)
            idx += 1
            if all(c >= s for c, s in zip(current, data_shape)):
                break
        expanded = _closer_to_target(current, prev, target_size, itemsize)
    elif mode == "iso":
        initial = np.array(chunks, dtype=np.uint64)
        current = initial
        prev = current
        i = 2
        while _get_size(current, itemsize) < target_size:
            prev = current
            current = initial * i
            current = (
                min(data_shape[0], current[0]),
                min(data_shape[1], current[1]),
                min(data_shape[2], current[2]),
            )
            i += 1
            if all(c >= s for c, s in zip(current, data_shape)):
                break
        expanded = _closer_to_target(current, prev, target_size, itemsize)
    else:
        raise ValueError(f"Invalid mode {mode}")

    return tuple(int(d) for d in expanded)

def _closer_to_target(
    shape1: tuple,
    shape2: tuple,
    target_bytes: int,
    itemsize: int,
) -> tuple:
    """
    Given two shapes with the same number of dimensions,
    find which one is closer to target_bytes.

    Args:
        shape1: the first shape
        shape2: the second shape
        target_bytes: the target size for the returned shape
        itemsize: number of bytes per array element
    """
    size1 = _get_size(shape1, itemsize)
    size2 = _get_size(shape2, itemsize)
    if abs(size1 - target_bytes) < abs(size2 - target_bytes):
        return shape1
    return shape2

def _get_size(shape: tuple, itemsize: int) -> int:
    """
    Return the size of an array with the given shape, in bytes.

    Args:
        shape: the shape of the array
        itemsize: number of bytes per array element

    Returns:
        the size of the array, in bytes
    """
    if any(s <= 0 for s in shape):
        raise ValueError("shape must be > 0 in all dimensions")
    return np.product(shape) * itemsize

class BlockedArrayWriter:
    @staticmethod
    def get_block_shape(arr, target_size_mb=409600, mode="cycle"):
        """
        Determines the optimal block shape for partitioning the array.
        """
        chunks = arr.chunksize[-3:] if isinstance(arr, dask.array.Array) else arr.chunks[-3:]
        return expand_chunks(chunks, arr.shape[-3:], target_size_mb * 1024**2, arr.itemsize, mode)

    @staticmethod
    def gen_slices(arr_shape, block_shape):
        """
        Generates slices for traversing an array in blocks of a given shape.
        """
        if len(arr_shape) != len(block_shape):
            raise ValueError("Array shape and block shape must have the same length")

        def _slice_along_dim(dim):
            if dim >= len(arr_shape):
                yield ()
            else:
                for i in range(0, arr_shape[dim], block_shape[dim]):
                    end_i = min(i + block_shape[dim], arr_shape[dim])
                    for rest in _slice_along_dim(dim + 1):
                        yield (slice(i, end_i),) + rest

        return _slice_along_dim(0)

    @staticmethod
    def store(in_array, out_array, block_shape):
        """
        Partitions the array into blocks and writes them sequentially to the output array.
        """
        for sl in BlockedArrayWriter.gen_slices(in_array.shape, block_shape):
            block = in_array[sl]
            dask.array.store(block, out_array, regions=sl, lock=False, compute=True, return_stored=False)

def store_array(arr, group, path, block_shape, compressor=None, dimension_separator="/"):
    """
    Stores the full-resolution layer of a Dask pyramid into a Zarr group.
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

def downsample_and_store(arr, group, n_lvls, scale_factors, block_shape, compressor=None):
    """
    Progressively downsamples the input array and stores the results as separate arrays in a Zarr group.
    """
    pyramid = [arr]
    for arr_index in range(1, n_lvls):
        downsampled = dask.array.coarsen(np.mean, arr, {dim: factor for dim, factor in enumerate(scale_factors)}, trim_excess=True)
        ds = group.create_dataset(
            str(arr_index),
            shape=downsampled.shape,
            chunks=downsampled.chunksize,
            dtype=downsampled.dtype,
            compressor=compressor,
            dimension_separator="/",
            overwrite=True,
        )
        BlockedArrayWriter.store(downsampled, ds, block_shape)
        arr = dask.array.from_array(ds, chunks=ds.chunks)
        pyramid.append(arr)
    return pyramid

def _get_bytes(data):
    """
    Computes the total number of bytes in the given data.
    """
    if isinstance(data, list):
        return sum(arr.nbytes for arr in data)
    return data.nbytes

def write_ome_ngff_metadata(group, arr, image_name, n_lvls, scale_factors, voxel_size, origin=None):
    """
    Writes OME-NGFF metadata to a Zarr group.
    """
    axes = [
        {"name": "t", "type": "time", "unit": "millisecond"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    datasets = [{"path": str(i)} for i in range(n_lvls)]
    for i, scale in enumerate(scale_factors):
        # datasets[i]["coordinateTransformations"] = [{"type": "scale", "scale": [1.0, 1.0, *scale]}]
        datasets[i]["coordinateTransformations"] = [{"type": "scale", "scale": [1.0, 1.0, scale]}]

    group.attrs["multiscales"] = [{"version": "0.4", "datasets": datasets, "axes": axes}]

def run_multiscale(full_res_arr: dask.array, 
                   out_group: zarr.group,
                   voxel_sizes_zyx: tuple): 
    
    arr = full_res_arr.rechunk((1, 1, 128, 128, 128))
    arr = ensure_array_5d(arr)
    LOGGER.info(f"input array: {arr}")
    LOGGER.info(f"input array size: {arr.nbytes / 2 ** 20} MiB")
    block_shape = ensure_shape_5d(BlockedArrayWriter.get_block_shape(arr))
    LOGGER.info(f"block shape: {block_shape}")
    
    scale_factors = (2, 2, 2) 
    scale_factors = ensure_shape_5d(scale_factors)
    # n_levels = 5
    n_levels = 8   # Need 8 levels for exaspim 
    compressor = None

    # Actual Processing
    write_ome_ngff_metadata(
            out_group,
            arr,
            out_group.path,
            n_levels,
            scale_factors[2:],
            voxel_sizes_zyx,
            origin=None,
        )

    t0 = time.time()
    
    store_array(arr, out_group, "0", block_shape, compressor)
    pyramid = downsample_and_store(
        arr, out_group, n_levels, scale_factors, block_shape, compressor
    )
    write_time = time.time() - t0

    LOGGER.info(
        f"Finished writing tile.\n"
        f"Took {write_time}s. {_get_bytes(pyramid) / write_time / (1024 ** 2)} MiB/s"
    )

def read_config_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict

def write_config_yaml(yaml_path: str, yaml_data: dict) -> None:
    with open(yaml_path, "w") as file:
        yaml.dump(yaml_data, file)

def multiscale(): 
    # Locate logging yaml, use multiscale_input.yml in the same directory.
    yml_path = '/Users/seanfite/Desktop/Fusion/multiscale_input.yml'
    # yml_path = str(Path(__file__).parent / 'multiscale_input.yml')
    params = read_config_yaml(yml_path)
    in_path = params['in_path']
    output_path = params['output_path']
    voxel_sizes_zyx = tuple(params['resolution_zyx'])

    # Initalize cluster.
    client = Client(LocalCluster(n_workers=64, threads_per_worker=1, processes=True))

    # Run multiscaling
    arr = zarr.open(in_path + '/0', mode='r')
    arr = dask.array.from_zarr(arr)

    s3 = s3fs.S3FileSystem(
        config_kwargs={
            'max_pool_connections': 64,
            'retries': {
                'mode': 'standard',
            }
        }
    )
    store = s3fs.S3Map(root=output_path, s3=s3)
    out_group = zarr.group(store=store, overwrite=True)

    # out_group = zarr.open_group(output_path, mode='w')
    run_multiscale(arr, out_group, voxel_sizes_zyx)    

    # Dump 'Done' file w/ Unique ID
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time() * 1000)
    unique_file_name = str(Path('/results') / f"file_{timestamp}_{unique_id}.yml")

    output_data = {}
    output_data['output_path'] = output_path
    output_data['resolution_zyx'] = params['resolution_zyx']
    write_config_yaml(yaml_path=unique_file_name, 
                      yaml_data=output_data)


if __name__ == '__main__':
    multiscale()