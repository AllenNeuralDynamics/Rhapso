from typing import Tuple, Union, Iterator
import math
import dask.array as da
import numpy as np


class DimensionsError(Exception):
    pass


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


def ensure_shape_5d(shape: Tuple[int, ...]) -> Tuple[int, int, int, int, int]:
    """
    Checks that the shape is 5D, adding singleton dimensions to the
    start of the sequence if less, throwing a DimensionsError if more.
    Args:
        shape: the sequence
    Returns:
        the 5D tuple
    Raises:
        DimensionsError: if the input has more than 5 dimensions
    """
    if len(shape) > 5:
        raise DimensionsError("Only shapes up to 5D are supported")
    while len(shape) < 5:
        shape = (1, *shape)
    return shape


def guess_chunks(
    data_shape: Tuple[int, int, int],
    target_size: int,
    itemsize: int,
    mode: str = "z",
) -> Tuple[int, int, int]:
    """
    Given the shape of a 3D array, determine the optimal chunk shape
    closest to target_size.
    Args:
        data_shape: the shape of the input array
        target_size: target chunk size in bytes
        itemsize: the number of bytes per array element
        mode: chunking strategy. Must be one of "z", "cycle", or "iso"
    Returns:
        the optimal chunk shape
    """
    if any(s < 1 for s in data_shape):
        raise ValueError("data_shape must be >= 1 for all dimensions")
    if target_size <= 0:
        raise ValueError("target_size must be > 0")
    if itemsize <= 0:
        raise ValueError("itemsize must be > 0")
    if mode == "z":
        plane_size = data_shape[1] * data_shape[2] * itemsize
        nplanes_per_chunk = int(math.ceil(target_size / plane_size))
        nplanes_per_chunk = min(nplanes_per_chunk, data_shape[0])
        chunks = (
            nplanes_per_chunk,
            data_shape[1],
            data_shape[2],
        )
    elif mode == "cycle":
        current = np.array(data_shape, dtype=np.uint64)
        idx = 0
        ndims = len(current)
        while _get_size(current, itemsize) > target_size:
            current[idx % ndims] = int(math.ceil(current[idx % ndims] / 2.0))
            idx += 1
        chunks = (current[0], current[1], current[2])
    elif mode == "iso":
        chunk_dim = int(math.ceil((target_size / itemsize) ** (1.0 / 3)))
        chunks = (
            min(data_shape[0], chunk_dim),
            min(data_shape[1], chunk_dim),
            min(data_shape[2], chunk_dim),
        )
    else:
        raise ValueError(f"Invalid mode {mode}")

    # convert numpy int64 to Python int or zarr will complain
    return tuple(int(d) for d in chunks)


def expand_chunks(
    chunks: Tuple[int, int, int],
    data_shape: Tuple[int, int, int],
    target_size: int,
    itemsize: int,
    mode: str = "iso",
) -> Tuple[int, int, int]:
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
        # get the spatial dimensions only
        current = np.array(chunks, dtype=np.uint64)
        prev = current.copy()
        idx = 0
        ndims = len(current)
        while _get_size(current, itemsize) < target_size:
            prev = current.copy()
            current[idx % ndims] = min(
                data_shape[idx % ndims], current[idx % ndims] * 2
            )
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


# https://stackoverflow.com/a/47518679
def range_with_end(start: int, end: int, step: int) -> Iterator[int]:
    """Generator for a range function which always includes the end value.
    Args:
        start: the start value
        end: the end value (inclusive)
        step: the step size
    Returns:
        A generator of integers
    Example:
        >>> list(range_with_end(0, 10, 3))
        >>> [0, 3, 6, 9, 10]
    """
    i = start
    while i < end:
        yield i
        i += step
    yield end


def _closer_to_target(
    shape1: Tuple[int, ...],
    shape2: Tuple[int, ...],
    target_bytes: int,
    itemsize: int,
) -> Tuple[int, ...]:
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


def _get_size(shape: Tuple[int, ...], itemsize: int) -> int:
    """
    Return the size of an array with the given shape, in bytes
    Args:
        shape: the shape of the array
        itemsize: number of bytes per array element
    Returns:
        the size of the array, in bytes
    """
    if any(s <= 0 for s in shape):
        raise ValueError("shape must be > 0 in all dimensions")
    return np.product(shape) * itemsize