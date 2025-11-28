"""
This module defines a class that takes
big chunks (compilation of chunks) from
a dask array and writes it on disk in
zarr format
"""

import logging
from typing import Generator, Tuple

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike

try:
    import ray
    RAY_AVAILABLE = True
    
    @ray.remote
    def _process_block_remote(in_array, out_array, sl, block_idx, total_blocks):
        """
        Ray remote function to process a single block.
        Reads block from input, computes if lazy, writes to output.
        
        Returns the block index for progress tracking.
        """
        # Read block from input array
        block = in_array[sl]
        
        # If it's a dask array, compute it to get numpy array
        if hasattr(block, 'compute'):
            block = block.compute()
        
        # Write block directly to output array (writes to S3)
        out_array[sl] = block
        
        return block_idx
    
except ImportError:
    RAY_AVAILABLE = False
    _process_block_remote = None  # Not available


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
    return np.prod(shape) * itemsize


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
    size1 = float(_get_size(shape1, itemsize))
    size2 = float(_get_size(shape2, itemsize))
    if abs(size1 - target_bytes) < abs(size2 - target_bytes):
        return shape1
    return shape2


def expand_chunks(  # noqa: C901
    chunks: Tuple[int, int, int],
    data_shape: Tuple[int, int, int],
    target_size: int,
    itemsize: int,
    mode: str = "iso",
) -> Tuple[int, int, int]:
    """
    Given the shape and chunk size of a pre-chunked 3D array,
    determine the optimal chunk shape closest to target_size.
    Expanded chunk dimensions are an integer multiple of
    the base chunk dimension, to ensure optimal access patterns.

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


class BlockedArrayWriter:
    """
    Static class to write a lazy array
    in big chunks to OMEZarr
    """

    @staticmethod
    def gen_slices(
        arr_shape: Tuple[int, ...], block_shape: Tuple[int, ...]
    ) -> Generator:
        """
        Generate a series of slices that can be
        used to traverse an array in blocks of a given shape.

        The method generates tuples of slices, each representing
        a block of the array. The blocks are generated by
        iterating over the array in steps of the block
        shape along each dimension.

        Parameters
        ----------
        arr_shape : tuple of int
            The shape of the array to be sliced.

        block_shape : tuple of int
            The desired shape of the blocks. This should be a
            tuple of integers representing the size of each
            dimension of the block. The length of `block_shape`
            should be equal to the length of `arr_shape`.
            If the array shape is not divisible by the block
            shape along a dimension, the last slice
            along that dimension is truncated.

        Returns
        -------
        generator of tuple of slice
            A generator yielding tuples of slices.
            Each tuple can be used to index the input array.
        """
        if len(arr_shape) != len(block_shape):
            raise Exception(
                "array shape and block shape have different lengths"
            )

        def _slice_along_dim(dim: int) -> Generator:
            """
            A helper generator function that
            slices along one dimension.
            """
            # Base case: if the dimension is beyond
            # the last one, return an empty tuple
            if dim >= len(arr_shape):
                yield ()
            else:
                # Iterate over the current dimension in steps of the block size
                for i in range(0, arr_shape[dim], block_shape[dim]):
                    # Calculate the end index for this block
                    end_i = min(i + block_shape[dim], arr_shape[dim])
                    # Generate slices for the remaining dimensions
                    for rest in _slice_along_dim(dim + 1):
                        yield (slice(i, end_i),) + rest

        # Start slicing along the first dimension
        return _slice_along_dim(0)

    @staticmethod
    def store(
        in_array: ArrayLike, out_array: ArrayLike, block_shape: tuple, use_ray: bool = True, ray_num_cpus: int = None
    ) -> None:
        """
        Partitions the last 3 dimensions of an array
        into non-overlapping blocks and writes them to a Zarr array.
        Can use Ray for parallel processing or sequential processing.

        :param in_array: The input array (can be dask array or numpy array)
        :param block_shape: Tuple of (block_depth, block_height, block_width)
        :param out_array: The output array
        :param use_ray: If True, use Ray for parallel processing. If False, use sequential processing.
        :param ray_num_cpus: Number of CPUs to use for Ray. If None, uses all available CPUs.
        """
        import sys
        logger = logging.getLogger(__name__)
        
        # Calculate total number of blocks for progress tracking
        total_blocks = 1
        for arr_dim, block_dim in zip(in_array.shape, block_shape):
            total_blocks *= (arr_dim + block_dim - 1) // block_dim
        
        # Check if Ray should be used
        use_ray = use_ray and RAY_AVAILABLE
        
        if use_ray:
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                logger.info("   Initializing Ray for parallel processing...")
                
                # Configure Ray with memory limits and object spilling
                import os
                
                # Set environment variables for Ray memory management BEFORE initialization
                # These must be set before ray.init() is called
                os.environ.setdefault("RAY_memory_monitor_refresh_ms", "250")
                os.environ.setdefault("RAY_memory_usage_threshold", "0.85")  # Kill workers at 85% memory usage
                
                ray_config = {
                    "ignore_reinit_error": True,
                    "object_store_memory": int(8 * 1024 * 1024 * 1024),  # 8 GB for object store
                }
                
                # Set number of CPUs if specified
                if ray_num_cpus is not None:
                    ray_config["num_cpus"] = ray_num_cpus
                    logger.info(f"   Limiting Ray to {ray_num_cpus} CPUs to prevent OOM")
                else:
                    ray_config["num_cpus"] = None  # Use all available CPUs
                
                ray.init(**ray_config)
                actual_cpus = ray.cluster_resources().get('CPU', 0)
                logger.info(f"   Ray initialized with {actual_cpus} CPUs and 8 GB object store")
                logger.info(f"   Memory monitor will kill workers at 85% memory usage")
            
            logger.info(f"   Writing {total_blocks} blocks IN PARALLEL using Ray (block shape: {block_shape})...")
            sys.stdout.flush()
            
            # Submit all blocks as Ray tasks
            futures = []
            block_idx = 0
            for sl in BlockedArrayWriter.gen_slices(in_array.shape, block_shape):
                future = _process_block_remote.remote(in_array, out_array, sl, block_idx, total_blocks)
                futures.append(future)
                block_idx += 1
            
            # Wait for all tasks to complete and show progress
            completed = 0
            while futures:
                # Wait for at least one task to complete
                done, futures = ray.wait(futures, num_returns=1, timeout=1.0)
                completed += len(done)
                if done:
                    progress_pct = (completed / total_blocks) * 100
                    logger.info(f"   Progress: {completed}/{total_blocks} blocks ({progress_pct:.1f}%)")
                    sys.stdout.flush()
            
            logger.info(f"   ✓ All {total_blocks} blocks written successfully using Ray!")
            
        else:
            # Sequential processing (original implementation)
            if not RAY_AVAILABLE:
                logger.warning("   Ray not available, falling back to sequential processing")
            else:
                logger.info("   Using sequential processing (Ray disabled)")
            
            logger.info(f"   Writing {total_blocks} blocks SEQUENTIALLY (block shape: {block_shape})...")
            sys.stdout.flush()
            
            block_idx = 0
            for sl in BlockedArrayWriter.gen_slices(in_array.shape, block_shape):
                logger.info(f"   Progress: {block_idx}/{total_blocks} blocks ({(block_idx/total_blocks)*100:.1f}%)")
                block_idx += 1
                
                # Read block from input array
                block = in_array[sl]
                
                # If it's a dask array, compute it to get numpy array
                if hasattr(block, 'compute'):
                    block = block.compute()
                
                # Write block directly to output array (writes to S3)
                out_array[sl] = block
            
            logger.info(f"   ✓ All {total_blocks} blocks written successfully!")
        
        sys.stdout.flush()

    @staticmethod
    def get_block_shape(arr, target_block_size_mb=409600, mode="cycle", chunks=None):
        """
        Given the shape and chunk size of a pre-chunked
        array, determine the optimal block shape closest
        to target block size. Expanded block dimensions are
        an integer multiple of the chunk dimension
        to ensure optimal access patterns.

        Args:
            arr: the input array
            target_block_size_mb: target block size in megabytes,
            default is 409600 mode: strategy.
            Must be one of "cycle", or "iso"

        Returns:
            the block shape
        """

        if chunks is None:
            if isinstance(arr, da.Array):
                chunks = arr.chunksize
            else:
                chunks = arr.chunks

        chunks = chunks[-3:]
        return expand_chunks(
            chunks,
            arr.shape[-3:],
            target_block_size_mb * 1024**2,
            arr.itemsize,
            mode,
        )
