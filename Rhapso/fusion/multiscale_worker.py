"""
Worker script to run multiscale conversion on a zarr dataset
"""

import os
import sys
import time
from pathlib import Path
import dask.array as da
import logging

from Rhapso.fusion.multiscale.aind_z1_radial_correction.array_to_zarr import convert_array_to_zarr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run():
    """
    Main run function for multiscale conversion
    """
    # Start timing
    start_time = time.time()
    
    # Input and output paths
    input_zarr_path = "s3://martin-test-bucket/output7/channel_488.zarr"
    output_zarr_path = "s3://martin-test-bucket/output10/multiscale_channel_488.zarr"
    
    # Set parameters for multiscale conversion
    # Adjust these parameters based on your data characteristics
    chunk_size = [128, 128, 128]  # Chunk size for the output zarr
    voxel_size = [1.0, 1.0, 1.0]  # Voxel size in micrometers (adjust if known)
    n_lvls = 6  # Number of pyramid levels
    scale_factor = [2, 2, 2]  # Downsampling factor per level
    target_block_size_mb = 512  # Target size for each processing block in MB (reduced from 512 to reduce memory pressure)
    use_ray = True  # Use Ray for parallel processing (set to False for sequential processing)
    ray_num_cpus = 12  # Limit Ray to 4 CPUs to prevent OOM (reduced from 16)

    logger.info(f"Starting multiscale conversion")
    logger.info(f"Input: {input_zarr_path}")
    logger.info(f"Output: {output_zarr_path}")
    
    # Load the zarr dataset
    # Assuming the data is in the root or scale "0" of the zarr
    try:
        # Try loading from scale "0" first (common for OME-Zarr)
        logger.info(f"Attempting to load from {input_zarr_path}/0...")
        sys.stdout.flush()
        dataset = da.from_zarr(f"{input_zarr_path}/0")
        logger.info(f"Successfully loaded data from {input_zarr_path}/0")
    except Exception as e:
        # If scale "0" doesn't exist, try loading from root
        logger.warning(f"Could not load from scale 0: {e}")
        try:
            logger.info(f"Attempting to load from {input_zarr_path}...")
            sys.stdout.flush()
            dataset = da.from_zarr(input_zarr_path)
            logger.info(f"Successfully loaded data from {input_zarr_path}")
        except Exception as e2:
            logger.error(f"Failed to load data: {e2}")
            raise ValueError(
                f"Could not load data from {input_zarr_path} or {input_zarr_path}/0. Error: {e2}"
            )
    
    logger.info(f"Dataset shape: {dataset.shape}")
    logger.info(f"Dataset dtype: {dataset.dtype}")
    logger.info(f"Dataset chunks: {dataset.chunks}")
    
    # Calculate dataset size
    import numpy as np
    dtype_bytes = np.dtype(dataset.dtype).itemsize
    total_size_gb = np.prod(dataset.shape) * dtype_bytes / (1024**3)
    logger.info(f"Dataset size: {total_size_gb:.2f} GB")
    
    # Use dask array directly instead of computing (don't load into memory)
    # Original implementation (loads entire array into memory - causes OOM for large datasets):
    # array = dataset.compute()
    
    # New implementation (keeps lazy evaluation for memory efficiency):
    logger.info("Using Dask array for lazy/chunked processing (not loading into memory)")
    array = dataset
    
    
    compressor_kwargs = {
        "cname": "zstd",
        "clevel": 3,
        "shuffle": 2,  # Blosc.SHUFFLE
    }
    
    logger.info("=" * 60)
    logger.info("Starting multiscale conversion with parameters:")
    logger.info(f"  Output path: {output_zarr_path}")
    logger.info(f"  Chunk size: {chunk_size}")
    logger.info(f"  Voxel size: {voxel_size}")
    logger.info(f"  Number of levels: {n_lvls}")
    logger.info(f"  Scale factor: {scale_factor}")
    logger.info(f"  Target block size: {target_block_size_mb} MB")
    logger.info(f"    → Each block will be ~{target_block_size_mb} MB in memory")
    logger.info(f"    → Larger blocks = fewer blocks but slower per block")
    logger.info(f"    → Smaller blocks = more blocks but faster per block")
    logger.info(f"    → Estimated blocks for level 0: ~{int(total_size_gb * 1024 / target_block_size_mb)}")
    if use_ray:
        logger.info(f"  Parallel processing: ENABLED (Ray with {ray_num_cpus} CPUs)")
        logger.info(f"    → Max parallel memory usage: ~{ray_num_cpus * target_block_size_mb} MB")
    else:
        logger.info(f"  Parallel processing: DISABLED (Sequential)")
    logger.info("=" * 60)
    sys.stdout.flush()
    
    # Convert to multiscale zarr with comprehensive error handling
    try:
        convert_array_to_zarr(
            array=array,
            chunk_size=chunk_size,
            output_path=output_zarr_path,
            voxel_size=voxel_size,
            n_lvls=n_lvls,
            scale_factor=scale_factor,
            compressor_kwargs=compressor_kwargs,
            target_block_size_mb=target_block_size_mb,
            use_ray=use_ray,
            ray_num_cpus=ray_num_cpus,
        )
    except MemoryError as e:
        elapsed_seconds = time.time() - start_time
        logger.error("=" * 60)
        logger.error("MEMORY ERROR: Out of memory!")
        logger.error("This typically happens when target_block_size_mb is too large.")
        logger.error(f"Try reducing target_block_size_mb (currently {target_block_size_mb} MB).")
        logger.error(f"Error details: {e}")
        logger.error(f"Failed after: {elapsed_seconds:.1f} seconds ({elapsed_seconds/60:.2f} minutes)")
        logger.error("=" * 60)
        sys.stdout.flush()
        raise
    except TimeoutError as e:
        elapsed_seconds = time.time() - start_time
        logger.error("=" * 60)
        logger.error("TIMEOUT ERROR: Operation timed out!")
        logger.error("This can happen with very large blocks or slow S3 connections.")
        logger.error(f"Current target_block_size_mb: {target_block_size_mb} MB")
        logger.error(f"Error details: {e}")
        logger.error(f"Failed after: {elapsed_seconds:.1f} seconds ({elapsed_seconds/60:.2f} minutes)")
        logger.error("=" * 60)
        sys.stdout.flush()
        raise
    except Exception as e:
        elapsed_seconds = time.time() - start_time
        logger.error("=" * 60)
        logger.error("CONVERSION FAILED!")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {e}")
        logger.exception("Full traceback:")
        logger.error("=" * 60)
        logger.error("Troubleshooting tips:")
        logger.error("  1. Check if S3 bucket is accessible")
        logger.error("  2. Verify AWS credentials are valid")
        logger.error(f"  3. Try reducing target_block_size_mb if blocks are too large (currently {target_block_size_mb} MB)")
        logger.error("  4. Check available memory and disk space")
        logger.error(f"Failed after: {elapsed_seconds:.1f} seconds ({elapsed_seconds/60:.2f} minutes)")
        logger.error("=" * 60)
        sys.stdout.flush()
        raise
    
    # Calculate elapsed time
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60
    elapsed_hours = elapsed_minutes / 60
    
    # Format elapsed time nicely
    if elapsed_hours >= 1:
        time_str = f"{elapsed_hours:.2f} hours ({elapsed_minutes:.1f} minutes)"
    elif elapsed_minutes >= 1:
        time_str = f"{elapsed_minutes:.2f} minutes ({elapsed_seconds:.1f} seconds)"
    else:
        time_str = f"{elapsed_seconds:.2f} seconds"
    
    logger.info("=" * 60)
    logger.info("MULTISCALE CONVERSION COMPLETED SUCCESSFULLY!")
    logger.info(f"Output written to: {output_zarr_path}")
    logger.info(f"Total time: {time_str}")
    logger.info("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    run()
