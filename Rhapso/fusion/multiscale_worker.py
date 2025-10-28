"""
Worker script to run multiscale conversion on a zarr dataset
"""

import os
import sys
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
    
    # Input and output paths
    input_zarr_path = "s3://martin-test-bucket/output7/channel_488.zarr"
    output_zarr_path = "s3://martin-test-bucket/output7/multiscale_channel_488.zarr"
    
    # Set parameters for multiscale conversion
    # Adjust these parameters based on your data characteristics
    chunk_size = [128, 128, 128]  # Chunk size for the output zarr
    voxel_size = [1.0, 1.0, 1.0]  # Voxel size in micrometers (adjust if known)
    n_lvls = 6  # Number of pyramid levels
    scale_factor = [2, 2, 2]  # Downsampling factor per level

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
    logger.info("=" * 60)
    sys.stdout.flush()
    
    # Convert to multiscale zarr
    convert_array_to_zarr(
        array=array,
        chunk_size=chunk_size,
        output_path=output_zarr_path,
        voxel_size=voxel_size,
        n_lvls=n_lvls,
        scale_factor=scale_factor,
        compressor_kwargs=compressor_kwargs,
        target_size_mb=24000,
    )
    
    logger.info("=" * 60)
    logger.info("MULTISCALE CONVERSION COMPLETED SUCCESSFULLY!")
    logger.info(f"Output written to: {output_zarr_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
