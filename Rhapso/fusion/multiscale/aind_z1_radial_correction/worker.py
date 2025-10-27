"""
Worker script to run multiscale conversion on a zarr dataset
"""

import os
import sys
from pathlib import Path
import dask.array as da

from Rhapso.fusion.multiscale.aind_z1_radial_correction.array_to_zarr import convert_array_to_zarr
from Rhapso.fusion.multiscale.aind_z1_radial_correction.utils import utils


def run():
    """
    Main run function for multiscale conversion
    """
    
    # Input and output paths
    input_zarr_path = "s3://martin-test-bucket/output/channel_488.zarr"
    output_zarr_path = "s3://martin-test-bucket/output/multiscale_channel_488.zarr"
    
    print(f"Loading data from: {input_zarr_path}")
    
    # Load the zarr dataset
    # Assuming the data is in the root or scale "0" of the zarr
    try:
        # Try loading from scale "0" first (common for OME-Zarr)
        dataset = da.from_zarr(f"{input_zarr_path}/0")
        print(f"Loaded data from {input_zarr_path}/0")
    except Exception as e:
        # If scale "0" doesn't exist, try loading from root
        print(f"Could not load from scale 0: {e}")
        try:
            dataset = da.from_zarr(input_zarr_path)
            print(f"Loaded data from {input_zarr_path}")
        except Exception as e2:
            raise ValueError(
                f"Could not load data from {input_zarr_path} or {input_zarr_path}/0. Error: {e2}"
            )
    
    print(f"Dataset shape: {dataset.shape}")
    print(f"Dataset dtype: {dataset.dtype}")
    print(f"Dataset chunks: {dataset.chunks}")
    
    # Compute the array (load into memory)
    # For large datasets, you might want to process this differently
    print("Computing array...")
    array = dataset.compute()
    print("Array computed successfully")
    
    # Set parameters for multiscale conversion
    # Adjust these parameters based on your data characteristics
    chunk_size = [128, 128, 128]  # Chunk size for the output zarr
    voxel_size = [1.0, 1.0, 1.0]  # Voxel size in micrometers (adjust if known)
    n_lvls = 6  # Number of pyramid levels
    scale_factor = [2, 2, 2]  # Downsampling factor per level
    
    compressor_kwargs = {
        "cname": "zstd",
        "clevel": 3,
        "shuffle": 2,  # Blosc.SHUFFLE
    }
    
    print(f"Converting to multiscale format and writing to: {output_zarr_path}")
    print(f"Chunk size: {chunk_size}")
    print(f"Voxel size: {voxel_size}")
    print(f"Number of levels: {n_lvls}")
    print(f"Scale factor: {scale_factor}")
    
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
    
    print(f"Multiscale conversion completed successfully!")
    print(f"Output written to: {output_zarr_path}")


if __name__ == "__main__":
    run()

