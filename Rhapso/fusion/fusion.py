import glob
import os
import uuid
import time
import multiprocessing as mp
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
import logging
import dask
from dask.distributed import Client, LocalCluster
import zarr
import torch
import numpy as np
from ome_zarr.io import parse_url
from aind_data_transfer.util.io_utils import BlockedArrayWriter
from aind_data_transfer.util.chunk_utils import ensure_shape_5d, ensure_array_5d
from aind_data_transfer.transformations.ome_zarr import (
    store_array,
    downsample_and_store,
    _get_bytes,
    write_ome_ngff_metadata
)

from Rhapso.fusion.main import run_fusion
from Rhapso.fusion.io import BigStitcherDataset, OutputParameters, RuntimeParameters
from Rhapso.fusion.blend import WeightedLinearBlending
from Rhapso.fusion.geometry import aabb_3d

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def get_tile_zyx_resolution(input_xml_path: str) -> list[int]: 
    """
    Parse tile resolution to store in ome_ngff metadata
    """
    tree = ET.parse(input_xml_path)
    root = tree.getroot()

    res_xyz = root.find('SequenceDescription').find('ViewSetups').find('ViewSetup').find('voxelSize').find('size').text
    res_zyx = [float(num) for num in res_xyz.split(' ')[::-1]]
    
    return res_zyx

def run_multiscale(full_res_arr: dask.array, 
                   out_group: zarr.group,
                   voxel_sizes_zyx: tuple): 
    """
    Generate multiscale representation of the fused output.
    """
    arr = full_res_arr.rechunk((1, 1, 128, 128, 128))
    arr = ensure_array_5d(arr)
    LOGGER.info(f"input array: {arr}")
    LOGGER.info(f"input array size: {arr.nbytes / 2 ** 20} MiB")
    block_shape = ensure_shape_5d(BlockedArrayWriter.get_block_shape(arr))
    LOGGER.info(f"block shape: {block_shape}")
    
    scale_factors = ensure_shape_5d(SCALE_FACTORS)
    compressor = None

    # Write metadata
    write_ome_ngff_metadata(
        out_group,
        arr,
        out_group.path,
        N_LEVELS,
        scale_factors[2:],
        voxel_sizes_zyx,
        origin=None,
    )

    t0 = time.time()
    store_array(arr, out_group, "0", block_shape, compressor)
    pyramid = downsample_and_store(
        arr, out_group, N_LEVELS, scale_factors, block_shape, compressor
    )
    write_time = time.time() - t0

    LOGGER.info(
        f"Finished writing multiscale data.\n"
        f"Took {write_time}s. {_get_bytes(pyramid) / write_time / (1024 ** 2)} MiB/s"
    )

def get_chunksize_from_dataset(dataset: BigStitcherDataset) -> tuple:
    """
    Dynamically fetch the chunk size from the input dataset.
    """
    first_tile = next(iter(dataset.tile_volumes_tczyx.values()))
    z, y, x = first_tile.shape[2:]  # Extract the spatial dimensions (z, y, x)
    return (1, 1, min(z, 128), min(y, 128), min(x, 128))  # Use a default max chunk size of 128

def execute_job():
    # Prep inputs
    resolution_zyx = get_tile_zyx_resolution(XML_PATH)
    dataset = BigStitcherDataset(XML_PATH, INPUT_PATH, 0)  # Using Dask (0)
    
    # Dynamically fetch the chunk size
    chunksize = get_chunksize_from_dataset(dataset)
    LOGGER.info(f"Using dynamic chunksize: {chunksize}")
    
    output_params = OutputParameters(
        path=OUTPUT_PATH,
        resolution_zyx=resolution_zyx,
        chunksize=chunksize,
        datastore=0  # Using Dask (0) instead of Tensorstore (1)
    )
    
    # Create the blending module
    dataset = BigStitcherDataset(XML_PATH, INPUT_PATH, 0)  # Using Dask (0)
    
    # Debug print statements
    print("Found tiles:", list(dataset.tile_volumes_tczyx.keys()))
    print("Found transforms:", list(dataset.tile_transforms_zyx.keys()))

    # Handle case where no tiles are loaded
    if not dataset.tile_volumes_tczyx:
        raise ValueError("No valid tiles were loaded. Please check the input paths and file formats.")
    
    # Calculate tile AABBs
    tile_sizes_zyx = {}
    tile_aabbs = {}
    tile_arrays = dataset.tile_volumes_tczyx
    tile_transforms = dataset.tile_transforms_zyx
    
    for tile_id, tile_arr in tile_arrays.items():
        zyx = tile_arr.shape[2:]
        tile_sizes_zyx[tile_id] = zyx
        
        # Calculate boundaries for each tile
        tile_boundaries = torch.Tensor([
            [0.0, 0.0, 0.0],
            [zyx[0], 0.0, 0.0],
            [0.0, zyx[1], 0.0],
            [0.0, 0.0, zyx[2]],
            [zyx[0], zyx[1], 0.0],
            [zyx[0], 0.0, zyx[2]],
            [0.0, zyx[1], zyx[2]],
            [zyx[0], zyx[1], zyx[2]],
        ])

        # Transform boundaries
        if tile_id in tile_transforms:
            tfm_list = tile_transforms[tile_id]
            for tfm in tfm_list:
                tile_boundaries = tfm.forward(tile_boundaries, device=torch.device("cpu"))

        # Calculate AABB
        tile_aabbs[tile_id] = aabb_3d(tile_boundaries)

    blend_module = WeightedLinearBlending(tile_aabbs)

    # Calculate cells based on dataset size
    first_tile = next(iter(dataset.tile_volumes_tczyx.values()))
    z, y, x = first_tile.shape[2:]
    cells = [(i,j,k) for i in range((z+127)//128) 
                     for j in range((y+127)//128)
                     for k in range((x+127)//128)]

    # Multiscale processing
    client = Client(LocalCluster(n_workers=1, threads_per_worker=1, processes=True))
    arr = zarr.open(OUTPUT_PATH + '/0', mode='r')
    arr = dask.array.from_zarr(arr)

    if not Path(MULTISCALE_OUTPUT_PATH).exists():
        Path(MULTISCALE_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    out_group = zarr.open_group(MULTISCALE_OUTPUT_PATH, mode='w')

    run_multiscale(arr, out_group, resolution_zyx)

    print(f"Fusion and multiscale processing completed. Multiscale output at: {MULTISCALE_OUTPUT_PATH}")


def runFusion(input_path, output_path, multiscale_output_path, xml_path, channel, cell_size, n_levels, scale_factors):
    """
    Run the fusion process with the provided parameters.
    """
    global INPUT_PATH, OUTPUT_PATH, MULTISCALE_OUTPUT_PATH, XML_PATH, CHANNEL, CELL_SIZE, N_LEVELS, SCALE_FACTORS

    INPUT_PATH = input_path
    OUTPUT_PATH = output_path
    MULTISCALE_OUTPUT_PATH = multiscale_output_path
    XML_PATH = xml_path
    CHANNEL = channel
    CELL_SIZE = cell_size
    N_LEVELS = n_levels
    SCALE_FACTORS = scale_factors

    print(f"Global variables set:")
    print(f"Input Path: {INPUT_PATH}")
    print(f"Output Path: {OUTPUT_PATH}")
    print(f"Multiscale Output Path: {MULTISCALE_OUTPUT_PATH}")
    print(f"XML Path: {XML_PATH}")
    print(f"Channel: {CHANNEL}")
    print(f"Cell Size: {CELL_SIZE}")
    print(f"Number of Levels: {N_LEVELS}")
    print(f"Scale Factors: {SCALE_FACTORS}")

    # Some configurations helpful for GPU processing.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    print(mp.get_start_method(allow_none=False))
    print(mp.set_start_method('forkserver', force=True))
    print(mp.get_start_method(allow_none=False))
    torch.cuda.empty_cache()

    execute_job()