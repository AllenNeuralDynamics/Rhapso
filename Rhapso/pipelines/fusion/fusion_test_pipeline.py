"""
How to run this file:
1. install rhapso locally:
    - In the root of this github repo, run 'pip install -e .'
2. Install dependencies:
    - In the root of this github repo, run 'pip install -r requirements.txt'
3. Run this file:
    - Change the paths to point to your input/output
    - Run the following command in your terminal: 'python Rhapso/pipelines/fusion/fusion_test_pipeline.py' 
"""

"""
NOTE:
Codebase intended for GPU/CPU device.
No fallback to CPU written until required.
"""

import glob
import os
import uuid
import time
import multiprocessing as mp
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

import torch

'''
pyyaml
torch
pandas
'''

from Rhapso.fusion.main import run_fusion
from Rhapso.fusion.io import BigStitcherDataset, OutputParameters, RuntimeParameters
from Rhapso.fusion.script_utils import read_config_yaml
from Rhapso.fusion.blend import WeightedLinearBlending
from Rhapso.fusion.geometry import aabb_3d

def get_tile_zyx_resolution(input_xml_path: str) -> list[int]: 
    """
    Parse tile resolution to store in ome_ngff metadata
    """
    tree = ET.parse(input_xml_path)
    root = tree.getroot()

    res_xyz = root.find('SequenceDescription').find('ViewSetups').find('ViewSetup').find('voxelSize').find('size').text
    res_zyx = [float(num) for num in res_xyz.split(' ')[::-1]]
    
    return res_zyx

def execute_job(yml_path, xml_path, output_path):
    # Prep inputs
    configs = read_config_yaml(yml_path)
    input_path = configs['input_path']
    output_s3_path = configs['output_path']
    channel = configs['channel']

    resolution_zyx = get_tile_zyx_resolution(xml_path)
    output_params = OutputParameters(
        path=output_s3_path,
        resolution_zyx=resolution_zyx,
        chunksize=(1, 1, 128, 128, 128),  # Standard 5D chunk size (t,c,z,y,x)
        datastore=0  # Using Dask (0) instead of Tensorstore (1)
    )
    
    # Create the blending module - Modified to use actual input path instead of s3_path
    dataset = BigStitcherDataset(xml_path, input_path, 0)  # Using Dask (0)
    
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

    # Run fusion
    run_fusion(
        dataset,
        output_params,
        RuntimeParameters(0, 1, cells),  # Single process execution with calculated cells
        (128, 128, 128),  # cell_size
        [],  # No post-registration transforms
        blend_module
    )

    # Print the location of the .zattrs file
    zattrs_path = os.path.join(output_params.path, ".zattrs")
    print(f"Fusion completed. .zattrs file should be located at: {zattrs_path}")

    # Check if .zattrs exists, and create it if missing
    if not os.path.exists(zattrs_path):
        print(f".zattrs file is missing. Creating a placeholder at: {zattrs_path}")
        zattrs_content = "{}"  # Placeholder content for .zattrs
        print(f"Contents of .zattrs file: {zattrs_content}")
        with open(zattrs_path, "w") as zattrs_file:
            zattrs_file.write(zattrs_content)

    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Log 'done' file for next capsule in pipeline.
    # Unique log filename
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time() * 1000)
    unique_file_name = str(Path(output_path) / f"file_{timestamp}_{unique_id}.yml")

    # Downstream in_path is this capsule's output:
    # in_path ex: 
    # s3://aind-open-data/HCR_BL6-000_2023-07-11_00-01-00/fused_full_res/channel_0.zarr    
    # output_path ex:
    # s3://aind-open-data/HCR_BL6-000_2023-07-11_00-01-00/fused/channel_0.zarr
    log_content = {}
    log_content['in_path'] = output_params.path
    log_content['output_path'] = output_params.path.replace("fused_full_res", "fused")
    log_content['resolution_zyx'] = list(output_params.resolution_zyx)

    with open(unique_file_name, "w") as file:
        yaml.dump(log_content, file)


if __name__ == '__main__':
    # Some configurations helpful for GPU processing.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    print(mp.get_start_method(allow_none=False))
    print(mp.set_start_method('forkserver', force=True))
    print(mp.get_start_method(allow_none=False))
    torch.cuda.empty_cache()

    try: #running in pipeline 
        xml_path = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/IP_TIFF_XML/4. IP_TIFF_XML (after fusion)/dataset.xml"
        yml_path = "/home/martin/Documents/Allen/repos/aind-cloud-fusion/data/fusion_worker_config.yml"
        output_path = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/IP_TIFF_XML/4. IP_TIFF_XML (after fusion)/fused_output"
    except: #running in capsule
        xml_path = str(glob.glob('/data/**/*.xml')[0])
        yml_path = str(glob.glob('/data/*.yml')[0])
        output_path = str(os.path.abspath('/results'))

    print(f'{xml_path=}')
    print(f'{yml_path=}')
    print(f'{output_path=}')

    execute_job(yml_path,
                xml_path,
                output_path)
