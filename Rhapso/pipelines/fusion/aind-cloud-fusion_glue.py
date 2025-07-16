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

import aind_cloud_fusion.fusion as fusion
import aind_cloud_fusion.io as io
import aind_cloud_fusion.script_utils as utils

def get_tile_zyx_resolution(input_xml_path: str) -> list[int]: 
    """
    Parse tile resolution to store in ome_ngff metadata
    """
    tree = ET.parse(input_xml_path)
    root = tree.getroot()

    res_xyz = root.find('SequenceDescription').find('ViewSetups').find('ViewSetup').find('voxelSize').find('size').text
    res_zyx = [float(num) for num in res_xyz.split(' ')[::-1]]
    
    return res_zyx

def execute_job(xml_path, output_path):
    # Hard-coded configuration values
    channel = 2
    input_path = "s3://aind-open-data/HCR_BL6-000_2023-07-11_00-01-00/radial_correction.ome.zarr/"
    output_s3_path = "s3://aind-open-data/HCR_BL6-000_2023-07-11_00-01-00/fused_full_res/channel_2.zarr"
    
    resolution_zyx = get_tile_zyx_resolution(xml_path)
    output_params = io.OutputParameters(
        path=output_s3_path,
        resolution_zyx=resolution_zyx
    )
    blend_option = 'weighted_linear_blending'

    # Run fusion
    fusion.run_fusion(
            input_path,
            xml_path,
            channel,
            output_params,
            blend_option
    )

    # Log 'done' file for next capsule in pipeline.
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time() * 1000)
    unique_file_name = str(Path(output_path) / f"file_{timestamp}_{unique_id}.yml")

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
        xml_path = str(glob.glob('../data/*.xml')[0])
        output_path = str(os.path.abspath('../results'))
    except: #running in capsule
        xml_path = str(glob.glob('/data/**/*.xml')[0])
        output_path = str(os.path.abspath('/results'))

    print(f'{xml_path=}')
    print(f'{output_path=}')

    execute_job(xml_path, output_path)
