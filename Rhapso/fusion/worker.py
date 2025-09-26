"""
NOTE:
Codebase intended for GPU/CPU device.
No fallback to CPU written until required.
"""

import uuid
import time
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
import yaml
import Rhapso.fusion.aind_cloud_fusion.fusion as fusion
import Rhapso.fusion.aind_cloud_fusion.input_output as input_output
import Rhapso.fusion.aind_cloud_fusion.script_utils as utils
import xml.etree.ElementTree as ET
import boto3
from io import BytesIO
import os
import multiprocessing as mp

def get_tile_zyx_resolution(input_xml_path: str) -> list[int]: 
    """
    Parse tile resolution to store in ome_ngff metadata
    """
    if input_xml_path.startswith('s3://'):
        # Handle S3 path
        s3 = boto3.resource('s3')
        bucket_name, key = input_xml_path[5:].split('/', 1)
        bucket = s3.Bucket(bucket_name)
        obj = bucket.Object(key)
        response = obj.get()
        file_stream = BytesIO(response['Body'].read())
        tree = ET.parse(file_stream)
    else:
        tree = ET.parse(input_xml_path)
    
    root = tree.getroot()

    res_xyz = root.find('SequenceDescription').find('ViewSetups').find('ViewSetup').find('voxelSize').find('size').text
    res_zyx = [float(num) for num in res_xyz.split(' ')[::-1]]
    
    return res_zyx

def execute_job(yml_path, xml_path, output_path):
    # Prep inputs
    # configs = utils.read_config_yaml(yml_path)
    input_path = "s3://aind-open-data/exaSPIM_708365_2024-04-29_12-46-15/SPIM.ome.zarr/"
    output_s3_path = "s3://rhapso-matching-test/fusion-9-26/fused-output/"
    channel = 488

    resolution_zyx = get_tile_zyx_resolution(xml_path)
    output_params = input_output.OutputParameters(
        path=output_s3_path,
        resolution_zyx=resolution_zyx
    )
    blend_option = 'weighted_linear_blending'

    # Run fusion
    print(f'Starting fusion at: {datetime.now()}')
    fusion.run_fusion(
            input_path,
            xml_path,
            channel,
            output_params,
            blend_option
    )

    # Log 'done' file for next capsule in pipeline.
    # Unique log filename
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

    # Force CPU-only execution settings
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide all GPUs
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    print(f"Current multiprocessing start method: {mp.get_start_method(allow_none=False)}")
    print(f"Setting multiprocessing start method to 'forkserver': {mp.set_start_method('forkserver', force=True)}")
    print(f"New multiprocessing start method: {mp.get_start_method(allow_none=False)}")

    xml_path = "s3://rhapso-zar-sample/dataset.xml"
    yml_path = 'not using yml config (hard coded instead)'
    output_path = 's3://rhapso-matching-test/fusion-9-26/results/'

    print(f'{xml_path=}')
    print(f'{yml_path=}')
    print(f'{output_path=}')

    execute_job(yml_path,
                xml_path,
                output_path)