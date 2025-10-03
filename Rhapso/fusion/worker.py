"""
NOTE:
Codebase intended for GPU/CPU device.
No fallback to CPU written until required.
"""

import uuid
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
# import aind_cloud_fusion.fusion as fusion
# import aind_cloud_fusion.input_output as input_output
# import aind_cloud_fusion.script_utils as utils
import xml.etree.ElementTree as ET
import boto3
from io import BytesIO
from .aind_cloud_fusion import fusion as fusion
from .aind_cloud_fusion import input_output as input_output
from .aind_cloud_fusion import script_utils as utils

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
    configs = utils.read_config_yaml(yml_path)
    input_path = configs['input_path']
    output_s3_path = configs['output_path']
    channel = configs['channel']

    resolution_zyx = get_tile_zyx_resolution(xml_path)
    output_params = input_output.OutputParameters(
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

    xml_path = 's3://rhapso-zar-sample/dataset.xml'
    yml_path = 's3://sean-fusion/worker_config.yml'
    output_path = 's3://sean-fusion/fusion-output'

    print(f'{xml_path=}')
    print(f'{yml_path=}')
    print(f'{output_path=}')

    execute_job(yml_path,
                xml_path,
                output_path)