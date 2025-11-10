"""
NOTE:
Codebase intended for GPU/CPU device.
No fallback to CPU written until required.
"""

import xml.etree.ElementTree as ET
import Rhapso.fusion_old.aind_cloud_fusion.fusion as fusion
import Rhapso.fusion_old.aind_cloud_fusion.input_output as input_output
import xml.etree.ElementTree as ET
import boto3
from io import BytesIO

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

def execute_job(xml_path, input_path, output_s3_path, channel):
    # Prep inputs

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

if __name__ == '__main__':
    # xml_path = "s3://aind-open-data/HCR_802704_2025-08-30_02-00-00_processed_2025-10-01_21-09-24/image_tile_alignment/bigstitcher.xml"
    # input_path = "s3://aind-open-data/HCR_802704_2025-08-30_02-00-00_processed_2025-10-01_21-09-24/image_radial_correction/"
    xml_path = "s3://aind-open-data/exaSPIM_720164_2025-07-07_17-55-45_processed_2025-07-15_16-22-02/tile_alignment/ip_affine_alignment/bigstitcher_affine.xml"
    input_path = "s3://aind-open-data/exaSPIM_720164_2025-07-07_17-55-45_processed_2025-07-15_16-22-02/flatfield_correction/SPIM.ome.zarr/"
    output_s3_path = "s3://sean-fusion/exaSPIM_output1/channel_488.zarr"
    channel = 488

    execute_job(xml_path,
                input_path,
                output_s3_path,
                channel)