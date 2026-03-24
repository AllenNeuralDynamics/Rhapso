import Rhapso.translation_fusion.fusion as fusion
import Rhapso.translation_fusion.input_output as input_output
import xml.etree.ElementTree as ET
import boto3
from io import BytesIO

class TranslationFusion:
    def __init__(self, xml_path, input_path, output_s3_path, channel, default_chunk_size, cpu_cell_size):
        self.xml_path = xml_path
        self.input_path = input_path
        self.output_s3_path = output_s3_path
        self.channel = channel
        self.default_chunk_size = default_chunk_size
        self.cpu_cell_size = cpu_cell_size

    def get_tile_zyx_resolution(self) -> list[int]: 
        """
        Parse tile resolution to store in ome_ngff metadata
        """
        if self.xml_path.startswith('s3://'):
            s3 = boto3.resource('s3')
            bucket_name, key = self.xml_path[5:].split('/', 1)
            bucket = s3.Bucket(bucket_name)
            obj = bucket.Object(key)
            response = obj.get()
            file_stream = BytesIO(response['Body'].read())
            tree = ET.parse(file_stream)
        else:
            tree = ET.parse(self.xml_path)
        
        root = tree.getroot()

        res_xyz = root.find('SequenceDescription').find('ViewSetups').find('ViewSetup').find('voxelSize').find('size').text
        res_zyx = [float(num) for num in res_xyz.split(' ')[::-1]]
        
        return res_zyx

    def execute_job(self):
        resolution_zyx = self.get_tile_zyx_resolution()
        output_params = input_output.OutputParameters(
            path=self.output_s3_path,
            resolution_zyx=resolution_zyx
        )
        blend_option = 'weighted_linear_blending'

        fusion.run_fusion(
                self.input_path,
                self.xml_path,
                self.channel,
                output_params,
                blend_option,
                self.default_chunk_size,
                self.cpu_cell_size
        )
    
    def run(self):
        self.execute_job()