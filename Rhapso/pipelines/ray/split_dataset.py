from Rhapso.split_dataset.xml_to_dataframe_split import XMLToDataFrameSplit
from Rhapso.split_dataset.compute_grid_rules import ComputeGridRules
from Rhapso.split_dataset.split_images import SplitImages
from Rhapso.split_dataset.save_xml import SaveXML
from Rhapso.split_dataset.save_points import SavePoints
import boto3

class SplitDataset:
    def __init__(self, xml_file_path, xml_output_file_path, n5_path, point_density, min_points, max_points, error, exclude_radius, 
                 target_image_size, target_overlap):
        self.xml_file_path = xml_file_path
        self.xml_output_file_path = xml_output_file_path
        self.n5_path = n5_path
        self.point_density = point_density
        self.min_points = min_points
        self.max_points = max_points
        self.error = error
        self.exclude_radius = exclude_radius
        self.target_image_size = target_image_size
        self.target_overlap = target_overlap
       
    def split(self):      
        if self.xml_file_path.startswith("s3://"):
            no_scheme = self.xml_file_path.replace("s3://", "", 1)
            bucket, key = no_scheme.split("/", 1)
            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=bucket, Key=key)
            xml_file = response["Body"].read().decode("utf-8")
        else:  
            with open(self.xml_file_path, "r", encoding="utf-8") as f:
                xml_file = f.read()

        xml_to_dataframe = XMLToDataFrameSplit(xml_file)
        data_global = xml_to_dataframe.run()

        split = ComputeGridRules(data_global, self.target_image_size, self.target_overlap)
        xyz_size, xyz_overlap, min_step_size = split.run()

        split_images = SplitImages(xyz_size, xyz_overlap, min_step_size, data_global, self.n5_path, self.point_density, self.min_points, self.max_points, 
                                   self.error, self.exclude_radius)
        new_split_interest_points, self_definition = split_images.run()

        save_xml = SaveXML(data_global, new_split_interest_points, self_definition, xml_file, self.xml_output_file_path)
        save_xml.run()

        save_points = SavePoints(
            new_split_interest_points,
            self.n5_path
        )

        save_points.run()
        print("Split Dataset Complete")
    
    def run(self):
        self.split()
