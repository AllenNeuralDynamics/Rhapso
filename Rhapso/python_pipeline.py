import boto3
import io
from data_preparation.xml_to_dataframe import XMLToDataFrame
from detection.view_transform_models import ViewTransformModels
from detection.overlap_detection import OverlapDetection
from detection.interest_point_detection import PythonInterestPointDetection
# from interest_point_detection.advanced_refinement import AdvancedRefinement
# from interest_point_detection.filtering_and_optimizing import FilteringAndOptimization
# from interest_point_detection.save_interest_points import SaveInterestPoints
# from data_preparation.dataframe_to_xml import DataFrameToXML

# Development testing pipeline / deployment template

# BUCKET NAME
xml_bucket_name = "rhapso-tif-sample"
# xml_bucket_name = "rhapso-zar-sample"
image_bucket_name = "aind-open-data"

# FILE KEY
xml_file_path = "IP_TIFF_XML/dataset.xml"
# xml_file_path = "dataset.xml"

# FILE PREFIX
prefix = '/Users/seanfite/Desktop/AllenInstitute/Rhapso/Data/IP_TIFF_XML/'
# prefix = 's3://rhapso-tif-sample/IP_TIFF_XML/'
# prefix = 's3://aind-open-data/exaSPIM_708365_2024-04-29_12-46-15/SPIM.ome.zarr/'
# TODO - local_zarr_image_data_prefix

# FILE TYPE
file_type = 'tiff'
# file_type = 'zarr'

# STRATEGY
strategy = 'python'

# DOWNSAMPLING
dsxy = 4
dsz = 2

# data input source
s3 = boto3.client('s3')

def fetch_from_s3(s3, bucket_name, input_file):
    response = s3.get_object(Bucket=bucket_name, Key=input_file)
    return response['Body'].read().decode('utf-8')

# INTEREST POINT DETECTION
# --------------------------

xml_file = fetch_from_s3(s3, xml_bucket_name, xml_file_path) 

# Load XML data into dataframes         
processor = XMLToDataFrame(xml_file)
dataframes = processor.run()
print("XML loaded")

# Create view transform matrices 
create_models = ViewTransformModels(dataframes)
view_transform_matrices = create_models.run()
print("Transforms Models have been created")

# Use view transform matrices to find areas of overlap
overlap_detection = OverlapDetection(view_transform_matrices, dataframes, dsxy, dsz, prefix, file_type)
overlapping_area = overlap_detection.run()
print("Overlap Detection is done")

# Use image data and areas of overlap to find interest points - (custom transform) 
interest_point_detection = PythonInterestPointDetection(dataframes, overlapping_area, dsxy, dsz, prefix, file_type, image_bucket_name)
interest_points = interest_point_detection.run()
print(interest_points)
print("Interest Point Detection is done")

# filtering_and_optimizing = FilteringAndOptimizing(interest_points)
# filtering_and_optimizing.run()

# advanced_refinement = AdvancedRefinement()
# advanced_refinement.run()

# save_interest_points = SaveInterestPoints()
# save_interest_points.run()

# INTEREST POINT MATCHING
# --------------------------

# SOLVE 
# --------------------------

# FUSION
# --------------------------
