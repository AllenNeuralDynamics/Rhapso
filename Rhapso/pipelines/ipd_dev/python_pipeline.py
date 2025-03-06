from ...data_prep.xml_to_dataframe import XMLToDataFrame
from ...detection.view_transform_models import ViewTransformModels
from ...detection.overlap_detection import OverlapDetection
from ...data_prep.load_image_data import LoadImageData
from ...detection.difference_of_gaussian import DifferenceOfGaussian
# from ...interest_point_detection.advanced_refinement import AdvancedRefinement
# from ...interest_point_detection.filtering_and_optimizing import FilteringAndOptimization
# from ...interest_point_detection.save_interest_points import SaveInterestPoints
# from ...data_prep.dataframe_to_xml import DataFrameToXML
import boto3
from dask import delayed
from dask import compute

strategy = 'python'
dsxy = 4
dsz = 2

# FILE TYPE - PICK ONE
file_type = 'tiff'
xml_bucket_name = "rhapso-tif-sample"
xml_file_path = "IP_TIFF_XML/dataset.xml"
prefix = '/Users/seanfite/Desktop/AllenInstitute/Rhapso/Data/IP_TIFF_XML/'
# prefix = 's3://rhapso-tif-sample/IP_TIFF_XML/'

# file_type = 'zarr'
# xml_bucket_name = "rhapso-zar-sample"
# image_bucket_name = "aind-open-data"
# xml_file_path = "dataset.xml"
# prefix = 's3://aind-open-data/exaSPIM_708365_2024-04-29_12-46-15/SPIM.ome.zarr/'
# TODO - local_zarr_image_data_prefix

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
print("Transforms models have been created")

# Use view transform matrices to find areas of overlap
overlap_detection = OverlapDetection(view_transform_matrices, dataframes, dsxy, dsz, prefix, file_type)
overlapping_area = overlap_detection.run()
print("Overlap detection is done")

# Load images
images_loader = LoadImageData(dataframes, overlapping_area, dsxy, dsz, prefix, file_type)
all_image_data = images_loader.run()
print("Loaded image data")

# Detect interest points using DoG algorithm
difference_of_gaussian = DifferenceOfGaussian()

# BASE PYTHON VERSION - SLOWEST RUN TIME
# def interest_point_detection(image_data):
#     interval_key = image_data['interval_key']
#     image_chunk = image_data['image_chunk']
#     interest_points = difference_of_gaussian.run(image_chunk)
#     return interval_key, interest_points
# mapped_results = map(interest_point_detection, all_image_data)
# interest_points = list(mapped_results)

# DASK MAP VERSION (SINGLE THREAD INTERPRETER ONLY) - 10X FASTER
interest_points = {}
delayed_results = []
delayed_keys = {} 
for image_data in all_image_data:
    interval_key = image_data['interval_key']
    image_chunk = image_data['image_chunk']
    dog_result = delayed(difference_of_gaussian.run)(image_chunk)
    delayed_results.append(dog_result)
    delayed_keys[dog_result] = interval_key
computed_results = compute(*delayed_results) 
for result, task in zip(computed_results, delayed_results):
    interval_key = delayed_keys[task]
    interest_points[interval_key] = result 

print(interest_points)
print("Interest point detection is done")

# # filtering_and_optimizing = FilteringAndOptimizing(interest_points)
# # filtering_and_optimizing.run()

# # advanced_refinement = AdvancedRefinement()
# # advanced_refinement.run()

# # save_interest_points = SaveInterestPoints()
# # save_interest_points.run()

# # INTEREST POINT MATCHING
# # --------------------------

# # SOLVE 
# # --------------------------

# # FUSION
# # --------------------------

# # TODO - Implement fetching of fused image data from s3