from ...data_prep.xml_to_dataframe import XMLToDataFrame
from ...detection.view_transform_models import ViewTransformModels
from ...detection.overlap_detection import OverlapDetection
from ...data_prep.load_image_data import LoadImageData
from ...data_prep.serialize_image_chunks import SerializeImageChunks
from ...data_prep.glue_crawler import GlueCrawler
# from ..interest_point_detection.advanced_refinement import AdvancedRefinement
# from ..interest_point_detection.filtering_and_optimizing import FilteringAndOptimization
# from ..interest_point_detection.save_interest_points import SaveInterestPoints
# from ..data_prep.dataframe_to_xml import DataFrameToXML
import boto3

# Development testing pipeline / deployment template to run the image prep stage locally

strategy = 'spark-etl'
dsxy = 4
dsz = 2

# SPARK ETL PARAMS 
parquet_bucket_path = 's3://interest-point-detection/ipd-staging/'
crawler_name = "InterestPointDetectionCrawler"
crawler_s3_path = "s3://interest-point-detection/ipd-staging/"
crawler_database_name = "interest_point_detection"
crawler_iam_role = "arn:aws:iam::443370675126:role/rhapso-s3"
glue_database = 'interest_point_detection'
glue_table_name = 'ipd_staging'

# FILE TYPE - PICK ONE
file_type = 'tiff'
xml_bucket_name = "rhapso-tif-sample"
xml_file_path = "IP_TIFF_XML/dataset.xml"
# prefix = '/Users/seanfite/Desktop/AllenInstitute/Rhapso/Data/IP_TIFF_XML/'
prefix = 's3://rhapso-tif-sample/IP_TIFF_XML/'

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
print("Image data has loaded")

# Flatten and serialize images to parquet
serialize_image_chunks = SerializeImageChunks(all_image_data, parquet_bucket_path)
serialized_images_dyf = serialize_image_chunks.run()
print("Serialized image data")

# Create and start crawler
glue_crawler = GlueCrawler(crawler_name, crawler_s3_path, crawler_database_name, crawler_iam_role)
glue_crawler.run()
print("Glue crawler created and started")