import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import SparkSession
from Rhapso.data_prep.xml_to_dataframe import XMLToDataFrame
from Rhapso.detection.view_transform_models import ViewTransformModels
from Rhapso.detection.overlap_detection import OverlapDetection
from Rhapso.data_prep.load_image_data import LoadImageData
from Rhapso.data_prep.glue_crawler import GlueCrawler
from Rhapso.data_prep.serialize_image_chunks import SerializeImageChunks
from Rhapso.data_prep.deserialize_image_chunks import DeserializeImageChunks
from Rhapso.detection.difference_of_gaussian import DifferenceOfGaussian
from Rhapso.detection.advanced_refinement import AdvancedRefinement
import boto3
import ast

# Spark ETL testing pipeline - this script runs in AWS Glue or in a AWS Glue Docker Container (dev only)

s3 = boto3.client('s3')

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext.getOrCreate() 
glueContext = GlueContext(sc)
spark = SparkSession.builder.appName("Interest Point Detection").getOrCreate()
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# DOWNSAMPLING
dsxy = 4
dsz = 4
strategy = 'spark-etl'

# SPARK ETL PARAMS
parquet_bucket_path = 's3://interest-point-detection/ipd-staging-v2/'
crawler_name = "NewestIPD"
crawler_s3_path = "s3://interest-point-detection/ipd-staging-v2/"
crawler_database_name = "NewestIPD"
crawler_iam_role = "arn:aws:iam::443370675126:role/rhapso-s3"

glue_database = 'newestipd'
glue_table_name = 'ipd_staging_v2'

# FILE TYPE - PICK ONE
file_type = 'tiff'
xml_bucket_name = "rhapso-tif-sample"
image_bucket_name = "tiff-sample"
prefix = 's3://rhapso-tif-sample/IP_TIFF_XML/'
xml_file_path = "IP_TIFF_XML/dataset.xml"

# file_type_zarr = 'zarr'
# xml_bucket_name = "rhapso-zar-sample"
# image_bucket_name = "aind-open-data"
# prefix = 's3://aind-open-data/exaSPIM_708365_2024-04-29_12-46-15/SPIM.ome.zarr'
# xml_file_path = "dataset.xml"

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

# Create dynamic frame using crawler schema
image_data_dyf = glueContext.create_dynamic_frame.from_catalog(
    database = glue_database,
    table_name = glue_table_name,
    transformation_ctx = "dynamic_frame"
)
print("Dynamic frame loaded")

# Detect interest points using DoG algorithm - custom transform
difference_of_gaussian = DifferenceOfGaussian()
deserialize_image_chunks = DeserializeImageChunks()
def interest_point_detection(record):
    try:
        view_id, interval, image_chunk = deserialize_image_chunks.run(record)     
        dog_results = difference_of_gaussian.run(image_chunk, dsxy, dsz)
        interest_points = dog_results['interest_points']
        intensities = dog_results['intensities']
        interest_points_as_strings = [str(point) for point in interest_points]
        results_dict = {
            'view_id': str(view_id),
            'interval_key': str(interval),
            'interest_points': interest_points_as_strings,
            'intensities': intensities.tolist() 
        }
        return results_dict
    except Exception as e:
        print("Error processing record:", str(e))
        return {}
mapped_results_dyf = image_data_dyf.map(interest_point_detection, transformation_ctx="map_interest_points")
print("Interest point detection done")

# Format results out of dynamic frame for advanced refinement
result_df = mapped_results_dyf.toDF()
interest_points_list = []
for row in result_df.collect():
    view_id = row['view_id']
    interval_key = row['interval_key']
    interest_points = [ast.literal_eval(point) for point in row['interest_points']]
    intensities = row['intensities']
    interest_points_list.append({
        'view_id': view_id,
        'interval_key': interval_key,
        'interest_points': interest_points,
        'intensities': intensities
    })
print("Results formatted and ready for advanced refinement")

# Use kdtree algorithm to filter out duplicated points of interest
advanced_refinement = AdvancedRefinement(interest_points_list)
final_interest_points = advanced_refinement.run()
print("Advanced refinement is complete.")

print("Final interest points:", final_interest_points)

# save_interest_points = SaveInterestPoints()
# save_interest_points.run()

# INTEREST POINT MATCHING
# --------------------------

# SOLVE 
# --------------------------

# FUSION
# --------------------------

job.commit()