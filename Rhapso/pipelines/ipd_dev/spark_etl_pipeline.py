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
from Rhapso.detection.save_interest_points import SaveInterestPoints
import sys
import boto3
import ast

# Spark ETL detection pipeline - this script runs in AWS Glue

s3 = boto3.client('s3')

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext.getOrCreate() 
glueContext = GlueContext(sc)
spark = SparkSession.builder.appName("Interest Point Detection").getOrCreate()
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# GENERAL PARAMS
strategy = 'spark-etl'
file_source = 's3'
output_bucket_name = 'interest-point-detection'
output_file_path = 'output'
dsxy = 30
dsz = 40
min_intensity = 0
max_intensity = 255
sigma = 1.8
threshold = 0.008 

# SPARK ETL PARAMS
parquet_bucket_path = 's3://interest-point-detection/ipd_zarr_staging_glue/'
crawler_name = "ipd_zarr_staging_glue"
crawler_s3_path = "s3://interest-point-detection/ipd_zarr_staging_glue/"
crawler_database_name = "ipd_zarr_staging_glue"
crawler_iam_role = "arn:aws:iam::443370675126:role/rhapso-s3"
glue_database = 'ipd_zarr_staging_glue'
glue_table_name = 'ipd_zarr_staging_glue'

# FILE TYPE - PICK ONE
# file_type = 'tiff'
# xml_bucket_name = "rhapso-tif-sample"
# xml_file_path = "IP_TIFF_XML/dataset.xml"
# image_bucket_name = "tiff-sample"
# image_file_path = 's3://rhapso-tif-sample/IP_TIFF_XML/'

file_type = 'zarr'
xml_bucket_name = "rhapso-zar-sample"
xml_file_path = "dataset.xml"
image_bucket_name = "aind-open-data"
image_file_path = 's3://aind-open-data/exaSPIM_708365_2024-04-29_12-46-15/SPIM.ome.zarr/'

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
# create_models = ViewTransformModels(dataframes)
# view_transform_matrices = create_models.run()
# print("Transforms Models have been created")

# # Use view transform matrices to find areas of overlap
# overlap_detection = OverlapDetection(view_transform_matrices, dataframes, dsxy, dsz, image_file_path, file_type)
# overlapping_area = overlap_detection.run()
# print("Overlap Detection is done")

# # Load images
# images_loader = LoadImageData(dataframes, overlapping_area, dsxy, dsz, image_file_path, file_type)
# all_image_data = images_loader.run()
# print("Image data has loaded")

# # Flatten and serialize images to parquet
# serialize_image_chunks = SerializeImageChunks(all_image_data, parquet_bucket_path)
# serialized_images_dyf = serialize_image_chunks.run()
# print("Serialized image data")

# # Create and start crawler
# glue_crawler = GlueCrawler(crawler_name, crawler_s3_path, crawler_database_name, crawler_iam_role)
# glue_crawler.run()
# print("Glue crawler created and started")

# Create dynamic frame using crawler schema
image_data_dyf = glueContext.create_dynamic_frame.from_catalog(
    database = glue_database,
    table_name = glue_table_name,
    transformation_ctx = "dynamic_frame"
)
print("Dynamic frame loaded")

# Detect interest points using DoG algorithm - custom transform
difference_of_gaussian = DifferenceOfGaussian(min_intensity, max_intensity, sigma, threshold)
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
print("Difference of gaussian is done")

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
consolidated_data = advanced_refinement.run()
print("Advanced refinement is complete.")

# Save interest points to N5 and update xml file
save_interest_points = SaveInterestPoints(dataframes, consolidated_data, xml_file_path, xml_bucket_name, 
                                          output_bucket_name, output_file_path, dsxy, dsz, min_intensity, 
                                          max_intensity, sigma, threshold, file_source)
save_interest_points.run()

print("Interest point detection is done")

job.commit()

# AWS GLUE DOCKER CONTAINER SETUP - LOCAL TESTING
# --------------------------------------------------

# BUILD PYTHON WHL FILE
# python setup.py bdist_wheel

# CREATE NEW AWS GLUE DOCKER CONTAINER (name -> spark-etl)
# docker run -d --name spark-etl --user root --memory 16g --memory-swap 16g -v /Users/seanfite/Desktop/Allen-Rhapso:/app/logs -it amazon/aws-glue-libs:glue_libs_4.0.0_image_01-arm64

# START CONTAINER
# docker start spark-etl

# CONFIGURE AWS AUTH IN CONTAINER
# docker exec -it spark-etl aws configure

# CREATE DIRECTORY IN CONTAINER
# docker exec spark-etl mkdir -p /app/spark_etl

# COPY WHL FILE TO CONTAINER
# docker cp /Users/seanfite/Desktop/Allen-Rhapso/dist/Rhapso-0.1.0-py3-none-any.whl spark-etl:/

# STEP INTO CONTAINER
# docker exec -it spark-etl bash

    # INSTALL PIP IN CONTAINER
    # yum install -y python3-pip

    # PIP INSTALL WHL IN CONTAINER
    # pip3 install /Rhapso-0.1.0-py3-none-any.whl

# COPY RUN SCRIPT IN DOCKER CONTAINER
# docker cp /Users/seanfite/Desktop/Allen-Rhapso/Rhapso/pipelines/ipd_dev/spark_etl_pipeline.py spark-etl:/app/spark_etl/rhapso_pipeline.py

# EXECUTE RUN SCRIPT IN DOCKER CONTAINER 
# docker exec -it spark-etl python3 /app/spark_etl/rhapso_pipeline.py --JOB_NAME new_job

# DOCKER ADMIN STUFF
# -----------------------

# ACCESS DOCKER CONTAINER
# docker exec -it spark-etl bash

# STOP ALL DOCKER CONTAINERS
# docker stop $(docker ps -aq)

# REMOVE ALL DOCKER CONTAINERS
# docker rm $(docker ps -aq)

# DYNAMIC FRAME LOADER DEV VERSION ---
# The docker container can only handle a few image chunks at a time. If doing local dev work in the docker 
# container, use this version to create the dynamic frame and set limit to a max 3 image chunks.
# specific_files = [
#     "s3://interest-point-detection/ipd-staging-v2/1.parquet/partition_key=12/9aa52395caba4e6582089464b2287c60-0.parquet"
#     "s3://interest-point-detection/ipd-staging-v2/1.parquet/partition_key=11/9aa52395caba4e6582089464b2287c60-0.parquet",
#     "s3://interest-point-detection/ipd-staging-v2/1.parquet/partition_key=15/9aa52395caba4e6582089464b2287c60-0.parquet"
# ]
# image_data_dyf = glueContext.create_dynamic_frame.from_options(
#     connection_type="s3",  
#     format="parquet",     
#     connection_options={"paths": specific_files}, 
#     transformation_ctx="dynamic_frame"
# )
# spark_df = image_data_dyf.toDF()
# limited_spark_df = spark_df.limit(2)
# limited_dyf = DynamicFrame.fromDF(limited_spark_df, glueContext, "limited_dyf")
# END DYNAMIC FRAME LOADER DEV VERSION ---