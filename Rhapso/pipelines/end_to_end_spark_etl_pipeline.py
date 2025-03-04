import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import SparkSession
from Rhapso.data_preparation.xml_to_dataframe import XMLToDataFrame
from Rhapso.detection.view_transform_models import ViewTransformModels
from Rhapso.detection.overlap_detection import OverlapDetection
from Rhapso.data_preparation.load_image_data import LoadImageData
from Rhapso.detection.difference_of_gaussian import DifferenceOfGaussian
from Rhapso.data_preparation.serialize_image_chunks import SerializeImageChunks
from Rhapso.data_preparation.deserialize_image_chunks import DeserializeImageChunks
from Rhapso.data_preparation.glue_crawler import GlueCrawler
import boto3
import base64
import numpy as np
import io

# Spark ETL testing pipeline - this script runs directly in AWS Glue

s3 = boto3.client('s3')

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext.getOrCreate() 
glueContext = GlueContext(sc)
spark = SparkSession.builder.appName("Interest Point Detection").getOrCreate()
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# BUCKET NAME
xml_bucket_name = "rhapso-tif-sample"
image_bucket_name = "aind-open-data"
# xml_bucket_name = "rhapso-zar-sample"

# FILE KEY
xml_file_path = "IP_TIFF_XML/dataset.xml"
# xml_file_path = "dataset.xml"

# BUCKET PREFIX
prefix = 's3://rhapso-tif-sample/IP_TIFF_XML/'
# prefix = 's3://aind-open-data/exaSPIM_708365_2024-04-29_12-46-15/SPIM.ome.zarr'

# DOWNSAMPLING
dsxy = 4
dsz = 2

# FILE TYPE
file_type = 'tiff'
# file_type_zarr = 'zarr'

# STRATEGY
strategy = 'spark-etl'

# PARQUET
parquet_bucket_path = 's3://interest-point-detection/ipd-staging/'

# CRAWLER
crawler_name = "InterestPointDetectionCrawler"
crawler_s3_path = "s3://interest-point-detection/ipd-staging/"
crawler_database_name = "interest_point_detection"
crawler_iam_role = "arn:aws:iam::443370675126:role/rhapso-s3"

# GLUE JOB
glue_database = 'interest_point_detection'
glue_table_name = 'idp_ipd_staging'

def fetch_from_s3(s3, bucket_name, input_file):
    response = s3.get_object(Bucket=bucket_name, Key=input_file)
    return response['Body'].read().decode('utf-8')

# INTEREST POINT DETECTION
# --------------------------

# xml_file = fetch_from_s3(s3, xml_bucket_name, xml_file_path) 

# # Load XML data into dataframes
# processor = XMLToDataFrame(xml_file)
# dataframes = processor.run()
# print("XML loaded")

# # Create view transform matrices 
# create_models = ViewTransformModels(dataframes)
# view_transform_matrices = create_models.run()
# print("Transforms Models have been created")

# # Use view transform matrices to find areas of overlap
# overlap_detection = OverlapDetection(view_transform_matrices, dataframes, dsxy, dsz, prefix, file_type)
# overlapping_area = overlap_detection.run()
# print("Overlap Detection is done")

# # Load images
# images_loader = LoadImageData(dataframes, overlapping_area, dsxy, dsz, prefix, file_type)
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
difference_of_gaussian = DifferenceOfGaussian()
deserialize_image_chunks = DeserializeImageChunks()
def interest_point_detection(record):
    image_chunk = deserialize_image_chunks.run(record)
    interest_points = difference_of_gaussian.run(image_chunk)
    interest_points_as_strings = [str(point) for point in interest_points]
    return {'interest_points': interest_points_as_strings}
mapped_results_dyf = image_data_dyf.map(interest_point_detection, transformation_ctx="map_interest_points")
print("Interest point detection done")

# View results
result_df = mapped_results_dyf.toDF()
result_df.show()

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

job.commit()

# AWS GLUE DOCKER CONTAINER SETUP - LOCAL TESTING
# --------------------------------------------------

# BUILD PYTHON WHL FILE
# python setup.py bdist_wheel

# CREATE NEW AWS GLUE DOCKER CONTAINER (name -> spark-etl)
# docker run -d --name spark-etl --user root -v /Users/seanfite/Desktop/Rhapso:/app/logs -it amazon/aws-glue-libs:glue_libs_4.0.0_image_01

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
# docker cp /Users/seanfite/Desktop/Rhapso/Rhapso/spark_etl_pipeline.py spark-etl:/app/spark_etl/rhapso_pipeline.py

# EXECUTE RUN SCRIPT IN DOCKER CONTAINER 
# docker exec -it spark-etl python3 /app/spark_etl/rhapso_pipeline.py --JOB_NAME new_job

# DOCKER ADMIN STUFF
# -----------------

# ACCESS DOCKER CONTAINER
# docker exec -it spark-etl bash

# STOP ALL DOCKER CONTAINERS
# docker stop $(docker ps -aq)

# REMOVE ALL DOCKER CONTAINERS
# docker rm $(docker ps -aq)