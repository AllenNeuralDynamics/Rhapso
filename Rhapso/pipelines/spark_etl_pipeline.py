from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import SparkSession
from Rhapso.detection.interest_point_detection_glue import InterestPointDetectionGlue
import sys

# Rhapso - Spark ETL Pipeline

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext.getOrCreate() 
glueContext = GlueContext(sc)
spark = SparkSession.builder.appName("Interest Point Detection").getOrCreate()
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

strategy = 'python'
dsxy = 4
dsz = 2
min_intensity = 0
max_intensity = 255
sigma = 1.8
threshold = 0.008

# Tiff - S3
file_type = 'tiff'
file_source = 's3'
xml_file_path = "IP_TIFF_XML/dataset.xml"
xml_bucket_name = "rhapso-tif-sample"
image_file_path = 's3://rhapso-tif-sample/IP_TIFF_XML/'
image_bucket_name = "rhapso-tif-sample"
output_file_path = "output"
output_bucket_name = 'interest-point-detection'

# SPARK ETL PARAMS
parquet_bucket_path = 's3://interest-point-detection/ipd_zarr_staging_glue/'
crawler_name = "ipd_zarr_staging_glue"
crawler_s3_path = "s3://interest-point-detection/ipd_zarr_staging_glue/"
crawler_database_name = "ipd_zarr_staging_glue"
crawler_iam_role = "arn:aws:iam::443370675126:role/rhapso-s3"
glue_database = 'ipd_zarr_staging_glue'
glue_table_name = 'ipd_zarr_staging_glue'

interest_point_detection_glue = InterestPointDetectionGlue(strategy, dsxy, dsz, min_intensity, max_intensity, sigma, threshold, file_type, file_source,
            xml_file_path, xml_bucket_name, image_file_path, image_bucket_name, output_file_path, output_bucket_name,
            parquet_bucket_path, crawler_name, crawler_s3_path, crawler_database_name, crawler_iam_role, glue_database,
            glue_table_name, glueContext)
interest_point_detection_glue.run()

job.commit()