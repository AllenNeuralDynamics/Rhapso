import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession
from Rhapso.data_preparation.xml_to_dataframe import XMLToDataFrame
from Rhapso.detection.view_transform_models import ViewTransformModels
from Rhapso.detection.overlap_detection import OverlapDetection
from Rhapso.detection.interest_point_detection import PythonInterestPointDetection
import boto3

s3 = boto3.client('s3')

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext.getOrCreate() 
glueContext = GlueContext(sc)
# spark = SparkSession.builder.master("local[*]").appName("MyApp").getOrCreate()
spark = SparkSession.builder.appName("Interest Point Detection").getOrCreate()
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# BUCKET NAME
xml_bucket_name = "rhapso-tif-sample"
# xml_bucket_name = "rhapso-zar-sample"
image_bucket_name = "aind-open-data"

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

# INTEREST POINT MATCHING
# --------------------------

# SOLVE 
# --------------------------

# FUSION
# --------------------------

job.commit()

# AWS GLUE DOCKER CONTAINER SETUP

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
# docker cp /Users/seanfite/Desktop/Allen-Rhapso/Rhapso/spark_etl_pipeline.py spark-etl:/app/spark_etl/rhapso_pipeline.py

# EXECUTE RUN SCRIPT IN DOCKER CONTAINER 
# docker exec -it spark-etl python3 /app/spark_etl/rhapso_pipeline.py --JOB_NAME new_job

# ADMIN STUFF

# ACCESS DOCKER CONTAINER
# docker exec -it spark-etl bash

# STOP ALL DOCKER CONTAINERS
# docker stop $(docker ps -aq)

# REMOVE ALL DOCKER CONTAINERS
# docker rm $(docker ps -aq)