import boto3
import io
from urllib.parse import urlparse
from data_preparation.xml_to_dataframe import XMLToDataFrame
from detection.view_transform_models import ViewTransformModels
from detection.overlap_detection import OverlapDetection
from detection.python.interest_point_detection import PythonInterestPointDetection
from detection.spark_etl.spark_pipeline import SparkPipeline
from detection.aws_glue.glue_pipeline import GluePipeline
# from detection.advanced_refinement import AdvancedRefinement
# from detection.filtering_and_optimizing import FilteringAndOptimization
# from detection.save_interest_points import SaveInterestPoints
# from data_preparation.dataframe_to_xml import DataFrameToXML

# Development testing pipeline / deployment template

def fetch_from_s3(s3, bucket_name, input_file):
    response = s3.get_object(Bucket=bucket_name, Key=input_file)
    print(f"Fetched from S3: {input_file} in bucket: {bucket_name}")
    return response['Body'].read().decode('utf-8')

def send_to_s3(s3, data, bucket_name, output_file):
    buffer = io.BytesIO()
    buffer.write(data.encode('utf-8'))
    buffer.seek(0)
    s3.put_object(Bucket=bucket_name, Body=buffer, Key=output_file)
    print(f"Sent to S3: {output_file} in bucket: {bucket_name}")

def fetch_from_local(file_path):
    with open(file_path, 'r') as file:
        print(f"Fetched from local file: {file_path}")
        return file.read()

def print_dataframe_info(dataframes):
    for name, df in dataframes.items():
        print(f"DataFrame: {name}")
        print(df.info())
        print(df.head())

def main(xml_key, image_data_prefix, dsxy, dsz, **kwargs):
    
    """Fetch XML file from specified storage."""
    if image_data_prefix.startswith("s3://"):                  
        # from s3
        s3 = boto3.client('s3')
        
        parsed_url = urlparse(image_data_prefix)
        bucket_name = parsed_url.netloc
        prefix = parsed_url.path.lstrip('/')
        
        if not prefix.endswith('/'):
            prefix += '/'
        
        full_xml_key = f"{prefix}{s3_xml_key_tiff}"
        xml_file = fetch_from_s3(s3, bucket_name, full_xml_key)
    else:
        # from local file
        image_data_key = image_data_prefix + xml_key
        xml_file = fetch_from_local(image_data_key)

    
    # INTEREST POINT DETECTION
    # --------------------------

    strategy_type = kwargs.pop('strategy_type', None)

    # Load XML data into dataframes
    if strategy_type == 'python':          
        processor = XMLToDataFrame(xml_file)
        dataframes = processor.run()
    elif strategy_type == 'aws_glue':
        print("hi")
    elif strategy_type == 'spark-etl':
        # TODO - update to utilize aws glue crawler and classifier        
        processor = XMLToDataFrame(xml_file)
        dataframes = processor.run()
    print("XML to dataframes done")

    # Create view_transform_matrices (affine, inverse.affine, degree, inverse.degree)
    create_models = ViewTransformModels(dataframes)
    view_transform_matrices = create_models.run()
    print("Created view transform models")

    # Use view_transform_matrices to calculate bounding boxes and find areas of overlap
    overlap_detection = OverlapDetection(view_transform_matrices, dataframes, dsxy, dsz, image_data_prefix)
    overlapping_area = overlap_detection.run()
    print("Found areas of overlap")

    # Use image data and areas of overlap to find interest points - (custom transform) 
    if strategy_type == "python":
        strategy = PythonInterestPointDetection(dataframes, overlapping_area, dsxy, dsz, image_data_prefix, **kwargs)
        strategy.run()
    elif strategy_type == "aws-glue":
        strategy = GluePipeline(dataframes, overlapping_area, dsxy, dsz, image_data_prefix, **kwargs)
        # TODO - implement
    elif strategy_type == "spark-etl":
        strategy = SparkPipeline(dataframes, overlapping_area, dsxy, dsz, image_data_prefix, **kwargs)
        strategy.run()

    # filtering_and_optimizing = FilteringAndOptimizing()
    # filtering_and_optimizing.run()

    # advanced_refinement = AdvancedRefinement()
    # advanced_refinement.run()

    # save_interest_points = SaveInterestPoints()
    # save_interest_points.run()

    # DATA PREPARATION - Dataframe -> XML 
    # processor = DataFrameToXML(dataframes)
    # processor.run()

    # INTEREST POINT MATCHING
    # --------------------------

    # SOLVE (TRANSFORMATIONS)
    # --------------------------
    
    # FUSION
    # --------------------------


# BASE PARAMS
# ------------------------------

# s3 storage - tiff
s3_xml_key_tiff = 'dataset.xml'
s3_image_data_prefix_tiff = 's3://rhapso-tif-sample/IP_TIFF_XML/'

# s3 storage - zarr
s3_xml_key_zarr = 'dataset.xml'
s3_image_data_prefix_zarr = 's3://rhapso-zar-sample/'

# local storage - tiff
local_xml_key_tiff = 'dataset.xml'
# local_image_data_prefix_tiff = '/mnt/c/Users/marti/Documents/Allen/repos/Rhapso-Sample-Data/IP_TIFF_XML/'
local_image_data_prefix_tiff = '/Users/seanfite/Desktop/AllenInstitute/Rhapso/Data/IP_TIFF_XML/'

# local storage - zarr
local_xml_key_zarr = 'dataset.xml'
local_image_data_prefix_zarr = '/Users/seanfite/Desktop/AllenInstitute/Rhapso/Data/Zarr/'

# downsampling
dsxy = 4
dsz = 4

# DYNAMIC PARAMS
# -------------------------------

# Python
python_params = {
    "strategy_type": "python", 
}

# Spark ETL 
spark_params = {
    "rhapso_prefix": '/Users/seanfite/Desktop/Rhapso',
    'bucket_name': 'interest-point-detection',
    "strategy_type": "spark-etl", 
    "region": 'us-west-2',
    "job_name": 'mouse-134548584',
    "role": 'arn:aws:iam::443370675126:role/rhapso-s3',
    "worker_type": 'G.1X',
    "number_of_workers": 10,
    "glue_version": '2.0',
    "job_timeout": 480,
    "retries": 3,
    "job_bookmark_option": 'disable',
    "flex_execution": False
}

# TODO
# glue_params = {
# }

main(local_xml_key_tiff, local_image_data_prefix_tiff, dsxy, dsz, **python_params)
