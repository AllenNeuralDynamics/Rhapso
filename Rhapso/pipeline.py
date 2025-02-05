import boto3
import io
from urllib.parse import urlparse
from data_preparation.xml_to_dataframe import XMLToDataFrame
from data_preparation.tiff_image_reader import TiffImageReader
from detection.view_transform_models import ViewTransformModels
from detection.overlap_detection import OverlapDetection
# from detection.interest_point_detection import InterestPointDetection
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

def main(file_location):

    prefix = 'filepath/to/image/data'
    dsxy = 4                    # amount downsampling x and y
    dsz = 2                     # amount downsampling z
    
    if file_location.startswith("s3://"):
        s3 = boto3.client('s3')
        
        # Parse the S3 URL
        parsed_url = urlparse(file_location)
        bucket_name = parsed_url.netloc
        input_file = parsed_url.path.lstrip('/')
        
        # Fetch XML from S3
        xml_file = fetch_from_s3(s3, bucket_name, input_file)
    else:
        # Fetch XML from local file
        xml_file = fetch_from_local(file_location)

    # INTEREST POINT DETECTION
    # --------------------------

    # Load XML data into dataframes
    processor = XMLToDataFrame(xml_file)
    dataframes = processor.run()

    # Create view_transform_matrices (affine, inverse.affine, degree, inverse.degree)
    create_models = ViewTransformModels(dataframes)
    view_transform_matrices = create_models.run()

    # Use view_transform_matrices to calculate bounding boxes and find areas of overlap
    overlap_detection = OverlapDetection(view_transform_matrices, dataframes, dsxy, dsz, prefix)
    overlapping_area = overlap_detection.run()

    # Load image data
    image_reader = TiffImageReader(dataframes, dsxy, dsz, prefix)                              
    images = image_reader.run()
    # image_reader = ZarrImageReader(dataframes)                              
    # images = image_reader.run()

    # Use image data and areas of overlap to find interest points - custom transform (parallel processing)
    # interest_point_detection = InterestPointDetection(overlapping_area, images)
    # interest_point_detection.run()

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


s3_url = "s3://rhapso-dev/rhapso-sample-data/dataset.xml"
local_xml_file = "/mnt/c/Users/marti/Documents/Allen/repos/Rhapso-Sample-Data/IP_TIFF_XML/dataset.xml"

# bucket_name = "rhapso-tif-sample"
# # bucket_name = "rhapso-zarr-sample"
# file_path = "IP_TIFF_XML/dataset.xml"
# # file_path = "dataset.xml"

# main(s3_url)  
main(local_xml_file)
