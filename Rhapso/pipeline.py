import boto3
import io
from urllib.parse import urlparse
from data_preparation.xml_to_dataframe import XMLToDataFrame
# from data_preparation.tiff_image_reader import TiffImageReader
from detection.overlap_detection import OverlapDetection
# from detection.interest_point_detection import InterestPointDetection
# from detection.advanced_refinement import AdvancedRefinement
# from detection.filtering_and_optimizing import FilteringAndOptimization
# from detection.save_interest_points import SaveInterestPoints
# from data_preparation.dataframe_to_xml import DataFrameToXML

# First Time Run
# conda create -n rhapso python=3.9
# conda activate rhapso

# conda install pandas boto3 -c conda-forge
# conda install pandas boto3 -c conda-forge
# conda install bottleneck=1.3.6 -c conda-forge
# conda install pillow

# Run
# conda activate rhapso
# python path/to/file.py

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

    # DATA PREPARATION - XML -> Dataframe
    processor = XMLToDataFrame(xml_file)
    dataframes = processor.run()

    # Print DataFrame info
    print_dataframe_info(dataframes)
    
    # print(dataframes)
    # image_reader = TiffImageReader()                              
    # reader = image_reader.run()

    # TRANSFORMS - call custom transforms here 
    overlap_detection = OverlapDetection()
    overlap_detection.run()

    # interest_point_detection = InterestPointDetection()
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


s3_url = "s3://rhapso-dev/rhapso-sample-data/dataset.xml"
local_xml_file = "/mnt/c/Users/marti/Documents/Allen/repos/Rhapso-Sample-Data/IP_TIFF_XML/dataset.xml"

# main(s3_url)  
main(local_xml_file)  
