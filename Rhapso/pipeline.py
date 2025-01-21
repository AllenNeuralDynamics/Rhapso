import boto3
import io
from data_preparation.xml_to_dataframe import XMLToDataFrame
from interest_point_detection.overlap_detection import OverlapDetection
# from interest_point_detection.interest_point_detection import InterestPointDetection
# from interest_point_detection.advanced_refinement import AdvancedRefinement
# from interest_point_detection.filtering_and_optimizing import FilteringAndOptimization
# from interest_point_detection.save_interest_points import SaveInterestPoints
# from data_preparation.dataframe_to_xml import DataFrameToXML

# First Time Run
# conda create -n rhapso python=3.9
# conda activate rhapso

# conda install pandas boto3 -c conda-forge
# conda install -c conda-forge tifffile
# conda install dask

# Run
# conda activate rhapso
# python path/to/file.xml

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

def main(bucket_name, input_file):

    s3 = boto3.client('s3')

    # DATA PREPARATION - XML -> Dataframe
    xml_file = fetch_from_s3(s3, bucket_name, input_file)
    processor = XMLToDataFrame(xml_file)
    dataframes = processor.run()
    # print(dataframes)

    # TRANSFORMS - call custom transforms here 
    overlap_detection = OverlapDetection(dataframes, s3)
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


bucket_name = "rhapso-tiff-sample"
# bucket_name = "rhapso-zarr-sample"
input_file = "IP_TIFF_XML/dataset.xml"
# input_file = "dataset.xml"

main(bucket_name, input_file)
