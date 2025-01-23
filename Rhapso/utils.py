import boto3
from urllib.parse import urlparse

def fetch_from_s3(s3, bucket_name, input_file):
    response = s3.get_object(Bucket=bucket_name, Key=input_file)
    print(f"Fetched from S3: {input_file} in bucket: {bucket_name}")
    return response['Body'].read().decode('utf-8')

def fetch_from_local(file_path):
    with open(file_path, 'r') as file:
        print(f"Fetched from local file: {file_path}")
        return file.read()

def xmlToDataframe(file_location, XMLToDataFrame):
    if file_location.startswith("s3://"):
        s3 = boto3.client('s3')
        parsed_url = urlparse(file_location)
        bucket_name = parsed_url.netloc
        input_file = parsed_url.path.lstrip('/')
        xml_file = fetch_from_s3(s3, bucket_name, input_file)
    else:
        xml_file = fetch_from_local(file_location)
    
    processor = XMLToDataFrame(xml_file)
    dataframes = processor.run()
    return dataframes
