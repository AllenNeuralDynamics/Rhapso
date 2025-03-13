from Rhapso.matching.interest_point_matching import matchViews, buildLabelMap
import boto3
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

def fetch_from_s3(s3, bucket_name, input_file):
    response = s3.get_object(Bucket=bucket_name, Key=input_file)
    return response['Body'].read().decode('utf-8')

def fetch_from_local(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def fetch_xml_file(file_location):
    if file_location.startswith("s3://"):
        s3 = boto3.client('s3')
        parsed_url = urlparse(file_location)
        bucket_name = parsed_url.netloc
        input_file = parsed_url.path.lstrip('/')
        xml_content = fetch_from_s3(s3, bucket_name, input_file)
    else:
        xml_content = fetch_from_local(file_location)
    return xml_content

print("Starting matching_dev_pipeline:")

# Specify xml input location

# s3 exaspim
xml_location = "s3://aind-open-data/exaSPIM_737563_2024-11-13_12-16-55_alignment_2024-11-27_18-30-20/dataset.xml"

# (breaks) s3 big stitcher spark example dataset tiff/xml
# xml_location = "s3://rhapso-dev/rhapso-sample-data/dataset.xml"

# xml_location = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/IP_TIFF_XML/2. IP_TIFF_XML (after matching)/dataset.xml"

# Fetch XML file content
xml_file_content = fetch_xml_file(xml_location)

# Parse XML content
xml_root = ET.fromstring(xml_file_content)

# Build label map
label_map_global, label_weights = buildLabelMap(xml_root)
print("label_map_global = ", label_map_global)

# Print the resulting label map and label weights
print("Label Map Global:")
for view_id, labels in label_map_global.items():
    print(f"ViewId: {view_id}")
    for label, weight in labels.items():
        print(f"\tLabel: {label}, Weight: {weight}")

print("\nLabel Weights:")
for label, weight in label_weights.items():
    print(f"Label: {label}, Weight: {weight}")

print("Finished matching_dev_pipeline.")