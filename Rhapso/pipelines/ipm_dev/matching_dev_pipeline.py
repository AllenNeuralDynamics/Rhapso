from Rhapso.matching.interest_point_matching import fetch_xml_file, buildLabelMap
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

print("Starting matching_dev_pipeline:")

# Specify xml input location

# s3 exaspim xml
xml_location = "s3://aind-open-data/exaSPIM_737563_2024-11-13_12-16-55_alignment_2024-11-27_18-30-20/dataset.xml"

# s3 big stitcher spark example dataset tiff/xml
# xml_location = "s3://rhapso-dev/rhapso-sample-data/dataset.xml"

# local xml
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