import json
import os
import s3fs
from Rhapso.matching.xml_parser import XMLParser
from Rhapso.matching.load_and_transform_points import LoadAndTransformPoints
from Rhapso.matching.ransac_matching import RansacMatching
from Rhapso.matching.save_matches import SaveMatches

# s3 input/output paths:
# xml_input_path = 's3://martin-test-bucket/output/dataset-detection.xml'            
# n5_output_path = 's3://martin-test-bucket/matching_output/'

# Local
# xml_input_path = "/Users/ai/Desktop/IP_TIFF_XML_rhapso_8_12/dataset-detection.xml"
# xml_input_path = "/Users/seanfite/Desktop/ip_rigid_alignment/bigstitcher_rigid.xml"
# xml_input_path = "/Users/seanfite/Desktop/interest_point_detection/bigstitcher_ip.xml"
# n5_output_path = '/Users/ai/Desktop/IP_TIFF_XML_rhapso_8_12/'

# # BS_S3 via local save down
s3_uri="s3://aind-open-data/exaSPIM_730223_2025-04-25_14-44-58_alignment_2025-05-13_23-58-19/interest_point_detection/"
local_dir="/Users/ai/Desktop/s3test/6/"
xml_input_path = ""
n5_output_path = "" ##These paths get taken after s3 is downloaded. Can be input manually as well.

# xml_input_path = "/Users/ai/Desktop/s3test/TEST4/aind-open-data/exaSPIM_730223_2025-04-25_14-44-58_alignment_2025-05-13_23-58-19/interest_point_detection/bigstitcher_ip.xml"
# n5_output_path = "/Users/ai/Desktop/s3test/TEST4/aind-open-data/exaSPIM_730223_2025-04-25_14-44-58_alignment_2025-05-13_23-58-19/interest_point_detection/interestpoints.n5"

# S3 - Big Stitcher Output
# xml_input_path = "s3://aind-open-data/exaSPIM_686951_2025-02-25_09-45-02_alignment_2025-06-12_19-58-52/interest_point_detection/bigstitcher_ip.xml"
# n5_output_path = 's3://rhapso-matching-test/output'

# S3 - Rhapso Output
# xml_input_path = "s3://rhapso-matching-test/output/dataset-detection.xml"
# n5_output_path = 's3://rhapso-matching-test/output'

input_type = "zarr"

# match_type = "rigid"               
# -- Finding Candidates --
# num_neighbors = 3
# redundancy = 0
# significance = 3                    # ratio of distance
# search_radius = 300
# num_required_neighbors = 3
# # -- Ransac --
# model_min_matches = 3               # min number matches
# inlier_factor = 100                 # max epsilon
# lambda_value = 0.1                  # min inlier ratio
# num_iterations = 10000
# regularization_weight = 1.0

# match_type = "affine"                   
# # -- Finding Candidates --
# num_neighbors = 3
# redundancy = 0
# significance = 3                    # ratio of distance
# search_radius = 300
# num_required_neighbors = 3
# # -- Ransac --
# model_min_matches = 3               # min number matches
# inlier_factor = 100                 # max epsilon
# lambda_value = 0.1                  # min inlier ratio
# num_iterations = 10000
# regularization_weight = 1.0

match_type = "affine"                   
# # -- Finding Candidates --
num_neighbors = 3
redundancy = 0
significance = 3                  # ratio of distance
search_radius = 100
num_required_neighbors = 3
# -- Ransac --
model_min_matches = 5           # min number inlier factor = 5
inlier_factor = 30                # max epsilon
lambda_value = 0.1                # min inlier ratio
num_iterations = 10000
regularization_weight = 1.0

# BIGSTITCHER S3 TO LOCAL
# --------------------------
def download_n5_from_s3_to_local( s3_uri, local_dir):
        """
        Recursively download an N5 dataset from S3 to a local directory.
        """
        s3 = s3fs.S3FileSystem(anon=False)
        s3_path = s3_uri.replace("s3://", "")
        all_keys = s3.find(s3_path, detail=True)
 
        for key, obj in all_keys.items():
            if obj["type"] == "file":
                rel_path = key.replace(s3_path + "/", "")
                local_file_path = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3.get(key, local_file_path)

                # Check for the specific interestpoints path
                if rel_path.endswith("beads/interestpoints/attributes.json") and "interestpoints.n5" in rel_path:
                    # Construct the path to the attributes file
                    attributes_path = os.path.join(os.path.dirname(local_file_path), "attributes.json")
                    attributes_data = {
                        "pointcloud": "1.0.0",
                        "type": "list",
                        "list version": "1.0.0"
                    }

                    with open(attributes_path, "w") as f:
                                        json.dump(attributes_data, f, indent=2)


# # BIGSTITCHER S3 TO LOCAL- Collect local paths
# # --------------------------
download_n5_from_s3_to_local(s3_uri, local_dir)

s3_path = s3_uri.replace("s3://", "")
full_local_path = os.path.join(local_dir, s3_path)

# Final paths
xml_input_path = os.path.join(full_local_path, "bigstitcher_ip.xml")
n5_output_path = os.path.join(full_local_path, "interestpoints.n5")

print("XML Input Path:", xml_input_path)
print("N5 Output Path:", n5_output_path)

# INTEREST POINT MATCHING
# --------------------------

# Initialize parser with XML content
parser = XMLParser(xml_input_path, input_type)
data_global, interest_points_folder = parser.run()
print("XML loaded and parsed")

# Load interest points and transform them into global space
data_loader = LoadAndTransformPoints(data_global, xml_input_path)
process_pairs, view_registrations = data_loader.run()
print("Points loaded and transformed into global space")

# Geometric Descriptor-Based Interest Point Matching with RANSAC
matcher = RansacMatching(
    data_global,
    num_neighbors, 
    redundancy, 
    significance, 
    num_required_neighbors, 
    match_type, 
    inlier_factor, 
    lambda_value, 
    num_iterations, 
    model_min_matches, 
    regularization_weight,
    search_radius,
    view_registrations,
    input_type
)

all_results = []
for pointsA, pointsB, viewA_str, viewB_str in process_pairs:  

    pointsA, pointsB = matcher.filter_for_overlapping_points(pointsA, pointsB, viewA_str, viewB_str)
    
    if len(pointsA) == 0 or len(pointsB) == 0: 
        continue

    candidates = matcher.get_candidates(pointsA, pointsB, viewA_str, viewB_str)
    inliers, regularized_model = matcher.compute_ransac(candidates)
    filtered_inliers = matcher.filter_inliers(inliers, regularized_model)

    percent = 100.0 * len(filtered_inliers) / len(candidates) if len(candidates) > 0 else 0.0
    print(f"✅ RANSAC inlier percentagE: {percent:.1f}% ({len(filtered_inliers)} of {len(candidates)} for {viewA_str}), {viewB_str}")
    
    if len(filtered_inliers) < 20: 
        continue
    
    all_results.extend(filtered_inliers if filtered_inliers else []) 
  
print("Matching is done")

# Save matches as N5
saver = SaveMatches(all_results, n5_output_path, data_global)
saver.run()
print("Matches Saved as N5")

print("Interest point matching is done")