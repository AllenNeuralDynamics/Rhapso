from Rhapso.matching.xml_parser import XMLParser
from Rhapso.matching.load_and_transform_points import LoadAndTransformPoints
from Rhapso.matching.ransac_matching import RansacMatching
from Rhapso.matching.save_matches import SaveMatches

# s3 input/output paths:
# xml_input_path = 's3://martin-test-bucket/output/dataset-detection.xml'            
# n5_output_path = 's3://martin-test-bucket/matching_output/'

# Local
# xml_input_path = "/Users/seanfite/Desktop/IP_TIFF_XML/dataset.xml"
xml_input_path = "/Users/seanfite/Desktop/ip_rigid_alignment/bigstitcher_rigid.xml"
# xml_input_path = "/Users/seanfite/Desktop/interest_point_detection/bigstitcher_ip.xml"
n5_output_path = '/Users/seanfite/Desktop/IP_TIFF_XML'

# S3 - Big Stitcher Output
# xml_input_path = "s3://aind-open-data/exaSPIM_686951_2025-02-25_09-45-02_alignment_2025-06-12_19-58-52/interest_point_detection/bigstitcher_ip.xml"
# n5_output_path = 's3://rhapso-matching-test/output'

# S3 - Rhapso Output
# xml_input_path = "s3://rhapso-matching-test/output/dataset-detection.xml"
# n5_output_path = 's3://rhapso-matching-test/output'

input_type = "zarr"

match_type = "rigid"               
# -- Finding Candidates --
num_neighbors = 3
redundancy = 0
significance = 3                    # ratio of distance
search_radius = 300
num_required_neighbors = 3
# -- Ransac --
model_min_matches = 3               # min number matches
inlier_factor = 100                 # max epsilon
lambda_value = 0.1                  # min inlier ratio
num_iterations = 10000
regularization_weight = 1.0

# match_type = "affine"                   
# # -- Finding Candidates --
# num_neighbors = 3
# redundancy = 0
# significance = 3                  # ratio of distance
# search_radius = 100
# num_required_neighbors = 3
# # -- Ransac --
# model_min_matches = 24            # min number inlier factor = 5
# inlier_factor = 30                # max epsilon
# lambda_value = 0.1                # min inlier ratio
# num_iterations = 10000
# regularization_weight = 1.0

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