from Rhapso.matching.xml_parser import XMLParser
from Rhapso.matching.load_and_transform_points import LoadAndTransformPoints
from Rhapso.matching.ransac_matching import RansacMatching
from Rhapso.matching.save_matches import SaveMatches

# s3 input/output paths:
# xml_input_path = 's3://martin-test-bucket/output/dataset-detection.xml'            
# n5_output_path = 's3://martin-test-bucket/matching_output/'

# Local
# xml_input_path = "/Users/seanfite/Desktop/IP_TIFF_XML-2/dataset.xml"
# n5_output_path = '/Users/seanfite/Desktop/IP_TIFF_XML'

# S3
xml_input_path = "s3://rhapso-matching-test/output/dataset-detection.xml"
n5_output_path = 's3://rhapso-matching-test/output'

# match_type = "rigid"               
# # -- Finding Candidates --
# num_neighbors = 3
# redundancy = 0
# significance = 3                    # ratio of distance
# search_radius = 300
# num_required_neighbors = 3
# # -- Ransac --
# model_min_matches = 18              # min number inlier factor = 6
# inlier_factor = 5.0                 # max epsilon
# lambda_value = 0.1                  # min inlier ratio
# num_iterations = 10000
# regularization_weight = 1.0

match_type = "affine"                   
# -- Finding Candidates --
num_neighbors = 3
redundancy = 0
significance = 3                  # ratio of distance
search_radius = 100
num_required_neighbors = 3
# -- Ransac --
model_min_matches = 28            # min number inlier factor = 5
inlier_factor = 5.0               # max epsilon
lambda_value = 0.05               # min inlier ratio
num_iterations = 10000
regularization_weight = 1.0

# INTEREST POINT MATCHING
# --------------------------

# Initialize parser with XML content
parser = XMLParser(xml_input_path)
data_global, interest_points_folder = parser.run()
print("XML loaded and parsed")

# Load interest points and transform them into global space
data_loader = LoadAndTransformPoints(data_global, xml_input_path)
process_pairs = data_loader.run()
print("Points loaded and transformed into global space")

# Geometric Descriptor-Based Interest Point Matching with RANSAC
matcher = RansacMatching(
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
    search_radius
)

all_results = []
for pointsA, pointsB, viewA_str, viewB_str in process_pairs:   
    candidates = matcher.get_candidates(pointsA, pointsB, viewA_str, viewB_str)
    inliers = matcher.compute_ransac(candidates)

    percent = 100.0 * len(inliers) / len(candidates)
    print(f"✅ RANSAC inlier percentage: {percent:.1f}% ({len(inliers)} of {len(candidates)} for {viewA_str}), {viewB_str}")

    # if len(inliers) < 20:
    #     continue

    all_results.extend(inliers if inliers else []) 
  
print("Matching is done")

# Save matches as N5
saver = SaveMatches(all_results, n5_output_path, data_global)
saver.run()
print("Matches Saved as N5")

print("Interest point matching is done")