import ray
from Rhapso.matching.xml_parser import XMLParser
from Rhapso.matching.load_and_transform_points import LoadAndTransformPoints
from Rhapso.matching.ransac_matching import RansacMatching
from Rhapso.matching.save_matches import SaveMatches

# Initialize Ray
ray.init()

# s3 input/output paths:
# xml_input_path = 's3://martin-test-bucket/output/dataset-detection.xml'
# n5_output_path = 's3://martin-test-bucket/matching_output/'

# xml_input_path = "/mnt/c/Users/marti/Documents/allen/data/IP_TIFF_XML/dataset.xml"
# n5_output_path = '/mnt/c/Users/marti/Documents/allen/data/IP_TIFF_XML'

xml_input_path = "/Users/seanfite/Desktop/interest_point_detection/bigstitcher_ip.xml"
n5_output_path = '/Users/seanfite/Desktop/interest_point_detection/output'

# Params
match_type = "rigid"               
# -- Finding Candidates --
num_neighbors = 3
redundancy = 0
significance = 3                    # ratio of distance
search_radius = 300
num_required_neighbors = 3
# -- Ransac --
model_min_matches = 18              # min number inlier factor = 6
inlier_factor = 5.0                 # max epsilon
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
# model_min_matches = 28            # min number inlier factor = 5
# inlier_factor = 5.0               # max epsilon
# lambda_value = 0.05               # min inlier ratio
# num_iterations = 10000
# regularization_weight = 1.0

# Load XML
parser = XMLParser(xml_input_path)
data_global, interest_points_folder = parser.run()
print("XML loaded and parsed")

# Load and transform points
data_loader = LoadAndTransformPoints(data_global, xml_input_path)
process_pairs = data_loader.run()
print("Points loaded and transformed into global space")

# --- Ray Remote Task ---
@ray.remote(memory=4 * 1024 * 1024 * 1024)  # 4GB
def match_pair(
    pointsA, pointsB, viewA_str, viewB_str, num_neighbors, redundancy, significance, num_required_neighbors,
    match_type, inlier_factor, lambda_value, num_iterations, model_min_matches, regularization_weight, search_radius
): 
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
    
    candidates = matcher.get_candidates(pointsA, pointsB, viewA_str, viewB_str)
    inliers = matcher.compute_ransac(candidates)

    percent = 100.0 * len(inliers) / len(candidates) if candidates else 0
    print(f"✅ RANSAC inlier percentage: {percent:.1f}% ({len(inliers)} of {len(candidates)})")

    if len(inliers) < 20:
        inliers = []

    return inliers if inliers else []

# --- Distribute ---
futures = [
    match_pair.remote(pointsA, pointsB, viewA_str, viewB_str, num_neighbors, redundancy, significance, num_required_neighbors,
                      match_type, inlier_factor, model_min_matches, num_iterations, model_min_matches, regularization_weight, 
                      search_radius)
    for pointsA, pointsB, viewA_str, viewB_str in process_pairs
]

# --- Collect ---
results = ray.get(futures)
all_results = [inlier for sublist in results for inlier in sublist]

# --- Save ---
saver = SaveMatches(all_results, n5_output_path, data_global)
saver.run()
print("Matches Saved as N5")

print("Interest point matching is done")

# -----------------------------------------------------------------------------
# How to run this pipeline script:
# 
#   1. Create virtual env: `python -m venv myNewEnv`
#   2. Activate env: `source myNewEnv/bin/activate`
#   3. Install dependencies from repo root: `pip install .[matching]` and Rhapso `pip install -e .` 
#   4. Change input vars xml_input_path and n5_output_path 
#   5. Run script: `python Rhapso/pipelines/matching/matching_test_pipeline.py`
# -----------------------------------------------------------------------------