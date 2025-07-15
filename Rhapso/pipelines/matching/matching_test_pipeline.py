import ray
from Rhapso.matching.xml_parser import XMLParser
from Rhapso.matching.load_and_transform_points import LoadAndTransformPoints
from Rhapso.matching.ransac_matching import RansacMatching
from Rhapso.matching.save_matches import SaveMatches

# Initialize Ray
ray.init()

# Tiff run time - 90 minutes

# s3 input/output paths:
# xml_input_path = 's3://martin-test-bucket/output/dataset-detection.xml'
# n5_output_path = 's3://martin-test-bucket/matching_output/'

# xml_input_path = '/home/martin/Documents/allen/clean-bss-mvr/IP_TIFF_XML_2/dataset.xml'
# n5__output_path = '/home/martin/Documents/allen/clean-bss-mvr/IP_TIFF_XML_2/output/interestpoints.n5'

xml_input_path = "/mnt/c/Users/marti/Documents/allen/data/IP_TIFF_XML/dataset.xml"
n5_output_path = '/mnt/c/Users/marti/Documents/allen/data/IP_TIFF_XML'

# Params
num_neighbors = 3
redundancy = 1
significance = 3
num_required_neighbors = 4

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
def match_pair(pointsA, pointsB, viewA_str, viewB_str, num_neighbors, redundancy, significance, num_required_neighbors):
    matcher = RansacMatching(
        [(pointsA, pointsB, viewA_str, viewB_str)],
        num_neighbors,
        redundancy,
        significance,
        num_required_neighbors
    )
    candidates = matcher.get_candidates(pointsA, pointsB, viewA_str, viewB_str)
    inliers = matcher.compute_ransac(candidates)

    percent = 100.0 * len(inliers) / len(candidates) if candidates else 0
    print(f"✅ RANSAC inlier percentage: {percent:.1f}% ({len(inliers)} of {len(candidates)})")

    return inliers if inliers else []

# --- Distribute ---
futures = [
    match_pair.remote(pointsA, pointsB, viewA_str, viewB_str, num_neighbors, redundancy, significance, num_required_neighbors)
    for pointsA, pointsB, viewA_str, viewB_str in process_pairs
]

# --- Collect ---
results = ray.get(futures)
all_results = [inlier for sublist in results for inlier in sublist]

print(f"Total matches found: {len(all_results)}")

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