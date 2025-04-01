from Rhapso.detection.interest_point_detection import InterestPointDetection
from Rhapso.pipelines.ipm_dev.pairwise_matching_imports import InterestPointMatching
from Rhapso.solver.solver import Solver

# DETECTION
# ------------------------

# run params
strategy = 'python'
dsxy = 60
dsz = 60
min_intensity = 0
max_intensity = 255
sigma = 1.8
threshold = 0.008

# input/output params
file_type = 'zarr'
file_source = 'local'
xml_file_path = "dataset.xml"
xml_bucket_name = "interest-point-detection"
image_file_path = 's3://aind-open-data/exaSPIM_708365_2024-04-29_12-46-15/SPIM.ome.zarr/'
image_bucket_name = "aind-open-data"
output_file_path = "output"
output_bucket_name = 'interest-point-detection'

# interest_point_detection = InterestPointDetection(strategy, dsxy, dsz, min_intensity, max_intensity, sigma, threshold, 
#                                                   file_type, file_source, xml_file_path, xml_bucket_name, image_file_path, 
#                                                   image_bucket_name, output_file_path, output_bucket_name)
# interest_point_detection.run()

# MATCHING
# -------------------------

xml_input_file = '/Users/seanfite/Desktop/dataset-detection.xml'
n5_base_path = '/Users/seanfite/Desktop/interestpoints.n5'

interest_point_matching = InterestPointMatching(xml_input_file, n5_base_path)
interest_point_matching.run()

# SOLVE
# -------------------------

# input/output params
file_source = file_source
xml_bucket_name = xml_bucket_name
xml_file_path = xml_input_file
data_prefix = n5_base_path 
xml_file_path_output = output_file_path + "/dataset-solver.xml"

# run params
fixed_views = [ 'timepoint: 18, setup: 0', 'timepoint: 30, setup: 0']
model = "affine"
alignment_option = 1
relative_threshold = 3.5
absolute_threshold = 7.0
min_matches = 3
damp = .4
max_iterations= 100000
max_allowed_error= 5.0
max_plateauwidth = 200

solver = Solver(file_source, xml_file_path_output, xml_bucket_name, xml_file_path, data_prefix, fixed_views, model, 
                alignment_option, relative_threshold, absolute_threshold, min_matches, damp, max_iterations, max_allowed_error,
                max_plateauwidth)
solver.run()

# Fusion
# -------------------------
# TODO