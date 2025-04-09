from Rhapso.detection.interest_point_detection import InterestPointDetection
from Rhapso.matching.interest_point_matching import start_matching
from Rhapso.solver.solver import Solver

# DETECTION
# ------------------------

# run params
strategy = "python"
dsxy = 4
dsz = 2
min_intensity = 0
max_intensity = 255
sigma = 1.8
threshold = 0.008

# input/output params
# file_type = 'zarr'
# file_source = 's3'
# xml_file_path = "dataset.xml"
# xml_bucket_name = "rhapso-zar-sample"
# image_file_path = 's3://aind-open-data/exaSPIM_708365_2024-04-29_12-46-15/SPIM.ome.zarr/'
# image_bucket_name = "aind-open-data"
# output_file_path = "output"
# output_bucket_name = 'rhapso-zarr-glue'

file_type = "tiff"
file_source = "local"
xml_file_path = "/Users/seanfite/Desktop/IP_TIFF_XML/dataset.xml"
image_file_path = "/Users/seanfite/Desktop/IP_TIFF_XML/"
output_file_path = "/Users/seanfite/Desktop/IP_TIFF_XML/output"
xml_bucket_name = None
image_bucket_name = None
output_bucket_name = None

interest_point_detection = InterestPointDetection(
    strategy,
    dsxy,
    dsz,
    min_intensity,
    max_intensity,
    sigma,
    threshold,
    file_type,
    file_source,
    xml_file_path,
    xml_bucket_name,
    image_file_path,
    image_bucket_name,
    output_file_path,
    output_bucket_name,
)
interest_point_detection.run()

# MATCHING
# -------------------------

# xml_input_file = "s3://rhapso-zarr-output-staged-for-fusion/output/dataset-detection.xml"
# n5_base_path = 's3://rhapso-zarr-output-staged-for-fusion/output/interestpoints.n5'
# xml_input_file = "s3://rhapso-zarr-glue/output/dataset-detection.xml"
# n5_base_path = 's3://rhapso-zarr-glue/output/interestpoints.n5'
# output_path = n5_base_path

xml_input_file = output_file_path + "/dataset-detection.xml"
n5_base_path = output_file_path + "/interestpoints.n5"
output_path = n5_base_path

start_matching(xml_input_file, n5_base_path, output_path)

# SOLVE
# -------------------------

# run params
# TODO - Update to automatically assign this in solver
fixed_views = ["timepoint: 18, setup: 0", "timepoint: 30, setup: 0"]
model = "affine"
alignment_option = 1
relative_threshold = 3.5
absolute_threshold = 7.0
min_matches = 3
damp = 0.4
max_iterations = 100000
max_allowed_error = 5.0
max_plateauwidth = 200

# input/output params
# file_source = file_source
# xml_bucket_name = output_bucket_name
# xml_file_path = 'output/dataset-detection.xml'
# data_prefix = n5_base_path
# xml_file_path_output = output_file_path + '/dataset-solve.xml'

file_source = file_source
xml_bucket_name = None
xml_file_path = xml_input_file
data_prefix = n5_base_path
xml_file_path_output = output_file_path + "/dataset-solve.xml"

solver = Solver(
    file_source,
    xml_file_path_output,
    xml_bucket_name,
    xml_file_path,
    data_prefix,
    fixed_views,
    model,
    alignment_option,
    relative_threshold,
    absolute_threshold,
    min_matches,
    damp,
    max_iterations,
    max_allowed_error,
    max_plateauwidth,
)
solver.run()
