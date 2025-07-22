from Rhapso.data_prep.xml_to_dataframe import XMLToDataFrame
from Rhapso.detection.view_transform_models import ViewTransformModels
from Rhapso.detection.overlap_detection import OverlapDetection
from Rhapso.data_prep.metadata_builder import MetadataBuilder
from Rhapso.detection.difference_of_gaussian import DifferenceOfGaussian
from Rhapso.detection.advanced_refinement import AdvancedRefinement
from Rhapso.detection.points_validation import PointsValidation
from Rhapso.detection.save_interest_points import SaveInterestPoints
from dask import delayed
from dask import compute
import boto3

strategy = 'python'
run_type = 'dask'
dsxy = 2
dsz = 2
min_intensity = 0
max_intensity = 255

# Zarr - s3
# sigma = 4                                                    # tuning for find_peaks      
# threshold = 0.01                                             # tuning for find_peaks
# combine_distance = 0.5                                       # tuning for kd tree
# mem_per_worker_bytes = 0 
# offset = 0
# file_type = 'zarr'
# file_source = 's3'
# xml_file_path = "dataset.xml"
# xml_bucket_name = "rhapso-zar-sample"
# image_file_path = 's3://aind-open-data/exaSPIM_708365_2024-04-29_12-46-15/SPIM.ome.zarr/'
# image_bucket_name = "aind-open-data"
# output_file_path = "output"
# output_bucket_name = 'interest-point-detection'

# Tiff - local
# sigma = 1.8                                                     # tuning for find_peaks
# threshold = 0.001                                               # tuning for find_peaks
# combine_distance = 0.5                                          # tuning for kd tree
# mem_per_worker_bytes = 0
# offset = 0
# file_type = 'tiff'
# file_source = 'local'
# xml_file_path = '/Users/seanfite/Desktop/IP_TIFF_XML/dataset.xml'
# image_file_path =  '/Users/seanfite/Desktop/IP_TIFF_XML/' 
# output_file_path = '/Users/seanfite/Desktop/IP_TIFF_XML/interestpoints.n5/'
# xml_bucket_name = None
# image_bucket_name = None
# output_bucket_name = None

# TIFF - S3
sigma = 1.8                                                     # tuning for find_peaks
threshold = 0.001                                               # tuning for find_peaks
combine_distance = 0.5                                          # tuning for kd tree
mem_per_worker_bytes = 0
offset = 0
file_type = 'tiff'
file_source = 's3'
xml_file_path = 'IP_TIFF_XML/dataset.xml'
image_file_path =  's3://rhapso-tif-sample/IP_TIFF_XML/' 
output_file_path = 'output'
xml_bucket_name = 'rhapso-tif-sample'
image_bucket_name = 'rhapso-tif-sample'
output_bucket_name = 'rhapso-matching-test'

# Get XML file
if file_source == 's3':
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=xml_bucket_name, Key=xml_file_path)
    xml_file = response['Body'].read().decode('utf-8')
elif file_source == 'local':
    with open(xml_file_path, 'r', encoding='utf-8') as file:
        xml_file = file.read()

# Load XML data into dataframes         
processor = XMLToDataFrame(xml_file)
dataframes = processor.run()
print("XML loaded")

# Create view transform matrices 
create_models = ViewTransformModels(dataframes)
view_transform_matrices = create_models.run()
print("Transforms models have been created")

# Use view transform matrices to find areas of overlap
overlap_detection = OverlapDetection(
    view_transform_matrices,
    dataframes,
    dsxy,
    dsz,
    image_file_path,
    file_type
)
overlapping_area = overlap_detection.run()
print("Overlap detection is done")

# Create metadata with pathways to image chunks
metadata_loader = MetadataBuilder(
    dataframes,
    overlapping_area,
    image_file_path,
    file_type,
    dsz,
    dsxy,
    mem_per_worker_bytes,
    sigma,
    run_type
)
image_chunk_metadata = metadata_loader.run()
print("Metadata has loaded")

# difference_of_gaussian = DifferenceOfGaussian(min_intensity, max_intensity, sigma, threshold)
# final_peaks = []
# for image_data in image_chunk_metadata:
#     view_id = image_data['view_id']
#     interval_key = image_data['interval_key']
#     image_chunk = image_data['image_chunk']
#     lower_bound = interval_key[0]
    
#     peaks = difference_of_gaussian.run(image_chunk, dsxy, dsz, offset, lower_bound)
#     interest_points = peaks['interest_points']
#     intensities = peaks['intensities']

#     print(f"length of interest points: {len(interest_points)}")
    
#     final_peaks.append({
#         'view_id': view_id,
#         'interval_key': interval_key,
#         'interest_points': interest_points,
#         'intensities': intensities
#     })
# print("Difference of gaussian is done")

# Detect interest points using DoG algorithm
difference_of_gaussian = DifferenceOfGaussian(min_intensity, max_intensity, sigma, threshold)
final_peaks, delayed_results, delayed_keys = [], [], {}
for image_data in image_chunk_metadata:
    view_id, interval_key, image_chunk = image_data['view_id'], image_data['interval_key'], image_data['image_chunk']
    lower_bounds = interval_key[0]
    dog_result = delayed(difference_of_gaussian.run)(image_chunk, dsxy, dsz, offset, lower_bounds)
    delayed_results.append(dog_result)
    delayed_keys[dog_result] = (view_id, interval_key)
computed_results = compute(*delayed_results) 
for result, task in zip(computed_results, delayed_results):
    view_id, interval_key = delayed_keys[task]
    final_peaks.append({
        'view_id': view_id,
        'interval_key': interval_key,
        'interest_points': result['interest_points'],
        'intensities': result['intensities']
    })
print("Difference of gaussian is done")

# Consolidate points and filter overlap duplicates using kd tree
advanced_refinement = AdvancedRefinement(final_peaks, combine_distance)
consolidated_data = advanced_refinement.run()
print("Advanced refinement is done")

# Print points metrics / validation tools
points_validation = PointsValidation(consolidated_data)
points_validation.run()

# Save interest points
save_interest_points = SaveInterestPoints(
    dataframes, 
    consolidated_data, 
    xml_file_path, 
    xml_bucket_name, 
    output_bucket_name, 
    output_file_path, 
    dsxy, 
    dsz, 
    min_intensity, 
    max_intensity, 
    sigma, 
    threshold, 
    file_source)
save_interest_points.run()
print("Interest points saved")

print("Interest point detection is done")


# DEBUG
# ----------------

# PEAK DETECTION - processing iteratively for debugging
# final_peaks = []
# for image_data in image_chunk_metadata:
#     view_id = image_data['view_id']
#     interval_key = image_data['interval_key']
#     image_chunk = image_data['image_chunk']
#     lower_bound = interval_key[0]
    
#     peaks = difference_of_gaussian.run(image_chunk, dsxy, dsz, offset, lower_bound)
#     interest_points = peaks['interest_points']
#     intensities = peaks['intensities']

#     print(f"length of interest points: {len(interest_points)}")
    
#     final_peaks.append({
#         'view_id': view_id,
#         'interval_key': interval_key,
#         'interest_points': interest_points,
#         'intensities': intensities
#     })
# print("Difference of gaussian is done")