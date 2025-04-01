from Rhapso.data_prep.xml_to_dataframe import XMLToDataFrame
from Rhapso.detection.view_transform_models import ViewTransformModels
from Rhapso.detection.overlap_detection import OverlapDetection
from Rhapso.data_prep.load_image_data import LoadImageData
from Rhapso.detection.difference_of_gaussian import DifferenceOfGaussian
from Rhapso.detection.advanced_refinement import AdvancedRefinement
from Rhapso.detection.save_interest_points import SaveInterestPoints
import boto3
from dask import delayed
from dask import compute
from memory_profiler import profile

strategy = 'python'
dsxy = 4
dsz = 4
min_intensity = 0
max_intensity = 255
sigma = 1.8
threshold = 0.008

# TODO - we should be able to get image_file_path from the xml

# Tiff - s3
file_type = 'tiff'
file_source = 's3'
xml_file_path = "IP_TIFF_XML/dataset.xml"
xml_bucket_name = "rhapso-tif-sample"
image_file_path = 's3://rhapso-tif-sample/IP_TIFF_XML/'
image_bucket_name = "rhapso-tif-sample"
output_file_path = "output"
output_bucket_name = 'interest-point-detection'

# Zarr - s3
# file_type = 'zarr'
# file_source = 's3'
# xml_file_path = "dataset.xml"
# xml_bucket_name = "rhapso-zar-sample"
# image_file_path = 's3://aind-open-data/exaSPIM_708365_2024-04-29_12-46-15/SPIM.ome.zarr/'
# image_bucket_name = "aind-open-data"
# output_file_path = "output"
# output_bucket_name = 'interest-point-detection'

# Tiff - local
# file_type = 'tiff'
# file_source = 'local'
# xml_file_path = '/Users/seanfite/Desktop/AllenInstitute/Rhapso/Data/IP_TIFF_XML/dataset.xml'
# image_file_path =  '/Users/seanfite/Desktop/AllenInstitute/Rhapso/Data/IP_TIFF_XML/'
# output_file_path = '/Users/seanfite/Desktop/AllenInstitute/Rhapso/Data/IP_TIFF_XML/output'
# xml_bucket_name = None
# image_bucket_name = None
# output_bucket_name = None

# data input source
s3 = boto3.client('s3')

def fetch_from_s3(s3, bucket_name, input_file):
    response = s3.get_object(Bucket=bucket_name, Key=input_file)
    return response['Body'].read().decode('utf-8')

def fetch_local_xml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# INTEREST POINT DETECTION
# --------------------------

# Fetch xml data
if file_source == 's3':
    xml_file = fetch_from_s3(s3, xml_bucket_name, xml_file_path) 
elif file_source == 'local':
    xml_file = fetch_local_xml(xml_file_path)

# Load XML data into dataframes         
processor = XMLToDataFrame(xml_file)
dataframes = processor.run()
print("XML loaded")

# Create view transform matrices 
create_models = ViewTransformModels(dataframes)
view_transform_matrices = create_models.run()
print("Transforms models have been created")

# Use view transform matrices to find areas of overlap
overlap_detection = OverlapDetection(view_transform_matrices, dataframes, dsxy, dsz, image_file_path, file_type)
overlapping_area = overlap_detection.run()
print("Overlap detection is done")

# Load images 
images_loader = LoadImageData(dataframes, overlapping_area, dsxy, dsz, image_file_path, file_type)
all_image_data = images_loader.run()
print("Image data loaded")

# Detect interest points using DoG algorithm
difference_of_gaussian = DifferenceOfGaussian(min_intensity, max_intensity, sigma, threshold)

# BASE PYTHON VERSION - SLOWEST RUN TIME (DEV ONLY)
# final_peaks = []
# for image_data in all_image_data:
#     view_id = image_data['view_id']
#     interval_key = image_data['interval_key']
#     image_chunk = image_data['image_chunk']
#     points = difference_of_gaussian.run(image_chunk, dsxy, dsz)
#     interest_points = points['interest_points']
#     intensities = points['intensities']
#     final_peaks.append({
#         'view_id': view_id,
#         'interval_key': interval_key,
#         'interest_points': interest_points, 
#         'intensities': intensities 
#     })
# print("Difference of gaussian is done")

# BASE PYTHON MAP VERSION - SLOWER RUN TIME (COULD OPTIMIZE WITH PYTHON THREADING)
# def interest_point_detection(image_data):
#     view_id = image_data['view_id']
#     interval_key = image_data['interval_key']
#     image_chunk = image_data['image_chunk']
#     points = difference_of_gaussian.run(image_chunk, dsxy, dsz)
#     interest_points = points['interest_points']
#     intensities = points['intensities']
#     return {
#         'view_id': view_id,
#         'interval_key': interval_key,
#         'interest_points': interest_points,
#         'intensities': intensities
#     }
# mapped_results = map(interest_point_detection, all_image_data)
# final_peaks = list(mapped_results)
# print("Difference of gaussian is done")

# DASK MAP VERSION (SINGLE THREAD INTERPRETER ONLY)
final_peaks = []
delayed_results = []
delayed_keys = {} 
for image_data in all_image_data:
    view_id = image_data['view_id']
    interval_key = image_data['interval_key']
    image_chunk = image_data['image_chunk']
    dog_result = delayed(difference_of_gaussian.run)(image_chunk, dsxy, dsz)
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

advanced_refinement = AdvancedRefinement(final_peaks)
consolidated_data = advanced_refinement.run()
print("Advanced refinement is done")

save_interest_points = SaveInterestPoints(dataframes, consolidated_data, xml_file_path, xml_bucket_name, 
                                          output_bucket_name, output_file_path, dsxy, dsz, min_intensity, 
                                          max_intensity, sigma, threshold, file_source)
save_interest_points.run()
print("Interest points saved")

print("Interest point detection is done")