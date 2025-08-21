from Rhapso.data_prep.xml_to_dictionary import XMLToDictionary
from Rhapso.detection.view_transform_models import ViewTransformModels
from Rhapso.detection.overlap_detection import OverlapDetection
from Rhapso.data_prep.metadata_builder import MetadataBuilder
from Rhapso.data_prep.image_reader import ImageReader
from Rhapso.detection.difference_of_gaussian import DifferenceOfGaussian
from Rhapso.detection.advanced_refinement import AdvancedRefinement
from Rhapso.detection.points_validation import PointsValidation
from Rhapso.detection.save_interest_points import SaveInterestPoints
import boto3
import ray

strategy = 'python'
run_type = 'ray'
dsxy = 4
dsz = 4
min_intensity = 0
max_intensity = 255 
                                                           
# Zarr - AWS s3
sigma = 4                                                    # tuning for find_peaks      
threshold = 0.01                                             # tuning for find_peaks
combine_distance = 0.5                                       # tuning for kd tree
mem_per_worker_bytes = 6 * 1024 * 1024 * 1024 
file_type = 'zarr'
file_source = 's3'
xml_file_path = "dataset.xml"
xml_bucket_name = "rhapso-zar-sample"
image_file_path = 's3://aind-open-data/exaSPIM_686951_2025-02-25_09-45-02_flatfield-correction_2025-06-10_06-58-54/SPIM.ome.zarr/'
image_bucket_name = "aind-open-data"
output_file_path = "output"
output_bucket_name = 'rhapso-matching-test'

# Tiff - Local
# sigma = 1.8                                                     # tuning for find_peaks
# threshold = 0.001                                               # tuning for find_peaks
# combine_distance = 0.5                                          # tuning for kd tree
# mem_per_worker_bytes = 0
# file_type = 'tiff'
# file_source = 'local'
# xml_file_path = '/Users/seanfite/Desktop/IP_TIFF_XML/dataset.xml'
# image_file_path =  '/Users/seanfite/Desktop/IP_TIFF_XML/' 
# output_file_path = '/Users/seanfite/Desktop/IP_TIFF_XML/output'
# xml_bucket_name = None
# image_bucket_name = None
# output_bucket_name = None

# Tiff - s3
# sigma = 1.8                                                     # tuning for find_peaks
# threshold = 0.001                                               # tuning for find_peaks
# combine_distance = 0.5                                          # tuning for kd tree
# mem_per_worker_bytes = 0
# offset = 0
# file_type = 'tiff'
# file_source = 's3'
# xml_file_path = 'IP_TIFF_XML/dataset.xml'
# image_file_path =  's3://rhapso-tif-sample/IP_TIFF_XML/' 
# output_file_path = 'output'
# xml_bucket_name = 'rhapso-tif-sample'
# image_bucket_name = 'rhapso-tif-sample'
# output_bucket_name = 'rhapso-matching-test'

# Initialize ray
ray.init()

# Get XML file
if file_source == 's3':
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=xml_bucket_name, Key=xml_file_path)
    xml_file = response['Body'].read().decode('utf-8')
elif file_source == 'local':
    with open(xml_file_path, 'r', encoding='utf-8') as file:
        xml_file = file.read()

# Load XML data into dataframes         
processor = XMLToDictionary(xml_file)
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

# Implement image chunking strategy as list of metadata 
metadata_loader = MetadataBuilder(dataframes, overlapping_area, image_file_path, file_type, dsz, dsxy, 
                                  mem_per_worker_bytes, sigma, run_type)
image_chunk_metadata = metadata_loader.run()
print("Metadata has loaded")

# Use Ray to distribute peak detection to image chunking metadata 
@ray.remote(memory=6 * 1024 * 1024 * 1024)
def process_peak_detection_task(chunk_metadata, dsxy, dsz, min_intensity, max_intensity, sigma, threshold):
    try:
        difference_of_gaussian = DifferenceOfGaussian(min_intensity, max_intensity, sigma, threshold)
        image_fetcher = ImageReader(file_type)
        view_id, interval, image_chunk, offset, lb = image_fetcher.run(chunk_metadata, dsxy, dsz)
        interest_points = difference_of_gaussian.run(image_chunk, dsxy, dsz, offset, lb)

        return {
            'view_id': view_id,
            'interval_key': interval,
            'interest_points': interest_points['interest_points'],
            'intensities': interest_points['intensities']
        }
    except Exception as e:
        return {'error': str(e), 'view_id': chunk_metadata.get('view_id', 'unknown')}

# Submit tasks to Ray
futures = [process_peak_detection_task.remote(chunk_metadata, dsxy, dsz, min_intensity, max_intensity, sigma, threshold)
    for chunk_metadata in image_chunk_metadata
]

# Gather and process results
results = ray.get(futures)
final_peaks = [r for r in results if 'error' not in r]
print("Peak detection is done")

# Consolidate points and filter overlap duplicates using kd tree
advanced_refinement = AdvancedRefinement(final_peaks, combine_distance)
consolidated_data = advanced_refinement.run()
print("Advanced refinement is done")

# Print points metrics / validation tools
points_validation = PointsValidation(consolidated_data)
points_validation.run()

# Save final interest points
save_interest_points = SaveInterestPoints(dataframes, consolidated_data, xml_file_path, xml_bucket_name, output_bucket_name, 
                                          output_file_path, dsxy, dsz, min_intensity, max_intensity, sigma, threshold, 
                                          file_source)
save_interest_points.run()
print("Interest points saved")

print("Interest point detection is done")


# DEBUG
# ---------------------------------

# Detect interest points using DoG algorithm
# difference_of_gaussian = DifferenceOfGaussian(min_intensity, max_intensity, sigma, threshold)
# image_fetcher = ImageFetching(file_type, mem_per_worker_bytes)
# final_peaks = []
# interest_point_count = 0

# for data in image_chunk_metadata: 
#     view_id, interval, image_chunk, offset, lb = image_fetcher.run(data, dsxy, dsz)
#     points = difference_of_gaussian.run(image_chunk, dsxy, dsz, offset, lb)
#     interest_points = points['interest_points']
#     interest_point_count += len(interest_points)
#     intensities = points['intensities']
    
#     final_peaks.append({
#         'view_id': view_id,
#         'interval_key': interval,
#         'interest_points': interest_points,
#         'intensities': intensities
#     })
# print(f"Difference of gaussian is done, total interest points found: {interest_point_count}")