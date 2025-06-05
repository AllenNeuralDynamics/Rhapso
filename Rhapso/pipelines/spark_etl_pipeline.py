from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import SparkSession
from Rhapso.data_prep.xml_to_dataframe import XMLToDataFrame
from Rhapso.detection.view_transform_models import ViewTransformModels
from Rhapso.detection.overlap_detection import OverlapDetection
from Rhapso.data_prep.load_image_data import LoadImageData
from Rhapso.data_prep.glue_crawler import GlueCrawler
from Rhapso.data_prep.serialize_image_chunks import SerializeImageChunks
from Rhapso.data_prep.deserialize_image_chunks import DeserializeImageChunks
from Rhapso.detection.difference_of_gaussian import DifferenceOfGaussian
from Rhapso.detection.advanced_refinement import AdvancedRefinement
from Rhapso.detection.save_interest_points import SaveInterestPoints
from Rhapso.pipelines.matching.matching_pipeline_classes import MatchingPipeline
from Rhapso.solver.solver import Solver
import sys
import ast
import boto3

# Rhapso - Spark ETL Pipeline

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext.getOrCreate() 
glueContext = GlueContext(sc)
spark = SparkSession.builder.appName("Interest Point Detection").getOrCreate()
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
s3 = boto3.client('s3')

# SPARK ETL PARAMS
parquet_bucket_path = 's3://interest-point-detection/ipd_zarr_staging_glue/'
crawler_name = "ipd_zarr_staging_glue"
crawler_s3_path = "s3://interest-point-detection/ipd_zarr_staging_glue/"
crawler_database_name = "ipd_zarr_staging_glue"
crawler_iam_role = "arn:aws:iam::443370675126:role/rhapso-s3"
glue_database = 'ipd_zarr_staging_glue'
glue_table_name = 'ipd_zarr_staging_glue'

# run params
strategy = 'python'
dsxy = 30
dsz = 30
min_intensity = 0
max_intensity = 255
sigma = 1.8
threshold = 0.008

# input/output params
file_type = 'zarr'
file_source = 's3'
xml_file_path = "dataset.xml"
xml_bucket_name = "rhapso-zar-sample"
image_file_path = 's3://aind-open-data/exaSPIM_708365_2024-04-29_12-46-15/SPIM.ome.zarr/'
image_bucket_name = "aind-open-data"
output_file_path = "output"
output_bucket_name = 'rhapso-zarr-glue'

# file_type = 'tiff'
# file_source = 'local'
# xml_file_path = '/Users/seanfite/Desktop/IP_TIFF_XML/dataset.xml'
# image_file_path =  '/Users/seanfite/Desktop/IP_TIFF_XML/'
# output_file_path = '/Users/seanfite/Desktop/IP_TIFF_XML/output'
# xml_bucket_name = None
# image_bucket_name = None
# output_bucket_name = None

# INTEREST POINT DETECTION
# --------------------------

# data input source
s3 = boto3.client('s3')

def fetch_from_s3(s3, bucket_name, input_file):
    response = s3.get_object(Bucket=bucket_name, Key=input_file)
    return response['Body'].read().decode('utf-8')

def fetch_local_xml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

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

# Flatten and serialize images to parquet
serialize_image_chunks = SerializeImageChunks(all_image_data, parquet_bucket_path)
serialized_images_dyf = serialize_image_chunks.run()
print("Serialized image data")

# Create and start crawler
glue_crawler = GlueCrawler(crawler_name, crawler_s3_path, crawler_database_name, crawler_iam_role)
glue_crawler.run()
print("Glue crawler created and started")

# Create dynamic frame using crawler schema
image_data_dyf = glueContext.create_dynamic_frame.from_catalog(
    database = glue_database,
    table_name = glue_table_name,
    transformation_ctx = "dynamic_frame"
)
print("Dynamic frame loaded")

# Detect interest points using DoG algorithm - custom transform
difference_of_gaussian = DifferenceOfGaussian(min_intensity, max_intensity, sigma, threshold)
deserialize_image_chunks = DeserializeImageChunks()
def interest_point_detection(record):
    try:
        view_id, interval, image_chunk = deserialize_image_chunks.run(record)     
        dog_results = difference_of_gaussian.run(image_chunk, dsxy, dsz)
        interest_points = dog_results['interest_points']
        intensities = dog_results['intensities']
        interest_points_as_strings = [str(point) for point in interest_points]
        results_dict = {
            'view_id': str(view_id),
            'interval_key': str(interval),
            'interest_points': interest_points_as_strings,
            'intensities': intensities.tolist() 
        }
        return results_dict
    except Exception as e:
        print("Error processing record:", str(e))
        return {}
mapped_results_dyf = image_data_dyf.map(interest_point_detection, transformation_ctx="map_interest_points")
print("Difference of gaussian is done")

# Format results out of dynamic frame for advanced refinement
result_df = mapped_results_dyf.toDF()
interest_points_list = []
for row in result_df.collect():
    view_id = row['view_id']
    interval_key = row['interval_key']
    interest_points = [ast.literal_eval(point) for point in row['interest_points']]
    intensities = row['intensities']
    interest_points_list.append({
        'view_id': view_id,
        'interval_key': interval_key,
        'interest_points': interest_points,
        'intensities': intensities
    })
print("Results formatted and ready for advanced refinement")

# Integrate final peaks into kd tree for refinement
advanced_refinement = AdvancedRefinement(interest_points_list)
consolidated_data = advanced_refinement.run()
print("Advanced refinement is done")

# Save interest points to N5 and metadata to XML
save_interest_points = SaveInterestPoints(dataframes, consolidated_data, xml_file_path, xml_bucket_name, 
                                        output_bucket_name, output_file_path, dsxy, dsz, min_intensity, 
                                        max_intensity, sigma, threshold, file_source)
save_interest_points.run()
print("Interest points saved")

print("Interest point detection is done")

# MATCHING
# -------------------------

xml_input_file = "s3://rhapso-zarr-glue/output/dataset-detection.xml"
n5_base_path = 's3://rhapso-zarr-glue/output/interestpoints.n5'
output_path = n5_base_path

# xml_input_file = output_file_path + "/dataset-detection.xml"
# n5_base_path = output_file_path + '/interestpoints.n5'
# output_path = n5_base_path

MatchingPipeline(xml_input_file, n5_base_path, output_path)

# SOLVE
# -------------------------

# run params
# TODO - Update to automatically assign this in solver
fixed_views = [ 'timepoint: 0, setup: 0']
model = "affine"
alignment_option = 1
relative_threshold = 3.5
absolute_threshold = 7.0
min_matches = 3
damp = .4
max_iterations= 100000
max_allowed_error= 5.0
max_plateauwidth = 200

# input/output params
file_source = file_source
xml_bucket_name = output_bucket_name
xml_file_path = 'output/dataset-detection.xml'
data_prefix = n5_base_path
xml_file_path_output = output_file_path + '/dataset-solve.xml'

# file_source = file_source
# xml_bucket_name = None
# xml_file_path = xml_input_file
# data_prefix = n5_base_path
# xml_file_path_output = output_file_path + '/dataset-solve.xml'

solver = Solver(file_source, xml_file_path_output, xml_bucket_name, xml_file_path, data_prefix, fixed_views, model, 
                alignment_option, relative_threshold, absolute_threshold, min_matches, damp, max_iterations, max_allowed_error,
                max_plateauwidth)
solver.run()

job.commit()