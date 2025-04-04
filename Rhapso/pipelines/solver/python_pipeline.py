from Rhapso.solver.input_validation import InputValidation
from Rhapso.data_prep.xml_to_dataframe import XMLToDataFrame
from Rhapso.solver.global_optimization import GlobalOptimization
from Rhapso.detection.view_transform_models import ViewTransformModels
from Rhapso.solver.data_prep import DataPrep
from Rhapso.solver.model_and_tile_setup import ModelAndTileSetup
from Rhapso.solver.align_tiles import AlignTiles
from Rhapso.solver.save_results import SaveResults
import boto3

file_source = 'local'
xml_file_path_output = "IP_TIFF_XML/dataset.xml"
xml_bucket_name = "rhapso-tif-sample"
xml_file_path = "/Users/seanfite/Desktop/IP_TIFF_XML-rhapso/output/dataset-detection.xml"
data_prefix = "/Users/seanfite/Desktop/IP_TIFF_XML/output/interestpoints.n5/" 
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

# Get affine matrices from view registration dataframe
create_models = ViewTransformModels(dataframes)
view_transform_matrices = create_models.run()
print("Transforms models have been created")

# Get data from n5 folders
data_prep = DataPrep(dataframes['view_interest_points'], view_transform_matrices, fixed_views, data_prefix)
connected_views, corresponding_interest_points, interest_points, label_map_global, view_id_set = data_prep.run()
print("Data prep is complete")

# Create models, tiles, and point matches
model_and_tile_setup = ModelAndTileSetup(connected_views, corresponding_interest_points, interest_points, 
                                         view_transform_matrices, view_id_set, label_map_global)
tiles, model, pmc = model_and_tile_setup.run()
print("Models and tiles created")

# Use point matches to transform models
align_tiles = AlignTiles(tiles, pmc, fixed_views)
tiles = align_tiles.run()
print("Tiles are aligned")

# Update all points with transform models and iterate through all tiles (views) and optimize alignment
global_optimization = GlobalOptimization(tiles, pmc, fixed_views, data_prefix, alignment_option, relative_threshold,
                                        absolute_threshold, min_matches, damp, max_iterations, max_allowed_error, 
                                        max_plateauwidth, model)
tiles = global_optimization.run()
print("Global optimization complete")

# Save results to xml - one new affine matrix per view registration
save_results = SaveResults(tiles, xml_file, xml_bucket_name, xml_file_path_output, fixed_views)
save_results.run()
print("Results have been saved")

print("Solve is done")