from Rhapso.data_prep.xml_to_dictionary import XMLToDictionary
from Rhapso.solver.global_optimization import GlobalOptimization
from Rhapso.solver.view_transforms import ViewTransformModels
from Rhapso.solver.data_prep import DataPrep
from Rhapso.solver.model_and_tile_setup import ModelAndTileSetup
from Rhapso.solver.compute_tiles import ComputeTiles
from Rhapso.solver.pre_align_tiles import PreAlignTiles
from Rhapso.solver.save_results import SaveResults
import boto3

# file_source = 's3'
# xml_bucket_name = "rhapso-matching-test"
# xml_file_path = "ip_rigid_alignment/rhapso_solver.xml"
# data_prefix = "ip_rigid_alignment/interestpoints.n5" 
# xml_file_path_output = "rhapso-solve.xml"
# metrics_output_path = "output/metrics/metrics.json"

file_source = 'local'
xml_bucket_name = None
xml_file_path = '/Users/seanfite/Desktop/ip_rigid_alignment/bigstitcher_rigid.xml'
data_prefix = '/Users/seanfite/Desktop/ip_rigid_alignment/interestpoints.n5'
xml_file_path_output = '/Users/seanfite/Desktop/ip_rigid_alignment/rhapso_solver.xml'
metrics_output_path = "output/metrics/metrics.json"

fixed_views = [ 'timepoint: 0, setup: 0' ]

run_type = "affine"
relative_threshold = 3.5
absolute_threshold = 7.0
min_matches = 3
damp = 1.0
max_iterations= 100000
max_allowed_error= float('inf')
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
processor = XMLToDictionary(xml_file)
dataframes = processor.run()
print("XML loaded")

# Get affine matrices from view registration dataframe
create_models = ViewTransformModels(dataframes)
view_transform_matrices = create_models.run()
print("Transforms models have been created")

# Get data from n5 folders
data_prep = DataPrep(dataframes['view_interest_points'], view_transform_matrices, fixed_views, data_prefix, file_source)
connected_views, corresponding_interest_points, interest_points, label_map_global, view_id_set = data_prep.run()
print("Data prep is complete")

# Create models, tiles, and point matches
model_and_tile_setup = ModelAndTileSetup(connected_views, corresponding_interest_points, interest_points, 
                                         view_transform_matrices, view_id_set, label_map_global)
model, pmc = model_and_tile_setup.run()
print("Models and tiles created")

# Find point matches and save to each tile
compute_tiles = ComputeTiles(pmc, fixed_views, view_id_set)
tiles = compute_tiles.run()
print("Tiles are computed")

# Use matches to update transformation matrices to represent rough alignment
pre_align_tiles = PreAlignTiles()
tc = pre_align_tiles.run(tiles)
print("Tiles are pre-aligned")

# Update all points with transform models and iterate through all tiles (views) and optimize alignment
global_optimization = GlobalOptimization(tc, fixed_views, data_prefix, relative_threshold,
                                        absolute_threshold, min_matches, damp, max_iterations, max_allowed_error, 
                                        max_plateauwidth, run_type, metrics_output_path)
tiles = global_optimization.run()
print("Global optimization complete")

# Save results to xml - one new affine matrix per view registration
save_results = SaveResults(tiles, xml_file, xml_bucket_name, xml_file_path_output, fixed_views, file_source)
save_results.run()
print("Results have been saved")

print("Solve is done")