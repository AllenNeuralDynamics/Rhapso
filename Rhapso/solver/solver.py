from Rhapso.data_prep.xml_to_dataframe import XMLToDataFrame
from Rhapso.solver.global_optimization import GlobalOptimization
from Rhapso.detection.view_transform_models import ViewTransformModels
from Rhapso.solver.data_prep import DataPrep
from Rhapso.solver.model_and_tile_setup import ModelAndTileSetup
from Rhapso.solver.align_tiles import AlignTiles
from Rhapso.solver.save_results import SaveResults
import boto3

# This class implements the Solver pipeline

class Solver:
    def __init__(self, file_source, xml_file_path_output, xml_bucket_name, xml_file_path, data_prefix, fixed_views,
                 model, alignment_option, relative_threshold, absolute_threshold, min_matches, damp, max_iterations,
                 max_allowed_error, max_plateauwidth):
        
        self.file_source = file_source
        self.xml_file_path_output = xml_file_path_output
        self.xml_bucket_name = xml_bucket_name
        self.xml_file_path = xml_file_path
        self.data_prefix = data_prefix
        self.fixed_views = fixed_views
        self.model = model
        self.alignment_option = alignment_option
        self.relative_threshold = relative_threshold
        self.absolute_threshold = absolute_threshold
        self.min_matches = min_matches
        self.damp = damp
        self.max_iterations = max_iterations
        self.max_allowed_error = max_allowed_error
        self.max_plateauwidth = max_plateauwidth
        self.s3 = boto3.client('s3')

    def solve(self):
        def fetch_from_s3(s3, bucket_name, input_file):
            response = s3.get_object(Bucket=bucket_name, Key=input_file)
            return response['Body'].read().decode('utf-8')

        def fetch_local_xml(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()

        # Fetch xml data
        if self.file_source == 's3':
            xml_file = fetch_from_s3(self.s3, self.xml_bucket_name, self.xml_file_path) 
        elif self.file_source == 'local':
            xml_file = fetch_local_xml(self.xml_file_path)

        # Load XML data into dataframes         
        processor = XMLToDataFrame(xml_file)
        dataframes = processor.run()
        print("XML loaded")

        # Get affine matrices from view registration dataframe
        create_models = ViewTransformModels(dataframes)
        view_transform_matrices = create_models.run()
        print("Transforms models have been created")

        # Get data from n5 folders
        data_prep = DataPrep(dataframes['view_interest_points'], view_transform_matrices, self.fixed_views, self.data_prefix, 
                             self.file_source)
        connected_views, corresponding_interest_points, interest_points, label_map_global, view_id_set = data_prep.run()
        print("Data prep is complete")

        # Create models, tiles, and point matches
        model_and_tile_setup = ModelAndTileSetup(connected_views, corresponding_interest_points, interest_points, 
                                                view_transform_matrices, view_id_set, label_map_global)
        tiles, model, pmc = model_and_tile_setup.run()
        print("Models and tiles created")

        # Use point matches to transform models
        align_tiles = AlignTiles(tiles, pmc, self.fixed_views)
        tiles = align_tiles.run()
        print("Tiles are aligned")

        # Update all points with transform models and iterate through all tiles (views) and optimize alignment
        global_optimization = GlobalOptimization(tiles, pmc, self.fixed_views, self.data_prefix, self.alignment_option, self.relative_threshold,
                                                self.absolute_threshold, self.min_matches, self.damp, self.max_iterations, self.max_allowed_error, 
                                                self.max_plateauwidth, model)
        tiles = global_optimization.run()
        print("Global optimization complete")

        # Save results to xml - one new affine matrix per view registration
        save_results = SaveResults(tiles, xml_file, self.xml_bucket_name, self.xml_file_path_output, self.fixed_views)
        save_results.run()
        print("Results have been saved")

        print("Solve is done")
    
    def run(self):
        self.solve()