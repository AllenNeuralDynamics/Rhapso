from Rhapso.solver.xml_to_dataframe_solver import XMLToDataFrameSolver
from Rhapso.solver.global_optimization import GlobalOptimization
from Rhapso.solver.view_transforms import ViewTransformModels
from Rhapso.solver.data_prep import DataPrep
from Rhapso.solver.model_and_tile_setup import ModelAndTileSetup
from Rhapso.solver.compute_tiles import ComputeTiles
from Rhapso.solver.pre_align_tiles import PreAlignTiles
from Rhapso.solver.connected_graphs import ConnectedGraphs
from Rhapso.solver.concatenate_models import ConcatenateModels
from Rhapso.solver.save_results import SaveResults
import boto3

"""
This class implements the Solver pipeline for rigid, affine, and split-affine optimizations
"""

class Solver:
    def __init__(self, xml_file_path_output, n5_input_path, xml_file_path, run_type, relative_threshold, absolute_threshold, 
                 min_matches, damp, max_iterations, max_allowed_error, max_plateauwidth, metrics_output_path, fixed_tile):
        self.xml_file_path_output = xml_file_path_output
        self.n5_input_path = n5_input_path
        self.xml_file_path = xml_file_path
        self.run_type = run_type
        self.relative_threshold = relative_threshold
        self.absolute_threshold = absolute_threshold
        self.min_matches = min_matches
        self.damp = damp
        self.max_iterations = max_iterations
        self.max_allowed_error = max_allowed_error
        self.max_plateauwidth = max_plateauwidth
        self.metrics_output_path = metrics_output_path
        self.fixed_tile = fixed_tile
        self.groups = None
        self.s3 = boto3.client('s3')

    def solve(self):
        # Get XML file
        if self.xml_file_path.startswith("s3://"):
            no_scheme = self.xml_file_path.replace("s3://", "", 1)
            bucket, key = no_scheme.split("/", 1)
            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=bucket, Key=key)
            xml_file = response["Body"].read().decode("utf-8")
        else:  
            with open(self.xml_file_path, "r", encoding="utf-8") as f:
                xml_file = f.read()

        # Load XML data into dataframes         
        processor = XMLToDataFrameSolver(xml_file)
        dataframes = processor.run()
        print("XML loaded")

        # Get affine matrices from view registration dataframe
        create_models = ViewTransformModels(dataframes)
        view_transform_matrices = create_models.run()
        print("Transforms models have been created")

        # Get data from n5 folders
        data_prep = DataPrep(dataframes['view_interest_points'], view_transform_matrices, self.xml_file_path,
                             self.n5_input_path)
        connected_views, corresponding_interest_points, interest_points, label_map_global, view_id_set = data_prep.run()
        print("Data prep is complete")

        # Create models, tiles, and point matches
        model_and_tile_setup = ModelAndTileSetup(connected_views, corresponding_interest_points, interest_points, 
                                                view_transform_matrices, view_id_set, label_map_global)
        pmc = model_and_tile_setup.run()
        print("Models and tiles created")    
            
        # Find point matches and save to each tile
        compute_tiles = ComputeTiles(pmc, view_id_set, self.groups, dataframes, self.run_type)
        tiles, view_map = compute_tiles.run()
        print("Tiles are computed")

        # Use matches to update transformation matrices to represent rough alignment
        pre_align_tiles = PreAlignTiles(self.min_matches, self.run_type, self.fixed_tile)
        tc = pre_align_tiles.run(tiles)
        print("Tiles are pre-aligned")

        # Update all points with transform models and iterate through all tiles (views) and optimize alignment
        global_optimization = GlobalOptimization(tc, self.relative_threshold, self.absolute_threshold, self.min_matches, self.damp, 
                                                self.max_iterations, self.max_allowed_error, self.max_plateauwidth, self.run_type, self.metrics_output_path)
        tiles, validation_stats = global_optimization.run()
        print("Global optimization complete")
        
        if(self.run_type == "split-affine"):
            
            # Combine splits into groups
            connected_graphs = ConnectedGraphs(tiles, dataframes)
            wlpmc, groups = connected_graphs.run()
            print("Tiles have been grouped")

            # Find point matches and save to each tile
            compute_tiles = ComputeTiles(wlpmc, view_id_set, groups, dataframes, self.run_type)
            tiles_round_2, view_map = compute_tiles.run()
            print("Tiles are computed")

            # Use matches to update transformation matrices to represent rough alignment
            pre_align_tiles = PreAlignTiles(self.min_matches, self.run_type, self.fixed_tile)
            tc = pre_align_tiles.run(tiles_round_2)
            print("Tiles are pre-aligned")

            # Update all points with transform models and iterate through all tiles (views) and optimize alignment
            global_optimization = GlobalOptimization(tc, self.relative_threshold, self.absolute_threshold, self.min_matches, self.damp, 
                                                    self.max_iterations, self.max_allowed_error, self.max_plateauwidth, self.run_type, self.metrics_output_path)
            tiles_round_2, validation_stats_round_2 = global_optimization.run()
            print("Global optimization complete")

            # Combine models/metrics for round 1 and 2 
            concatenate_models = ConcatenateModels(tiles, tiles_round_2, groups, validation_stats, validation_stats_round_2, view_map)
            tiles, validation_stats = concatenate_models.run()
            print("Models and metrics have been combined")

        # Save results to xml - one new affine matrix per view registration
        save_results = SaveResults(tiles, xml_file, self.xml_file_path_output, self.run_type, validation_stats, self.n5_input_path)
        save_results.run()
        print("Results have been saved")
    
    def run(self):
        self.solve()