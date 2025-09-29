from Rhapso.solver.xml_to_dataframe_solver import XMLToDataFrameSolver
from Rhapso.solver.get_xml import GetXML
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
    def __init__(self, run_type, relative_threshold, absolute_threshold, min_matches, damp, max_iterations, max_allowed_error,
                 fixed_tile, max_plateauwidth, metrics_output_path, **kwargs):
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
        
        self.groups = False
        self.s3 = boto3.client('s3')
        suffixes = sorted({
            k.split('_c')[-1]
            for k in kwargs
            if (f"_{self.run_type}_c" in k) and k.split('_c')[-1].isdigit()
        }, key=int)

        self.extra_xmls = []
        for s in suffixes:
            solver_xml = kwargs.get(f"xml_file_path_solver_{self.run_type}_c{s}")
            output_xml = kwargs.get(f"xml_file_path_output_{self.run_type}_c{s}")
            n5_path = kwargs.get(f"n5_input_path_c{s}")
            if solver_xml or output_xml:
                self.extra_xmls.append({
                    "solver_xml": solver_xml,     
                    "output_xml": output_xml, 
                    "n5_path": n5_path    
                })

    def solve(self):
        # Get XML file(s)
        xml_fetcher = GetXML(self.extra_xmls)
        xml = xml_fetcher.run()

        # Load XML data into dataframes        
        processor = XMLToDataFrameSolver(xml)
        dataframes = processor.run()
        print("XML loaded")

        # Get affine matrices from view registration dataframe
        create_models = ViewTransformModels(dataframes)
        view_transform_matrices = create_models.run()
        print("Transforms models have been created")

        # Get data from n5 folders
        data_prep = DataPrep(view_transform_matrices)
        data_map = data_prep.run()
        print("Data prep is complete")

        # Create models, tiles, and point matches
        model_and_tile_setup = ModelAndTileSetup(data_map)
        pmc, view_id_set = model_and_tile_setup.run()
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
            # TODO - Updated dataframes implementation to handle list
            # Combine splits into groups
            connected_graphs = ConnectedGraphs(tiles, dataframes)
            wlpmc, groups = connected_graphs.run()
            print("Tiles have been grouped")

            # TODO - Updated dataframes implementation to handle list
            # Find point matches and save to each tile
            compute_tiles = ComputeTiles(wlpmc, view_id_set, groups, dataframes, self.run_type)
            tiles_round_2, view_map = compute_tiles.run()
            print("Tiles are computed")

            # Use matches to update transformation matrices to represent rough alignment
            pre_align_tiles = PreAlignTiles(self.min_matches, self.run_type)
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
        save_results = SaveResults(tiles, xml, self.run_type, validation_stats)
        save_results.run()
        print("Results have been saved")
    
    def run(self):
        self.solve()