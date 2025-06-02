from Rhapso.matching.xml_parser import XMLParser
from Rhapso.matching.data_loader import DataLoader
from Rhapso.matching.matcher import Matcher
from Rhapso.matching.result_saver import ResultSaver
from Rhapso.matching.ransac import RANSAC
import os

class MatchingPipeline:
    def __init__(self, xml_file, interest_points_folder):
        self.xml_file = xml_file
        self.parser = XMLParser(xml_file)
        # Pass both the interest_points_folder and xml_file directory
        self.data_loader = DataLoader(interest_points_folder)
        self.data_loader.xml_base_path = os.path.dirname(xml_file)  # Store XML directory
        
        # Create matcher with explicit model parameters using nested enums
        self.matcher = Matcher(
            transform_model=Matcher.TransformationModel.AFFINE,
            reg_model=Matcher.RegularizationModel.RIGID,
            lambda_val=0.1
        )

    def run(self):
        try:
            # Load complete dataset information
            data_global = self.parser.parse()
            
            # Print parsed view information for debugging
            view_ids_global = data_global['viewsInterestPoints']
            print(f"Found {len(view_ids_global)} views in dataset:")
            for view_id, info in view_ids_global.items():
                print(f"  View {view_id}: {info}")
            
            # Use data_global for all subsequent operations
            view_ids_global = data_global['viewsInterestPoints']
            view_registrations = data_global['viewRegistrations']
            
            # Set up view groups using complete dataset info
            setup = self.parser.setup_groups(view_registrations)
            
            # Build label map using view IDs only
            label_map_global = self.data_loader.build_label_map(view_ids_global)
            
            # Process all view pairs for matching
            all_results = []
            total_pairs = len(setup['pairs'])
            for idx, pair in enumerate(setup['pairs'], 1):
                viewA, viewB = pair
                labelA = label_map_global.get(viewA, "Unknown")
                labelB = label_map_global.get(viewB, "Unknown")
                print(f"Processing pair {idx}/{total_pairs}: ({viewA}, {viewB}) with labels ({labelA}, {labelB})")
                task_result = self._process_matching_task(pair, label_map_global)
                all_results.extend(task_result if task_result else [])

            print(f"Total matches found: {len(all_results)}")

            # Save correspondences for each view that has matches
            # for view_id in view_ids_global:
            #     corresponding_matches = [r for r in all_results if r[0] == view_id or r[1] == view_id]
            #     if corresponding_matches:
            #         self.saver.save_correspondences_for_view(view_id, corresponding_matches, data_global)
            
            # Matching is now finished!

        except Exception as e:
            print(f"Error during matching pipeline execution: {e}")

    def _process_matching_task(self, pair, label_map):
        """Process a single matching task"""
        try:
            viewA, viewB = pair

            # Get view data for path lookup
            data_global = self.parser.get_data_global()
            view_data = data_global['viewsInterestPoints']
            view_registrations = data_global['viewRegistrations']
            
            print(f"🔧 Processing viewA: {viewA}")
            
            # Get complete path to the 'loc' folder for interest points
            view_info_A = view_data[viewA]
            loc_path_A = f"{view_info_A['path']}/{view_info_A['label']}/interestpoints/loc"
            print(f"loc_path_A: {loc_path_A}")
            
            # Debug: Show the store configuration
            print(f"DataLoader base_path: {self.data_loader.base_path}")
            print(f"DataLoader store type: {type(self.data_loader.store)}")
            
            # Load interest points directly from the 'loc' dataset
            raw_pointsA = self.data_loader.load_interest_points_from_path(loc_path_A)
            print(f"raw_pointsA shape: {raw_pointsA.shape}")
            
            transformA = self.data_loader.get_transformation_matrix(viewA, view_data, view_registrations)
            print(f"transformA: {transformA}")
            
            pointsA = self.data_loader.transform_interest_points(raw_pointsA, transformA)
            print(f"pointsA shape: {pointsA.shape}")

            print(f"Loaded {len(pointsA)} transformed points for viewA: {viewA}")

            print(f"🔧 Processing viewB: {viewB}")
            view_info_B = view_data[viewB]
            loc_path_B = f"{view_info_B['path']}/{view_info_B['label']}/interestpoints/loc"
            raw_pointsB = self.data_loader.load_interest_points_from_path(loc_path_B)
            transformB = self.data_loader.get_transformation_matrix(viewB, view_data, view_registrations)
            pointsB = self.data_loader.transform_interest_points(raw_pointsB, transformB)
            print(f"Loaded {len(pointsB)} transformed points for viewB: {viewB}")
            
            # # Get candidates using matcher
            # candidates = self.matcher._get_candidates(pointsA, pointsB)
            
            # # Compute matches using geometric matcher
            # matches = self.matcher._compute_match(candidates, pointsA, pointsB)
            
            # # Filter matches using RANSAC
            filtered_matches = []
            # filtered_matches = self.ransac.filter_matches(pointsA, pointsB, matches)
            
            return [(viewA, viewB, m[0], m[1]) for m in filtered_matches]
        except Exception as e:
            print(f"ERROR: Failed in _process_matching_task for pair {pair}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception details: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

def main(xml_file):

    # Get directory of xml_file and add 'interestpoints.n5'
    xml_dir = os.path.dirname(xml_file)
    interest_points_folder = os.path.join(xml_dir, 'interestpoints.n5')
    print(f"Derived interest points input n5 folder location: {interest_points_folder}")
    
    pipeline = MatchingPipeline(xml_file, interest_points_folder)
    pipeline.run()

if __name__ == "__main__":
    xml_file = "/home/martin/Documents/Allen/Data/IP_TIFF_XML_2/dataset.xml"
    main(xml_file)
