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
            # Load complete dataset xml information
            data_global = self.parser.parse()
            
            # Use data_global for all subsequent operations
            view_ids_global = data_global['viewsInterestPoints']
            view_registrations = data_global['viewRegistrations']
            
            # Set up view groups using complete dataset info
            setup = self.parser.setup_groups(view_registrations)
            
            # Build label map using view IDs only
            label_map_global = self.data_loader.build_label_map(view_ids_global)
            
            # Process all view pairs for matching
            all_results = []  # Store all matching results across pairs
            total_pairs = len(setup['pairs'])  # Total number of view pairs to process

            # Iterate through each pair of views to perform matching
            for idx, pair in enumerate(setup['pairs'], 1):
                viewA, viewB = pair  # Unpack the current pair of view IDs

                # Retrieve human-readable labels for each view, fallback to "Unknown" if missing
                labelA = label_map_global.get(viewA, "Unknown")
                labelB = label_map_global.get(viewB, "Unknown")

                # Print progress and which pair is being processed
                print(f"Processing pair {idx}/{total_pairs}: ({viewA}, {viewB}) with labels ({labelA}, {labelB})")

                # Run the matching task for the current pair and get results
                task_result = self._process_matching_task(pair, label_map_global)

                # Add the results for this pair to the global results list (if any)
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
            
            def get_transformed_points(view_id):
                """Retrieve and transform interest points for a given view."""
                view_info = view_data[view_id]
                loc_path = f"{view_info['path']}/{view_info['label']}/interestpoints/loc"
                raw_points = self.data_loader.load_interest_points_from_path(loc_path)
                transform = self.data_loader.get_transformation_matrix(view_id, view_data, view_registrations)
                return self.data_loader.transform_interest_points(raw_points, transform)

            # Retrieve and transform interest points for both views
            pointsA = get_transformed_points(viewA)
            pointsB = get_transformed_points(viewB)

            print(f"Loaded {len(pointsA)} transformed points for viewA: {viewA}")
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
