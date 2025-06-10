# -----------------------------------------------------------------------------
# matching_test_pipeline.py – Matching pipeline test script
#
# Setup & run:
#   1. python -m venv myNewEnv           # create a fresh Python virtual environment
#   2. source myNewEnv/bin/activate      # activate the virtual environment
#   3. pip install .[matching]           # install matching dependencies from setup.py
#   4. python Rhapso/pipelines/matching/matching_test_pipeline.py
#      # execute the test pipeline
# -----------------------------------------------------------------------------
'''
Good ctrl+f search points:

PROCESSING VIEW PAIR:

... transform interest points ...

📊 SUMMARY:
   View A ((18, 0)): 2755 transformed points
   View B ((18, 1)): 3149 transformed points

🔗 Matching points between viewA (18, 0) and viewB (18, 1)...
🔍 Finding candidates between point sets of size 2755 and 3149
🛠️ Creating descriptors...
🔍 Creating descriptors with 4 redundancy
🎯 Basis Point 0: [...]

... print all basis point analysis used to create descriptor(s) ...

🔢 Total descriptors created: 12596
📊 Created 11020 descriptors for A, 12596 descriptors for B
🌳 Built KDTree with 12596 nodes, 6 dimensions

🔍 _compute_matching() - Input Parameters:
  📊 descriptors_A size: 11020
  📊 descriptors_B size: 12596
  📊 difference_threshold: inf
  📊 ratio_of_distance: 3.0

... print every Lowe's ratio test ...

✅ _compute_matching() - Results:
  📊 Processed descriptors: 11020
  📊 Matches passed Lowe's ratio test: 785
  📊 Unique pairs after deduplication: 316
  📊 Final correspondences added: 785
  📊 Total correspondences in list: 785

🎯 Found 785 correspondence candidates after Lowe's ratio test
Found 785 correspondence candidates after Lowe's ratio test
RANSAC filtering retained 564 inlier matches
Processing pair 2/42: ((18, 0), (18, 2)) with labels ({'beads': 1.0}, {'beads': 1.0})

finished, now we process the next pair 
'''

from Rhapso.matching.xml_parser import XMLParser
from Rhapso.matching.data_loader import DataLoader
from Rhapso.matching.matcher import Matcher
from Rhapso.matching.result_saver import ResultSaver
from Rhapso.matching.ransac import RANSAC
import os

class MatchingPipeline:
    def __init__(self, xml_file, interest_points_folder, logging=None):
        # Set default logging configuration if none provided
        self.logging = {
            'detailed_descriptor_breakdown': True,
            'interest_point_transformation': True,
            'ratio_test_output': True,
            'basis_point_details': 3,  # Number of basis points to show details for
        }
        
        # Update with user-provided logging settings
        if logging is not None:
            self.logging.update(logging)
            
        self.xml_file = xml_file
        self.parser = XMLParser(xml_file)
        # Pass both the interest_points_folder and xml_file directory
        self.data_loader = DataLoader(interest_points_folder, logging=self.logging)
        self.data_loader.xml_base_path = os.path.dirname(xml_file)  # Store XML directory
        
        # Create matcher with explicit model parameters using nested enums
        self.matcher = Matcher(
            transform_model=Matcher.TransformationModel.AFFINE,
            reg_model=Matcher.RegularizationModel.RIGID,
            lambda_val=0.1,
            logging=self.logging
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
                print(f"\n🔍 Processing view {view_id}...")
                
                view_info = view_data[view_id]
                loc_path = f"{view_info['path']}/{view_info['label']}/interestpoints/loc"
                print(f"📁 Loading from path: {loc_path}")
                
                raw_points = self.data_loader.load_interest_points_from_path(loc_path)
                print(f"📥 Loaded {len(raw_points)} raw interest points")
                
                transform = self.data_loader.get_transformation_matrix(view_id, view_data, view_registrations)
                print(f"🔄 Applying transformation matrix for view {view_id}:")
                
                transformed_points = self.data_loader.transform_interest_points(raw_points, transform)
                
                return transformed_points

            # Retrieve and transform interest points for both views
            print(f"\n{'='*60}")
            print(f"PROCESSING VIEW PAIR: {viewA} ↔ {viewB}")
            print(f"{'='*60}")
            
            print(f"\n--- Processing View A: {viewA} ---")
            pointsA = get_transformed_points(viewA)
            
            print(f"\n--- Processing View B: {viewB} ---")
            pointsB = get_transformed_points(viewB)

            print(f"\n📊 SUMMARY:")
            print(f"   View A ({viewA}): {len(pointsA)} transformed points")
            print(f"   View B ({viewB}): {len(pointsB)} transformed points")
            
            # Use the matcher to find correspondences
            print(f"\n🔗 Matching points between viewA {viewA} and viewB {viewB}...")
            matches = self.matcher.match(pointsA, pointsB)
            
            filtered_matches = matches  # matches are already filtered by RANSAC
            
            return [(viewA, viewB, m[0], m[1]) for m in filtered_matches]
        
        except Exception as e:
            print(f"❌ ERROR: Failed in _process_matching_task for pair {pair}")
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
    
    # Configure the logging options
    logging_config = {
        'save_logs_to_view_folder': True,  # Create a 'logs' folder with a subfolder for each view "(18, 1)" and save logs there
        'basis_points_details': 3,          # Show details for first 3 basis points
        'detailed_descriptor_breakdown': False,  # Turn off detailed descriptor calculations
        'interest_point_transformation': 10,     # Show only first 10 point transformations
        'ratio_test_output': 15                   # Show only first 15 ratio test calculations
    }
    
    pipeline = MatchingPipeline(xml_file, interest_points_folder, logging=logging_config)
    pipeline.run()

if __name__ == "__main__":
    xml_file = "/home/martin/Documents/Allen/Data/IP_TIFF_XML_2/dataset.xml"
    main(xml_file)
