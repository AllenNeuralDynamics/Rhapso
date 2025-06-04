'''
Run with the following commands:
    python -m venv myNewEnv
    source myNewEnv/bin/activate
    pip install .[matching]
    python Rhapso/pipelines/matching/matching_test_pipeline.py 
'''

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

    def _save_interest_points_to_log(self, view_id, raw_points, transformed_points, view_data):
        """Save interest points before and after transformation to a log file"""
        import datetime
        
        # Create log filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"interest_points_log_{timestamp}.log"
        log_path = os.path.join(os.path.dirname(self.xml_file), log_filename)
        
        # Get view information
        view_info = view_data[view_id]
        timepoint, setup_id = view_id
        loc_path = f"{view_info['path']}/{view_info['label']}/interestpoints/loc"
        full_path = os.path.join(self.data_loader.base_path, loc_path)
        
        # Write to log file
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"View ID: {view_id}\n")
            f.write(f"Timepoint: {timepoint}\n")
            f.write(f"Setup ID: {setup_id}\n")
            f.write(f"Label: {view_info['label']}\n")
            f.write(f"Local N5 Path: {loc_path}\n")
            f.write(f"Full Path: {full_path}\n")
            f.write(f"Total Points: {len(raw_points)}\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n")
            f.write("INTEREST POINTS TRANSFORMATION:\n")
            f.write("BEFORE → AFTER\n")
            f.write("-" * 80 + "\n")
            
            # Log sample points (first 10)
            num_samples = min(10, len(raw_points))
            for i in range(num_samples):
                before = [f"{x:.2f}" for x in raw_points[i]]
                after = [f"{x:.2f}" for x in transformed_points[i]]
                f.write(f"Point {i}: [{', '.join(before)}] → [{', '.join(after)}]\n")
            
            if len(raw_points) > num_samples:
                f.write(f"... and {len(raw_points) - num_samples} more points\n")
            
            f.write("\n")
        
        print(f"📝 Interest points logged to: {log_path}")

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
                transformed_points = self.data_loader.transform_interest_points(raw_points, transform)
                
                # Save to log file
                self._save_interest_points_to_log(view_id, raw_points, transformed_points, view_data)
                
                return transformed_points

            # Retrieve and transform interest points for both views
            pointsA = get_transformed_points(viewA)
            pointsB = get_transformed_points(viewB)

            print(f"Loaded {len(pointsA)} transformed points for viewA: {viewA}")
            print(f"Loaded {len(pointsB)} transformed points for viewB: {viewB}")
            
            # Call method to save interest points before and after to local .log file like so:
                # Point 0: [1007.65, 45.47, 0.95] → [313.65, -422.89, 248.20]
                # Point 1: [183.95, 271.82, 1.14] → [-510.05, -262.70, 88.28]
                # Point 2: [550.95, 282.40, 0.51] → [-143.05, -255.66, 80.35]
            # make sure we label eahc view, time point, id and local path where the interest points were loaded from

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
