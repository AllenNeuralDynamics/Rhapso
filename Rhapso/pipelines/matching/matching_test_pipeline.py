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
from Rhapso.matching.data_saver import save_correspondences
import os
import sys
import datetime
import io
from contextlib import redirect_stdout
import boto3

class MatchingPipeline:
    def __init__(self, xml_content, interest_points_folder, logging=None, xml_file_path=None, n5_output_path=None):
        # Set default logging configuration if none provided
        self.logging = {
            'detailed_descriptor_breakdown': True,
            'interest_point_transformation': True,
            'ratio_test_output': True,
            'basis_point_details': 3,  # Number of basis points to show details for
            'save_logs_to_view_folder': False,  # Default is not to save logs to view folder
        }
        
        # Store the n5_output_path for later use
        self.n5_output_path = n5_output_path
        
        # Update with user-provided logging settings
        if logging is not None:
            self.logging.update(logging)
        
        # Store XML content and path information    
        self.xml_content = xml_content
        
        # Determine the base path for logs and other operations
        self.xml_file_path = xml_file_path
        
        # If we have a file path, use it to set up log directories
        if xml_file_path:
            self.xml_dir = os.path.dirname(xml_file_path)
        else:
            # If no path provided, use current directory
            self.xml_dir = os.getcwd()
            
        self.logs_dir = os.path.join(self.xml_dir, 'logs')
        
        # Create logs directory if saving logs is enabled
        if self.logging.get('save_logs_to_view_folder', False):
            os.makedirs(self.logs_dir, exist_ok=True)
            print(f"📝 Log directory created at: {self.logs_dir}")
            
        # Initialize parser with XML content rather than file path
        self.parser = XMLParser(xml_content)
        
        # Pass both the interest_points_folder and xml_dir
        self.data_loader = DataLoader(interest_points_folder, logging=self.logging)
        self.data_loader.xml_base_path = self.xml_dir  # Store XML directory
        
        # Create matcher with explicit model parameters using nested enums
        self.matcher = Matcher(
            transform_model=Matcher.TransformationModel.AFFINE,
            reg_model=Matcher.RegularizationModel.RIGID,
            lambda_val=0.1,
            logging=self.logging
        )

    def _get_view_log_path(self, viewA, viewB):
        """Create a log file path for a specific view pair."""
        if not self.logging.get('save_logs_to_view_folder', False):
            return None
            
        # Format views as strings, regardless of their original type
        viewA_str = f"{viewA[0]}_{viewA[1]}" if isinstance(viewA, tuple) else str(viewA)
        viewB_str = f"{viewB[0]}_{viewB[1]}" if isinstance(viewB, tuple) else str(viewB)
        
        # Create a view-specific directory
        view_dir = os.path.join(self.logs_dir, f"view_{viewA_str}_to_{viewB_str}")
        os.makedirs(view_dir, exist_ok=True)
        
        # Generate a timestamped log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(view_dir, f"matching_log_{timestamp}.txt")
        
        return log_file

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

            # Save results to N5 format using the path provided during initialization
            self.save_n5_results(all_results, self.n5_output_path)

            # Matching is now finished!
        except Exception as e:
            print(f"Error during matching pipeline execution: {e}")
            sys.exit(1)

    def _process_matching_task(self, pair, label_map):
        """Process a single matching task"""
        viewA, viewB = pair
        
        # Get log file path if logging to files is enabled
        log_file_path = self._get_view_log_path(viewA, viewB)
        
        # If logging to file, capture all standard output
        captured_output = io.StringIO()
        
        # Use the context manager to redirect stdout to our StringIO object
        try:
            if log_file_path:
                with redirect_stdout(captured_output):
                    result = self._run_matching_task(viewA, viewB, label_map)
                
                # Write the captured output to the log file
                with open(log_file_path, 'w') as log_file:
                    log_file.write(f"=== MATCHING LOG: View {viewA} to {viewB} ===\n")
                    log_file.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    log_file.write(captured_output.getvalue())
                
                # Print a confirmation message about the log file
                print(f"📝 Log saved to: {log_file_path}")
                
                return result
            else:
                # Run normally without redirection if logging is disabled
                return self._run_matching_task(viewA, viewB, label_map)
        except Exception as e:
            print(f"❌ ERROR: Failed in _process_matching_task for pair {pair}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception details: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _run_matching_task(self, viewA, viewB, label_map):
        """Run the actual matching task without any I/O redirection"""
        try:
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
            print(f"❌ ERROR: Failed in _run_matching_task for views {viewA} and {viewB}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception details: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def create_matched_views(self, reference_timepoint, reference_view_setup, reference_label, target_view_setups):
        """
        Create the matched_views list for correspondences saving.
        
        Args:
            reference_timepoint: Reference timepoint ID
            reference_view_setup: Reference view setup ID
            reference_label: Reference label (e.g., "beads")
            target_view_setups: List of target view setup IDs
            
        Returns:
            List of tuples containing (timepoint, viewSetup, label)
        """
        matched_views = [(reference_timepoint, reference_view_setup, reference_label)]
        matched_views.extend([(reference_timepoint, vs, reference_label) for vs in target_view_setups])
        return matched_views

    def save_n5_results(self, all_results, n5_output_path):
        """
        Save the matching results to an N5 file format.
        
        Args:
            all_results: List of all matching results across pairs
            n5_output_path: Path to the output N5 file/directory
        """
        # Gather all unique (timepoint, setup, label) from the dataset
        data_global = self.parser.get_data_global()
        views_interest_points = data_global['viewsInterestPoints']
        matched_views = []
        for (tp, setup), view_info in views_interest_points.items():
            label = view_info.get('label', 'beads')
            matched_views.append((int(tp), int(setup), label))

        # Print detected matched views for debugging
        print(f"Detected matched views: {matched_views}")

        # Save correspondences
        save_correspondences(
            n5_output_path=n5_output_path,
            reference_tp=None,  # Not needed for new logic
            reference_vs=None,
            ref_label=None,
            correspondences=all_results,
            matched_views=matched_views
        )

def get_xml_content(xml_file):
    """
    Fetches XML content from either S3 or local filesystem based on path prefix.
    
    Args:
        xml_file: Path to XML file (local path or s3:// URL)
        
    Returns:
        Tuple containing (XML content as string, interest points folder path)
    """
    # Determine the directory and interest points folder based on path type
    if xml_file.startswith('s3://'):
        # Parse S3 URL components
        s3_path = xml_file[5:]  # Remove 's3://'
        parts = s3_path.split('/', 1)
        bucket_name = parts[0]
        file_key = parts[1]
        
        print(f"Detected S3 path. Fetching from bucket: {bucket_name}, key: {file_key}")
        
        # Initialize S3 client and fetch content
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        xml_content = response["Body"].read().decode("utf-8")
        
        # Create S3 path for interest points folder
        xml_dir = os.path.dirname(file_key)
        interest_points_folder = f"s3://{bucket_name}/{xml_dir}/interestpoints.n5"
    else:
        print(f"Detected local path: {xml_file}")

        def fetch_local_xml(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        

        xml_content = fetch_local_xml(xml_file)
        

        # Create local path for interest points folder
        xml_dir = os.path.dirname(xml_file)
        interest_points_folder = os.path.join(xml_dir, 'interestpoints.n5')
    
    return xml_content, interest_points_folder
        
def main():
    # local input / local output
    xml_input_path = "/home/martin/Documents/allen/IP_TIFF_XML_2/dataset.xml"
    n5_output_path = '/home/martin/Documents/allen/IP_TIFF_XML_2/n5out/'
    
    # s3 input / s3 output
    # xml_input_path = 's3://martin-test-bucket/output/dataset-detection.xml'
    # n5_output_path = 's3://martin-test-bucket/matching_output/'
    
    # Get XML content and interest points folder location
    xml_content, interest_points_folder = get_xml_content(xml_input_path)
    
    print(f"Derived interest points input n5 folder location: {interest_points_folder}")
    
    # Configure the logging options
    logging_config = {
        'save_logs_to_view_folder': False,  # Enable saving logs to view-specific folders
        'basis_points_details': False,          # Show details for first 3 basis points
        'detailed_descriptor_breakdown': False,  # Turn off detailed descriptor calculations
        'interest_point_transformation': False,     # Show only first 10 point transformations
        'ratio_test_output': False                   # Show only first 15 ratio test calculations
    }
    
    # Pass both the XML content and the original file path
    pipeline = MatchingPipeline(
        xml_content=xml_content, 
        interest_points_folder=interest_points_folder,
        logging=logging_config, 
        xml_file_path=xml_input_path,
        n5_output_path=n5_output_path  # Pass the n5_output_path to the pipeline
    )
    pipeline.run()

if __name__ == "__main__":
    main()
