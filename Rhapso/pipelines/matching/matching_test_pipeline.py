# -----------------------------------------------------------------------------
# How to run this pipeline script:
# 
#   1. Create virtual env: `python -m venv myNewEnv`
#   2. Activate env: `source myNewEnv/bin/activate`
#   3. Install dependencies from repo root: `pip install .[matching]` and Rhapso `pip install -e .` 
#   4. Change input vars xml_input_path and n5_output_path 
#   5. Run script: `python Rhapso/pipelines/matching/matching_test_pipeline.py`
# -----------------------------------------------------------------------------

from Rhapso.matching.xml_parser import XMLParser
from Rhapso.matching.data_loader import DataLoader
from Rhapso.matching.matcher import Matcher
from Rhapso.matching.data_saver import save_correspondences
import os
import sys
import datetime
import io
from contextlib import redirect_stdout
import boto3

# Configuration

# local input/output paths:
xml_input_path = "/home/martin/Documents/Allen/Data/IP_TIFF_XML_2/dataset.xml"
n5_output_path = '/home/martin/Documents/Allen/Data/IP_TIFF_XML_2/n5Output6.20'

# s3 input/output paths:
# xml_input_path = 's3://martin-test-bucket/output/dataset-detection.xml'
# n5_output_path = 's3://martin-test-bucket/matching_output/'

# Configure the logging options
logging_config = {
    # Enable saving logs to view-specific folders
    'save_logs_to_view_folder': False,  

    # Show interest point transformation details
    'interest_point_transformation': False,  

    # Show descriptor creation basis point details
    'basis_points_details': False,      

    # Show more detailed descriptor creation basis point details
    'detailed_descriptor_breakdown': False, 

    # Show Lowe's Ratio Test details
    'ratio_test_output': False          
}

# Configure RANSAC parameters
ransac_config = {
    'iterations': 1000,    # Number of RANSAC iterations
    'threshold': 5.0       # Inlier threshold for RANSAC
}

def fetch_from_s3(s3, bucket_name, input_file):
    response = s3.get_object(Bucket=bucket_name, Key=input_file)
    return response["Body"].read().decode("utf-8")

def fetch_local_xml(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"pipeline failed, could not find xml file located at '{file_path}'")
        return None
    except Exception as e:
        print(f"pipeline failed, error while parsing xml file at '{file_path}': {e}")
        return None


def get_xml_content(xml_file):
    """
    Fetches XML content from either S3 or local filesystem based on path prefix.
    Returns (xml_content, interest_points_folder) or (None, None) if not found.
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
        xml_content = fetch_local_xml(xml_file)
        if xml_content is None:
            return None, None
        # Create local path for interest points folder
        xml_dir = os.path.dirname(xml_file)
        interest_points_folder = os.path.join(xml_dir, 'interestpoints.n5')
    
    return xml_content, interest_points_folder

def get_transformed_points(view_id, view_data, view_registrations, data_loader):
    """Retrieve and transform interest points for a given view."""
    # Unpack view_id for clarity
    if isinstance(view_id, tuple) and len(view_id) == 2:
        tpId, setupId = view_id
        view_str = f"(tpId={tpId}, setupId={setupId})"
    else:
        view_str = str(view_id)
    print(f"\n🔍 Processing view {view_str}...")
    
    view_info = view_data[view_id]
    loc_path = f"{view_info['path']}/{view_info['label']}/interestpoints/loc"
    print(f"📁 Loading from path: {loc_path}")
    
    raw_points = data_loader.load_interest_points_from_path(loc_path)
    print(f"📥 Loaded {len(raw_points)} raw interest points")
    
    transform = data_loader.get_transformation_matrix(view_id, view_data, view_registrations)
    print(f"🔄 Applying transformation matrix for view {view_id}:")
    
    transformed_points = data_loader.transform_interest_points(raw_points, transform)
    
    return transformed_points



def save_n5_results(all_results, n5_output_path, data_global):
    """
    Save the matching results to an N5 file format.
    
    Args:
        all_results: List of all matching results across pairs
        n5_output_path: Path to the output N5 file/directory
        data_global: Parsed XML data containing view information
    """
    # Gather all unique (timepoint, setup, label) from the dataset
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

def process_matching_task(pair, view_data, view_registrations, data_loader, matcher):
    """Process a single matching task"""
    viewA, viewB = pair
    
    try:
        # Retrieve and transform interest points for both views
        # Unpack for clarity
        if isinstance(viewA, tuple) and len(viewA) == 2:
            tpA, setupA = viewA
            viewA_str = f"(tpId={tpA}, setupId={setupA})"
        else:
            viewA_str = str(viewA)
        if isinstance(viewB, tuple) and len(viewB) == 2:
            tpB, setupB = viewB
            viewB_str = f"(tpId={tpB}, setupId={setupB})"
        else:
            viewB_str = str(viewB)
        print(f"\n{'='*60}")
        print(f"PROCESSING VIEW PAIR: {viewA_str} ↔ {viewB_str}")
        print(f"{'='*60}")
        
        pointsA = get_transformed_points(viewA, view_data, view_registrations, data_loader)
        pointsB = get_transformed_points(viewB, view_data, view_registrations, data_loader)

        print(f"\n📊 SUMMARY:")
        print(f"   View A {viewA_str}: {len(pointsA)} transformed points")
        print(f"   View B {viewB_str}: {len(pointsB)} transformed points")
        
        # Use the matcher to find correspondences
        print(f"\n🔗 Matching points between viewA {viewA_str} and viewB {viewB_str}...")
        matches = matcher.match(pointsA, pointsB)
        
        filtered_matches = matches  # matches are already filtered by RANSAC
        
        return [(viewA, viewB, m[0], m[1]) for m in filtered_matches]
    except Exception as e:
        print(f"❌ ERROR: Failed in process_matching_task for views {viewA} and {viewB}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception details: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


# INTEREST POINT MATCHING
# --------------------------

# Get XML content and interest points folder location
xml_content, interest_points_folder = get_xml_content(xml_input_path)
if xml_content is None:
    print("Aborting: XML file could not be loaded.")
    sys.exit(1)
print(f"Derived interest points input n5 folder location: {interest_points_folder}")

# Initialize parser with XML content
parser = XMLParser(xml_content)

# Load complete dataset xml information
data_global = parser.parse()
print("XML loaded and parsed")

# Use data_global for all subsequent operations
view_ids_global = data_global['viewsInterestPoints']
view_registrations = data_global['viewRegistrations']

# Set up view groups using complete dataset info
setup = parser.setup_groups(view_registrations)
print("View groups setup completed")

# Initialize data loader
xml_dir = os.path.dirname(xml_input_path) if not xml_input_path.startswith('s3://') else ""
data_loader = DataLoader(interest_points_folder, logging=logging_config)
if xml_dir:
    data_loader.xml_base_path = xml_dir

# Build label map using view IDs only
label_map_global = data_loader.build_label_map(view_ids_global)
print("Label map built")

# Create matcher with explicit model parameters using nested enums
matcher = Matcher(
    transform_model=Matcher.TransformationModel.AFFINE,
    reg_model=Matcher.RegularizationModel.RIGID,
    lambda_val=0.1,
    ransac_params=ransac_config,
    logging=logging_config
)
print(f"Matcher initialized with RANSAC iterations: {ransac_config['iterations']}, threshold: {ransac_config['threshold']}")

# Process all view pairs for matching
all_results = []  # Store all matching results across pairs
total_pairs = len(setup['pairs'])  # Total number of view pairs to process

# Iterate through each pair of views to perform matching
for idx, pair in enumerate(setup['pairs'], 1):
    viewA, viewB = pair  # Unpack the current pair of view IDs

    # Retrieve human-readable labels for each view, fallback to "Unknown" if missing
    labelA = label_map_global.get(viewA, "Unknown")
    labelB = label_map_global.get(viewB, "Unknown")

    # Unpack for clarity
    if isinstance(viewA, tuple) and len(viewA) == 2:
        tpA, setupA = viewA
        viewA_str = f"(tpId={tpA}, setupId={setupA})"
    else:
        viewA_str = str(viewA)
    if isinstance(viewB, tuple) and len(viewB) == 2:
        tpB, setupB = viewB
        viewB_str = f"(tpId={tpB}, setupId={setupB})"
    else:
        viewB_str = str(viewB)
    print(f"Processing pair {idx}/{total_pairs}: {viewA_str}, {viewB_str} with labels ({labelA}, {labelB})")

    # Run the matching task for the current pair and get results
    task_result = process_matching_task(pair, view_ids_global, view_registrations, data_loader, matcher)

    # Add the results for this pair to the global results list (if any)
    all_results.extend(task_result if task_result else [])

print(f"Total matches found: {len(all_results)}")

# Save results to N5 format
save_n5_results(all_results, n5_output_path, data_global)
print("Results saved to N5 format")

print("Interest point matching is done")
