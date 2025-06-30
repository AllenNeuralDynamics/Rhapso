# -----------------------------------------------------------------------------
# How to run this pipeline script:
# 
#   1. Create virtual env: `python -m venv myNewEnv`
#   2. Activate env: `source myNewEnv/bin/activate`
#   3. Install dependencies from repo root: `pip install .[matching]` and Rhapso `pip install -e .` 
#   4. Change input vars xml_input_path and n5_output_path 
#   5. Run script: `python Rhapso/pipelines/matching/matching_pipeline.py`
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

# Default matching configuration - used when running this file standalone
matching_config = {
    'xml_file_path_matching': '/home/martin/Documents/allen/clean-bss-mvr/IP_TIFF_XML_2/dataset.xml',
    'n5_matching_output_path': '/home/martin/Documents/allen/clean-bss-mvr/IP_TIFF_XML_2/output/interestpoints.n5',
    'label': 'beads',
    'transformation_model': 'AFFINE',
    'regularization_model': 'RIGID',
    'lambda_val': 0.1,
    'views_to_match': 'OVERLAPPING_ONLY',
    'clear_correspondences': True,
    'matching_method': 'FAST_ROTATION',
    'significance': 3.0,
    'redundancy': 1,
    'neighboring_points': 3,
    'ransac_iterations': 200,
    'ransac_minimum_inlier_ratio': 0.1,
    'ransac_minimum_inlier_factor': 3.0,
    'ransac_threshold': 5.0
}

# Legacy paths for backward compatibility (when xml_input_path and n5_output_path are used directly)
xml_input_path = matching_config['xml_file_path_matching']
n5_output_path = matching_config['n5_matching_output_path']

# Configure the output logging options
logging_config = {
    # Enable saving logs to view-specific folders
    'save_logs_to_view_folder': False,  

    # Show interest point transformation details
    'interest_point_transformation': False,  

    # Show more detailed descriptor creation basis point details
    'detailed_descriptor_breakdown': False, 

    # Show Lowe's Ratio Test details
    'lowes_ratio_test_output': False,

    # Show additional lowes ratio test details
    'lowes_ratio_test_details': False,          
}

# Configure RANSAC parameters (legacy - now included in matching_config)
ransac_config = {
    'iterations': matching_config['ransac_iterations'],
    'threshold': matching_config['ransac_threshold']
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

def run_matching_with_config(config, logging_config=None):
    """
    Run the matching pipeline using a comprehensive configuration dictionary.
    
    Args:
        config (dict): Complete matching configuration dictionary
        logging_config (dict): Optional logging configuration
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("=" * 60)
    print("RUNNING MATCHING PIPELINE WITH CONFIG")
    print("=" * 60)
    print("Configuration values:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Extract RANSAC parameters from config
    ransac_params = {
        'iterations': config['ransac_iterations'],
        'threshold': config['ransac_threshold'],
        'minimum_inlier_ratio': config['ransac_minimum_inlier_ratio'],
        'minimum_inlier_factor': config['ransac_minimum_inlier_factor']
    }
    
    # Convert string model names to enum values if needed
    if isinstance(config['transformation_model'], str):
        transform_model = getattr(Matcher.TransformationModel, config['transformation_model'].upper())
    else:
        transform_model = config['transformation_model']
        
    if isinstance(config['regularization_model'], str):
        reg_model = getattr(Matcher.RegularizationModel, config['regularization_model'].upper())
    else:
        reg_model = config['regularization_model']
    
    return run_matching_pipeline(
        xml_input_path=config['xml_file_path_matching'],
        n5_output_path=config['n5_matching_output_path'],
        transform_model=transform_model,
        reg_model=reg_model,
        lambda_val=config['lambda_val'],
        ransac_params=ransac_params,
        logging_config=logging_config,
        label=config['label'],
        views_to_match=config['views_to_match'],
        clear_correspondences=config['clear_correspondences'],
        matching_method=config['matching_method'],
        significance=config['significance'],
        redundancy=config['redundancy'],
        neighboring_points=config['neighboring_points']
    )

def run_matching_pipeline(xml_input_path, n5_output_path, 
                          transform_model=None, reg_model=None, lambda_val=0.1,
                          ransac_params=None, logging_config=None,
                          label='beads', views_to_match='OVERLAPPING_ONLY',
                          clear_correspondences=True, matching_method='PRECISE_TRANSLATION',
                          significance=3.0, redundancy=1, neighboring_points=3):

    # Use default configurations if not provided
    if ransac_params is None:
        ransac_params = {
            'iterations': 200,
            'threshold': 5.0,
            'minimum_inlier_ratio': 0.1,
            'minimum_inlier_factor': 3.0
        }
    
    if logging_config is None:
        logging_config = {
            'save_logs_to_view_folder': False,
            'interest_point_transformation': False,
            'detailed_descriptor_breakdown': False,
            'lowes_ratio_test_output': False,
            'lowes_ratio_test_details': False,
        }
    
    if transform_model is None:
        transform_model = Matcher.TransformationModel.AFFINE
    
    if reg_model is None:
        reg_model = Matcher.RegularizationModel.RIGID

    # Print logging configuration for debugging
    print("=" * 60)
    print("LOGGING CONFIGURATION:")
    print("=" * 60)
    for key, value in logging_config.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    # INTEREST POINT MATCHING
    # --------------------------

    # Get XML content and interest points folder location
    xml_content, interest_points_folder = get_xml_content(xml_input_path)
    if xml_content is None:
        print("Aborting: XML file could not be loaded.")
        return False
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
        transform_model=transform_model,
        reg_model=reg_model,
        lambda_val=lambda_val,
        ransac_params=ransac_params,
        logging=logging_config
    )
    print(f"Matcher initialized with RANSAC iterations: {ransac_params['iterations']}, threshold: {ransac_params['threshold']}")

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
        
        # Run the matching task for the current pair and get results
        task_result = process_matching_task(pair, view_ids_global, view_registrations, data_loader, matcher)

        # Add the results for this pair to the global results list (if any)
        all_results.extend(task_result if task_result else [])

    print(f"Total matches found: {len(all_results)}")

    # Save results to N5 format
    save_n5_results(all_results, n5_output_path, data_global)
    print("Results saved to N5 format")

    print("Interest point matching is done")
    return True

# Main execution block - runs when script is executed directly
if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING MATCHING TEST PIPELINE AS STANDALONE SCRIPT")
    print("=" * 80)
    print("Using the following configuration from matching_config:")
    print("=" * 80)
    
    # Run with comprehensive config when executed as a script
    success = run_matching_with_config(matching_config, logging_config)
