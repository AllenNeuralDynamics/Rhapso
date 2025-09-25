from Rhapso.pipelines.ray.interest_point_detection import InterestPointDetection
from Rhapso.pipelines.ray.interest_point_matching import InterestPointMatching
from Rhapso.pipelines.ray.solver import Solver
from Rhapso.pipelines.ray.split_dataset import SplitDataset
import yaml
import ray

# Initialize Ray
ray.init()

# Point to param file
with open("Rhapso/pipelines/ray/param/dev/zarr_s3_sean.yml", "r") as file:
    config = yaml.safe_load(file)

# -- INITIALIZE EACH COMPONENT --

# INTEREST POINT DETECTION
interest_point_detection = InterestPointDetection(
    dsxy=config['dsxy'],
    dsz=config['dsz'],
    min_intensity=config['min_intensity'],
    max_intensity=config['max_intensity'],
    sigma=config['sigma'],
    threshold=config['threshold'],
    file_type=config['file_type'],
    xml_file_path=config['xml_file_path_detection'],
    image_file_prefix=config['image_file_prefix'],
    xml_output_file_path=config['xml_output_file_path'],
    n5_output_file_prefix=config['n5_output_file_prefix'],
    combine_distance=config['combine_distance'],
    chunks_per_bound=config['chunks_per_bound'],
    run_type=config['detection_run_type'],
    max_spots=config['max_spots'],
    median_filter=config['median_filter'],
)

# INTEREST POINT MATCHING RIGID
interest_point_matching_rigid = InterestPointMatching(
    xml_input_path=config['xml_file_path_matching_rigid'],
    n5_output_path=config['n5_matching_output_path'],
    input_type = config['input_type'],
    match_type=config['match_type_rigid'],
    num_neighbors=config['num_neighbors_rigid'],
    redundancy=config['redundancy_rigid'],
    significance=config['significance_rigid'],
    search_radius=config['search_radius_rigid'],
    num_required_neighbors=config['num_required_neighbors_rigid'],
    model_min_matches=config['model_min_matches_rigid'],
    inlier_factor=config['inlier_factor_rigid'],
    lambda_value=config['lambda_value_rigid'],
    num_iterations=config['num_iterations_rigid'],
    regularization_weight=config['regularization_weight_rigid'],
    image_file_prefix=config['image_file_prefix'],
)             

# INTEREST POINT MATCHING AFFINE
interest_point_matching_affine = InterestPointMatching(
    xml_input_path=config['xml_file_path_matching_affine'],
    n5_output_path=config['n5_matching_output_path'],
    input_type = config['input_type'],
    match_type=config['match_type_affine'],
    num_neighbors=config['num_neighbors_affine'],
    redundancy=config['redundancy_affine'],
    significance=config['significance_affine'],
    search_radius=config['search_radius_affine'],
    num_required_neighbors=config['num_required_neighbors_affine'],
    model_min_matches=config['model_min_matches_affine'],
    inlier_factor=config['inlier_factor_affine'],
    lambda_value=config['lambda_value_affine'],
    num_iterations=config['num_iterations_affine'],
    regularization_weight=config['regularization_weight_affine'],
    image_file_prefix=config['image_file_prefix'],
)

# INTEREST POINT MATCHING SPLIT AFFINE
interest_point_matching_split_affine = InterestPointMatching(
    xml_input_path=config['xml_file_path_matching_split_affine'],
    n5_output_path=config['n5_matching_output_path'],
    input_type = config['input_type'],
    match_type=config['match_type_split_affine'],
    num_neighbors=config['num_neighbors_split_affine'],
    redundancy=config['redundancy_split_affine'],
    significance=config['significance_split_affine'],
    search_radius=config['search_radius_split_affine'],
    num_required_neighbors=config['num_required_neighbors_split_affine'],
    model_min_matches=config['model_min_matches_split_affine'],
    inlier_factor=config['inlier_factor_split_affine'],
    lambda_value=config['lambda_value_split_affine'],
    num_iterations=config['num_iterations_split_affine'],
    regularization_weight=config['regularization_weight_split_affine'],
    image_file_prefix=config['image_file_prefix'],
)

# SOLVER RIGID
solver_rigid = Solver(
    xml_file_path_output=config['xml_file_path_output_rigid'],
    n5_input_path=config['n5_input_path'],
    xml_file_path=config['xml_file_path_solver_rigid'],
    run_type=config['run_type_solver_rigid'],   
    relative_threshold=config['relative_threshold'],
    absolute_threshold=config['absolute_threshold'],
    min_matches=config['min_matches'],
    damp=config['damp'],
    max_iterations=config['max_iterations'],
    max_allowed_error=config['max_allowed_error'],
    max_plateauwidth=config['max_plateauwidth'],
    metrics_output_path=config['metrics_output_path'],
    fixed_tile=config['fixed_tile']
)

# SOLVER AFFINE
solver_affine = Solver(
    xml_file_path_output=config['xml_file_path_output_affine'],
    n5_input_path=config['n5_input_path'],
    xml_file_path=config['xml_file_path_solver_affine'],
    run_type=config['run_type_solver_affine'],  
    relative_threshold=config['relative_threshold'],
    absolute_threshold=config['absolute_threshold'],
    min_matches=config['min_matches'],
    damp=config['damp'],
    max_iterations=config['max_iterations'],
    max_allowed_error=config['max_allowed_error'],
    max_plateauwidth=config['max_plateauwidth'],
    metrics_output_path=config['metrics_output_path'],
    fixed_tile=config['fixed_tile']
)

# SOLVER SPLIT AFFINE
solver_split_affine = Solver(
    xml_file_path_output=config['xml_file_path_output_split_affine'],
    n5_input_path=config['n5_input_path'],
    xml_file_path=config['xml_file_path_solver_split_affine'],
    run_type=config['run_type_solver_split_affine'],  
    relative_threshold=config['relative_threshold'],
    absolute_threshold=config['absolute_threshold'],
    min_matches=config['min_matches'],
    damp=config['damp'],
    max_iterations=config['max_iterations'],
    max_allowed_error=config['max_allowed_error'],
    max_plateauwidth=config['max_plateauwidth'],
    metrics_output_path=config['metrics_output_path'],
    fixed_tile=config['fixed_tile']
)

# SPLIT DATASETS
split_dataset = SplitDataset(
    xml_file_path=config['xml_file_path_split'],
    xml_output_file_path=config['xml_output_file_path_split'],
    n5_path=config['n5_path_split'],
    point_density=config['point_density'],
    min_points=config['min_points'],
    max_points=config['max_points'],
    error=config['error'],
    exclude_radius=config['exclude_radius'], 
    target_image_size=config['target_image_size'],
    target_overlap=config['target_overlap'],
)

# -- ALIGNMENT PIPELINE --
# interest_point_detection.run()
# interest_point_matching_rigid.run()
solver_rigid.run()
# interest_point_matching_affine.run()
# solver_affine.run()
# split_dataset.run()
# interest_point_matching_split_affine.run()
# solver_split_affine.run()