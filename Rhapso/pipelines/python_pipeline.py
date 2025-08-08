from Rhapso.detection.interest_point_detection import InterestPointDetection
from Rhapso.matching.interest_point_matching import InterestPointMatching
from Rhapso.solver.solver import Solver
import yaml

with open("Rhapso/pipelines/param_configs/tiff_local.yml", "r") as file:
    config = yaml.safe_load(file)

# INTEREST POINT DETECTION
interest_point_detection = InterestPointDetection(
    strategy=config['strategy'],
    dsxy=config['dsxy'],
    dsz=config['dsz'],
    min_intensity=config['min_intensity'],
    max_intensity=config['max_intensity'],
    sigma=config['sigma'],
    threshold=config['threshold'],
    offset=config['offset'],
    combine_distance=config['combine_distance'],
    mem_per_worker_bytes=config['mem_per_worker_bytes'], 
    file_type=config['file_type'],
    file_source=config['file_source_detection'],
    xml_file_path=config['xml_file_path_detection'],
    xml_bucket_name=config['xml_bucket_name_detection'],
    image_file_path=config['image_file_path'],
    image_bucket_name=config['image_bucket_name'],
    output_file_path=config['output_file_path'],
    output_bucket_name=config['output_bucket_name'],
    run_type=config['detection_run_type']
)
interest_point_detection.run()

# INTEREST POINT MATCHING
interest_point_matching = InterestPointMatching(
    xml_input_path=config['xml_file_path_matching'],
    n5_output_path=config['n5_matching_output_path'],
    match_type=config['match_type'],
    num_neighbors=config['num_neighbors'],
    redundancy=config['redundancy'],
    significance=config['significance'],
    search_radius=config['search_radius'],
    num_required_neighbors=config['num_required_neighbors'],
    model_min_matches=config['model_min_matches'],
    inlier_factor=config['inlier_factor'],
    lambda_value=config['lambda_value'],
    num_iterations=config['num_iterations'],
    regularization_weight=config['regularization_weight'],
    input_type = config['input_type']
)
interest_point_matching.run()

# SOLVER
solver = Solver(
    file_source=config['file_source_solver'],
    xml_file_path_output=config['xml_file_path_output'],
    xml_bucket_name=config['xml_bucket_name_solver'],
    xml_file_path=config['xml_file_path_solver'],
    data_prefix=config['data_prefix'],
    fixed_views=config['fixed_views'],
    run_type=config['run_type_solver'],
    relative_threshold=config['relative_threshold'],
    absolute_threshold=config['absolute_threshold'],
    min_matches=config['min_matches'],
    damp=config['damp'],
    max_iterations=config['max_iterations'],
    max_allowed_error=config['max_allowed_error'],
    max_plateauwidth=config['max_plateauwidth'],
    metrics_output_path=config['metrics_output_path'],
)
solver.run()