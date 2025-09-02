from Rhapso.CLI.eval import MetricReviewCLI
from Rhapso.pipelines.accuracy_metrics.matching_stats_pipeline import StatsPipeline
from Rhapso.pipelines.ray.interest_point_detection import InterestPointDetection
from Rhapso.pipelines.ray.interest_point_matching import InterestPointMatching
from Rhapso.pipelines.ray.solver import Solver
import yaml
import ray

ray.init()

with open("Rhapso/pipelines/ray/param/exaSPIM_720164.yml", "r") as file:
    config = yaml.safe_load(file)

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
    median_filter=config['median_filter']
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
)

# SOLVER RIGID
solver_rigid = Solver(
    xml_file_path_output=config['xml_file_path_output_rigid'],
    n5_input_path=config['n5_input_path'],
    xml_file_path=config['xml_file_path_solver_rigid'],
    metrics_output_path=config['metrics_output_path'],
    fixed_views=config['fixed_views'],
    run_type=config['run_type_solver_rigid'],   
    relative_threshold=config['relative_threshold'],
    absolute_threshold=config['absolute_threshold'],
    min_matches=config['min_matches'],
    damp=config['damp'],
    max_iterations=config['max_iterations'],
    max_allowed_error=config['max_allowed_error'],
    max_plateauwidth=config['max_plateauwidth'],
)

# SOLVER AFFINE
solver_affine = Solver(
    xml_file_path_output=config['xml_file_path_output_affine'],
    n5_input_path=config['n5_input_path'],
    xml_file_path=config['xml_file_path_solver_rigid'],
    metrics_output_path=config['metrics_output_path'],
    fixed_views=config['fixed_views'],
    run_type=config['run_type_solver_affine'],  
    relative_threshold=config['relative_threshold'],
    absolute_threshold=config['absolute_threshold'],
    min_matches=config['min_matches'],
    damp=config['damp'],
    max_iterations=config['max_iterations'],
    max_allowed_error=config['max_allowed_error'],
    max_plateauwidth=config['max_plateauwidth'],
)

eval_stats = StatsPipeline(
    args = config['args'],
    xml_file = config['xml_file'],
    base_path = config['base_path'],
    metrics_output_path = config['metrics_output_path'],
    KDE_type = config['KDE_type'],
    bandwidth = config['bandwidth'], 
    view_id = config['view_id'], 
    pair = config['pair'], 
    plot = config['plot'],
    thresholding= config['thresholding'],
    min_alignment= config['min_alignment'],
    max_alignment= config['max_alignment'], 
    minimum_points= config['minimum_points'], 
    maximum_points= config['maximum_points'], 
    minimum_total_matches= config['minimum_total_matches'], 
    maximum_total_matches= config['maximum_total_matches'], 
    max_kde= config['max_kde'],
    min_kde= config['min_kde'], 
    max_cv= config['max_cv'],
    min_cv= config['min_cv']
    )

evaluation = MetricReviewCLI(
    file_path=config['file_path'],
    matching_affine=interest_point_matching_affine,
    solve_affine=solver_affine, 
    matching_rigid=interest_point_matching_rigid, 
    solve_rigid=solver_rigid)


interest_point_detection.run()
interest_point_matching_rigid.run()
solver_rigid.run()
interest_point_matching_affine.run()
solver_affine.run()
eval_stats.run()
# Interactive Evaluation
evaluation.run()

