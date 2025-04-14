from Rhapso.detection.interest_point_detection import InterestPointDetection
from Rhapso.matching.interest_point_matching import start_matching
from Rhapso.solver.solver import Solver
import yaml

with open('Rhapso/pipelines/param_configs/tiff_local.yml', 'r') as file:
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
    file_type=config['file_type'],
    file_source=config['file_source_detection'],
    xml_file_path=config['xml_file_path_detection'],
    xml_bucket_name=config['xml_bucket_name_detection'],
    image_file_path=config['image_file_path'],
    image_bucket_name=config['image_bucket_name'],
    output_file_path=config['output_file_path'],
    output_bucket_name=config['output_bucket_name']
)
interest_point_detection.run()

# INTEREST POINT MATCHING
xml_input_file = config['xml_input_file']
n5_base_path = config['n5_base_path']
output_path = config['output_path']
start_matching(xml_input_file, n5_base_path, output_path)

# SOLVER
solver = Solver(
    file_source=config['file_source_solver'],
    xml_file_path_output=config['xml_file_path_output'],
    xml_bucket_name=config['xml_bucket_name_solver'],
    xml_file_path=config['xml_file_path_solver'],
    data_prefix=config['data_prefix'],
    fixed_views=config['fixed_views'],
    model=config['model'],
    alignment_option=config['alignment_option'],
    relative_threshold=config['relative_threshold'],
    absolute_threshold=config['absolute_threshold'],
    min_matches=config['min_matches'],
    damp=config['damp'],
    max_iterations=config['max_iterations'],
    max_allowed_error=config['max_allowed_error'],
    max_plateauwidth=config['max_plateauwidth']
)
solver.run()