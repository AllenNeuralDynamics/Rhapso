from Rhapso.matching.interest_point_matching import InterestPointMatching
import yaml

with open('Rhapso/pipelines/param_configs/tiff_local.yml', 'r') as file:
    config = yaml.safe_load(file)

# INTEREST POINT MATCHING
interest_point_matching = InterestPointMatching(
    xml_input_path=config['xml_file_path_matching'],
    n5_output_path=config['n5_matching_output_path'],
    label=config['label'],
    transformation_model=config['transformation_model'],
    regularization_model=config['regularization_model'],
    lambda_val=config['lambda_val'],
    views_to_match=config['views_to_match'],
    clear_correspondences=config['clear_correspondences'],
    matching_method=config['matching_method'],
    significance=config['significance'],
    redundancy=config['redundancy'],
    neighboring_points=config['neighboring_points'],
    ransac_iterations=config['ransac_iterations'],
    ransac_minimum_inlier_ratio=config['ransac_minimum_inlier_ratio'],
    ransac_minimum_inlier_factor=config['ransac_minimum_inlier_factor'],
    ransac_threshold=config['ransac_threshold']
)
interest_point_matching.run()