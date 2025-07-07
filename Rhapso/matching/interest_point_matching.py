from Rhapso.pipelines.matching.matching_pipeline import run_matching_pipeline
from Rhapso.matching.matcher import Matcher

class InterestPointMatching:
    """
    Class-based interface for interest point matching pipeline.
    Provides a clean interface similar to InterestPointDetection.
    """
    
    def __init__(self, xml_input_path, n5_output_path, 
                 label='beads', transformation_model='AFFINE', regularization_model='RIGID',
                 lambda_val=0.1, views_to_match='OVERLAPPING_ONLY', clear_correspondences=True,
                 matching_method='PRECISE_TRANSLATION', significance=3.0, redundancy=1,
                 neighboring_points=3, ransac_iterations=200, ransac_minimum_inlier_ratio=0.1,
                 ransac_minimum_inlier_factor=3.0, ransac_threshold=5.0,
                 save_logs_to_view_folder=False, interest_point_transformation=False,
                 detailed_descriptor_breakdown=10, lowes_ratio_test_output=10,
                 lowes_ratio_test_details=True):

        self.xml_input_path = xml_input_path
        self.n5_output_path = n5_output_path
        self.label = label
        self.transformation_model = transformation_model
        self.regularization_model = regularization_model
        self.lambda_val = lambda_val
        self.views_to_match = views_to_match
        self.clear_correspondences = clear_correspondences
        self.matching_method = matching_method
        self.significance = significance
        self.redundancy = redundancy
        self.neighboring_points = neighboring_points
        self.ransac_iterations = ransac_iterations
        self.ransac_minimum_inlier_ratio = ransac_minimum_inlier_ratio
        self.ransac_minimum_inlier_factor = ransac_minimum_inlier_factor
        self.ransac_threshold = ransac_threshold
        
        # Convert string model names to enum values
        if isinstance(transformation_model, str):
            self.transform_model = getattr(Matcher.TransformationModel, transformation_model.upper())
        else:
            self.transform_model = transformation_model
            
        if isinstance(regularization_model, str):
            self.reg_model = getattr(Matcher.RegularizationModel, regularization_model.upper())
        else:
            self.reg_model = regularization_model
        
        # RANSAC parameters
        self.ransac_params = {
            'iterations': ransac_iterations,
            'threshold': ransac_threshold,
            'minimum_inlier_ratio': ransac_minimum_inlier_ratio,
            'minimum_inlier_factor': ransac_minimum_inlier_factor
        }
        
        # Logging configuration
        self.logging_config = {
            'save_logs_to_view_folder': save_logs_to_view_folder,
            'interest_point_transformation': interest_point_transformation,
            'detailed_descriptor_breakdown': detailed_descriptor_breakdown,
            'lowes_ratio_test_output': lowes_ratio_test_output,
            'lowes_ratio_test_details': lowes_ratio_test_details,
        }
    
    def matching(self):
        """
        Run the interest point matching pipeline.
        """
        # Print all configuration values
        print("=" * 60)
        print("INTEREST POINT MATCHING CONFIGURATION")
        print("=" * 60)
        print(f"XML Input Path: {self.xml_input_path}")
        print(f"N5 Output Path: {self.n5_output_path}")
        print(f"Label: {self.label}")
        print(f"Transformation Model: {self.transformation_model}")
        print(f"Regularization Model: {self.regularization_model}")
        print(f"Lambda: {self.lambda_val}")
        print(f"Views to Match: {self.views_to_match}")
        print(f"Clear Correspondences: {self.clear_correspondences}")
        print(f"Matching Method: {self.matching_method}")
        print(f"Significance: {self.significance}")
        print(f"Redundancy: {self.redundancy}")
        print(f"Neighboring Points: {self.neighboring_points}")
        print(f"RANSAC Iterations: {self.ransac_iterations}")
        print(f"RANSAC Minimum Inlier Ratio: {self.ransac_minimum_inlier_ratio}")
        print(f"RANSAC Minimum Inlier Factor: {self.ransac_minimum_inlier_factor}")
        print(f"RANSAC Threshold: {self.ransac_threshold}")
        print("=" * 60)
        
        success = run_matching_pipeline(
            xml_input_path=self.xml_input_path,
            n5_output_path=self.n5_output_path,
            transform_model=self.transform_model,
            reg_model=self.reg_model,
            lambda_val=self.lambda_val,
            ransac_params=self.ransac_params,
            logging_config=self.logging_config,
            label=self.label,
            views_to_match=self.views_to_match,
            clear_correspondences=self.clear_correspondences,
            matching_method=self.matching_method,
            significance=self.significance,
            redundancy=self.redundancy,
            neighboring_points=self.neighboring_points
        )
        return success
    
    def run(self):
        """
        Run the interest point matching pipeline.
        """
        return self.matching()
    
    @classmethod
    def from_config(cls, config, xml_input_key='xml_file_path_matching', output_path_key='n5_matching_output_path'):
        """
        Create an InterestPointMatching instance from a config dictionary.
        
        Args:
            config (dict): Configuration dictionary
            xml_input_key (str): Key for XML input path in config
            output_path_key (str): Key for output path in config
        
        Returns:
            InterestPointMatching: Configured instance
        """
        return cls(
            xml_input_path=config[xml_input_key],
            n5_output_path=config[output_path_key],
            label=config.get('label', 'beads'),
            transformation_model=config.get('transformation_model', 'AFFINE'),
            regularization_model=config.get('regularization_model', 'RIGID'),
            lambda_val=config.get('lambda_val', 0.1),
            views_to_match=config.get('views_to_match', 'OVERLAPPING_ONLY'),
            clear_correspondences=config.get('clear_correspondences', True),
            matching_method=config.get('matching_method', 'PRECISE_TRANSLATION'),
            significance=config.get('significance', 3.0),
            redundancy=config.get('redundancy', 1),
            neighboring_points=config.get('neighboring_points', 3),
            ransac_iterations=config.get('ransac_iterations', 200),
            ransac_minimum_inlier_ratio=config.get('ransac_minimum_inlier_ratio', 0.1),
            ransac_minimum_inlier_factor=config.get('ransac_minimum_inlier_factor', 3.0),
            ransac_threshold=config.get('ransac_threshold', 5.0)
        )
