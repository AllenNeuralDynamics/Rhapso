from Rhapso.matching.xml_parser import XMLParserMatching
from Rhapso.matching.load_and_transform_points import LoadAndTransformPoints
from Rhapso.matching.ransac_matching import RansacMatching
from Rhapso.matching.save_matches import SaveMatches
import ray

class InterestPointMatching:
    def __init__(self, xml_input_path, n5_output_path, input_type, match_type, num_neighbors, redundancy, significance, 
                 search_radius, num_required_neighbors, model_min_matches, inlier_factor, lambda_value, num_iterations, 
                 regularization_weight, image_file_prefix):
        self.xml_input_path = xml_input_path
        self.n5_output_path = n5_output_path
        self.input_type = input_type
        self.match_type = match_type              
        self.num_neighbors = num_neighbors
        self.redundancy = redundancy
        self.significance = significance                 
        self.search_radius = search_radius
        self.num_required_neighbors = num_required_neighbors
        self.model_min_matches = model_min_matches        
        self.inlier_factor = inlier_factor          
        self.lambda_value = lambda_value               
        self.num_iterations = num_iterations
        self.regularization_weight = regularization_weight
        self.image_file_prefix = image_file_prefix

    def match(self):
        # Load XML
        parser = XMLParserMatching(self.xml_input_path, self.input_type)
        data_global = parser.run()
        print("XML loaded and parsed")

        # Load and transform points
        data_loader = LoadAndTransformPoints(data_global, self.xml_input_path, self.n5_output_path, self.match_type)
        process_pairs, view_registrations = data_loader.run()
        print("Points loaded and transformed into global space")

        # Distribute interest point matching with Ray
        @ray.remote
        def match_pair(pointsA, pointsB, viewA_str, viewB_str, label, num_neighbors, redundancy, significance, num_required_neighbors, 
                       match_type, inlier_factor, lambda_value, num_iterations, model_min_matches, regularization_weight, search_radius,
                       view_registrations, input_type, image_file_prefix): 
            
            matcher = RansacMatching(data_global, num_neighbors, redundancy, significance, num_required_neighbors, match_type, inlier_factor, 
                                     lambda_value, num_iterations, model_min_matches, regularization_weight, search_radius, view_registrations,
                                     input_type, image_file_prefix)
            
            pointsA, pointsB = matcher.filter_for_overlapping_points(pointsA, pointsB, viewA_str, viewB_str)

            if len(pointsA) == 0 or len(pointsB) == 0:
                return []
            
            candidates = matcher.get_candidates(pointsA, pointsB, viewA_str, viewB_str, label)
            inliers, regularized_model = matcher.compute_ransac(candidates)
            filtered_inliers = matcher.filter_inliers(inliers, regularized_model)

            percent = 100.0 * len(filtered_inliers) / len(candidates) if candidates else 0
            print(f"✅ RANSAC inlier percentage: {percent:.1f}% ({len(filtered_inliers)} of {len(candidates)} for {viewA_str}), {viewB_str}")

            if len(filtered_inliers) < self.model_min_matches:
                return []

            return filtered_inliers if filtered_inliers else []

        # --- Distribute ---
        futures = [
            match_pair.remote(pointsA, pointsB, viewA_str, viewB_str, label, self.num_neighbors, self.redundancy, self.significance, self.num_required_neighbors,
                            self.match_type, self.inlier_factor, self.lambda_value, self.num_iterations, self.model_min_matches, self.regularization_weight, 
                            self.search_radius, view_registrations, self.input_type, self.image_file_prefix)
            for pointsA, pointsB, viewA_str, viewB_str, label in process_pairs
        ]

        # --- Collect ---
        results = ray.get(futures)
        all_results = [inlier for sublist in results for inlier in sublist]

        # --- Save ---
        saver = SaveMatches(all_results, self.n5_output_path, data_global, self.match_type)
        saver.run()
        print("Matches Saved as N5")

        print("Interest point matching is done")
    
    def run(self):
        self.match()


# DEBUG MATCHING
# all_results = []
# for pointsA, pointsB, viewA_str, viewB_str, label in process_pairs:
#     matcher = RansacMatching(data_global, self.num_neighbors, self.redundancy, self.significance, self.num_required_neighbors, self.match_type, self.inlier_factor, 
#                              self.lambda_value, self.num_iterations, self.model_min_matches, self.regularization_weight, self.search_radius, view_registrations,
#                              self.input_type, self.image_file_prefix)
    
#     pointsA, pointsB = matcher.filter_for_overlapping_points(pointsA, pointsB, viewA_str, viewB_str)

#     if len(pointsA) == 0 or len(pointsB) == 0:
#         continue
    
#     candidates = matcher.get_candidates(pointsA, pointsB, viewA_str, viewB_str, label)
#     inliers, regularized_model = matcher.compute_ransac(candidates)
#     filtered_inliers = matcher.filter_inliers(inliers, regularized_model)

#     percent = 100.0 * len(filtered_inliers) / len(candidates) if candidates else 0
#     print(f"✅ RANSAC inlier percentage: {percent:.1f}% ({len(filtered_inliers)} of {len(candidates)} for {viewA_str}), {viewB_str}")

#     if len(filtered_inliers) < self.model_min_matches:
#         continue

#     all_results.append(filtered_inliers)