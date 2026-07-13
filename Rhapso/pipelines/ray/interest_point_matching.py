from Rhapso.matching.xml_parser import XMLParserMatching
from Rhapso.matching.generate_pairs import GeneratePairs
from Rhapso.matching.load_and_transform_points import LoadAndTransformPoints
from Rhapso.matching.ransac_matching import RansacMatching
from Rhapso.matching.save_matches import SaveMatches
import ray

class InterestPointMatching:
    def __init__(self, xml_input_path, n5_output_path, input_type, match_type, num_neighbors, redundancy, significance, 
                 search_radius, num_required_neighbors, model_min_inliers, ransac_sample_size, inlier_threshold, min_inlier_ratio, 
                 num_iterations, regularization_weight, image_file_prefix):
        self.xml_input_path = xml_input_path
        self.n5_output_path = n5_output_path
        self.input_type = input_type
        self.match_type = match_type              
        self.num_neighbors = num_neighbors
        self.redundancy = redundancy
        self.significance = significance                 
        self.search_radius = search_radius
        self.num_required_neighbors = num_required_neighbors
        self.model_min_inliers = model_min_inliers  
        self.ransac_sample_size = ransac_sample_size      
        self.inlier_threshold = inlier_threshold          
        self.min_inlier_ratio = min_inlier_ratio               
        self.num_iterations = num_iterations
        self.regularization_weight = regularization_weight
        self.image_file_prefix = image_file_prefix

    def match(self):
        print("Starting Interest Point Matching")

        # Load XML
        parser = XMLParserMatching(self.xml_input_path, self.input_type)
        data_global = parser.run()

        # Generate matching pairs
        generate_pairs = GeneratePairs(data_global, self.match_type)
        process_pairs, view_registrations = generate_pairs.run()

        # Distribute interest point matching with Ray
        @ray.remote
        def match_pair(viewA, viewB, viewA_str, viewB_str, label, num_neighbors, redundancy, significance, num_required_neighbors, 
                       match_type, inlier_threshold, min_inlier_ratio, num_iterations, model_min_inliers, regularization_weight, search_radius,
                       view_registrations, input_type, image_file_prefix, ransac_sample_size, n5_output_path): 
            
            points_loader = LoadAndTransformPoints(data_global, view_registrations, label, n5_output_path)
            matcher = RansacMatching(data_global, num_neighbors, redundancy, significance, num_required_neighbors, match_type, inlier_threshold, 
                                     min_inlier_ratio, num_iterations, model_min_inliers, regularization_weight, search_radius, view_registrations,
                                     input_type, image_file_prefix, ransac_sample_size)

            pointsA, pointsB, viewA_str, viewB_str = points_loader.run(viewA, viewB)
            pointsA, pointsB = matcher.filter_for_overlapping_points(pointsA, pointsB, viewA_str, viewB_str)

            if len(pointsA) == 0 or len(pointsB) == 0:
                return []
            
            candidates = matcher.get_candidates(pointsA, pointsB, viewA_str, viewB_str, label)
            inliers, regularized_model = matcher.compute_ransac(candidates)
            filtered_inliers = matcher.filter_inliers(inliers, regularized_model)

            percent = 100.0 * len(filtered_inliers) / len(candidates) if candidates else 0
            print(f"✅ RANSAC inlier percentage: {percent:.1f}% ({len(filtered_inliers)} of {len(candidates)} for {viewA_str}), {viewB_str}")

            if len(filtered_inliers) < model_min_inliers:
                return []

            return filtered_inliers if filtered_inliers else []

        # --- Distribute ---
        futures = [
            match_pair.remote(viewA, viewB, viewA_str, viewB_str, label, self.num_neighbors, self.redundancy, self.significance, self.num_required_neighbors,
                            self.match_type, self.inlier_threshold, self.min_inlier_ratio, self.num_iterations, self.model_min_inliers, self.regularization_weight, 
                            self.search_radius, view_registrations, self.input_type, self.image_file_prefix, self.ransac_sample_size, self.n5_output_path)
            for viewA, viewB, viewA_str, viewB_str, label in process_pairs
        ]

        # --- Collect ---
        results = ray.get(futures)
        all_results = [inlier for sublist in results for inlier in sublist]

        # --- Save ---
        saver = SaveMatches(all_results, self.n5_output_path, data_global, self.match_type)
        saver.run()
        print("Interest Point Matching is Done")
    
    def run(self):
        self.match()

# DEBUG MATCHING
# all_results = []
# for viewA, viewB, viewA_str, viewB_str, label in process_pairs:

#     points_loader = LoadAndTransformPoints(data_global, view_registrations, label, self.n5_output_path)
#     matcher = RansacMatching(data_global, self.num_neighbors, self.redundancy, self.significance, self.num_required_neighbors, self.match_type, self.inlier_threshold, 
#                                 self.min_inlier_ratio, self.num_iterations, self.model_min_inliers, self.regularization_weight, self.search_radius, view_registrations,
#                                 self.input_type, self.image_file_prefix, self.ransac_sample_size)
    
#     pointsA, pointsB, viewA_str, viewB_str = points_loader.run(viewA, viewB)
#     pointsA, pointsB = matcher.filter_for_overlapping_points(pointsA, pointsB, viewA_str, viewB_str)

#     if len(pointsA) == 0 or len(pointsB) == 0:
#         continue
    
#     candidates = matcher.get_candidates(pointsA, pointsB, viewA_str, viewB_str, label)
#     inliers, regularized_model = matcher.compute_ransac(candidates)
#     filtered_inliers = matcher.filter_inliers(inliers, regularized_model)

#     percent = 100.0 * len(filtered_inliers) / len(candidates) if candidates else 0
#     print(f"✅ RANSAC inlier percentage: {percent:.1f}% ({len(filtered_inliers)} of {len(candidates)} for {viewA_str}), {viewB_str}")

#     if len(filtered_inliers) < self.model_min_inliers:
#         continue

#     all_results.append(filtered_inliers)
