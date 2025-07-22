from Rhapso.matching.xml_parser import XMLParser
from Rhapso.matching.load_and_transform_points import LoadAndTransformPoints
from Rhapso.matching.ransac_matching import RansacMatching
from Rhapso.matching.save_matches import SaveMatches

class InterestPointMatching:
    def __init__(self, xml_input_path, n5_output_path, match_type, num_neighbors, redundancy, significance, search_radius,
                 num_required_neighbors, model_min_matches, inlier_factor, lambda_value, num_iterations, regularization_weight):
        self.xml_input_path = xml_input_path
        self.n5_output_path = n5_output_path
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

    def match(self):
        # Initialize parser with XML content
        parser = XMLParser(self.xml_input_path)
        data_global, _ = parser.run()
        print("XML loaded and parsed")

        # Load interest points and transform them into global space
        data_loader = LoadAndTransformPoints(data_global, self.xml_input_path)
        process_pairs = data_loader.run()
        print("Points loaded and transformed into global space")

        # Geometric Descriptor-Based Interest Point Matching with RANSAC
        matcher = RansacMatching(
            self.num_neighbors, 
            self.redundancy, 
            self.significance, 
            self.num_required_neighbors, 
            self.match_type, 
            self.inlier_factor, 
            self.lambda_value, 
            self.num_iterations, 
            self.model_min_matches, 
            self.regularization_weight,
            self.search_radius
        )

        all_results = []
        for pointsA, pointsB, viewA_str, viewB_str in process_pairs:   
            candidates = matcher.get_candidates(pointsA, pointsB, viewA_str, viewB_str)
            inliers = matcher.compute_ransac(candidates)

            percent = 100.0 * len(inliers) / len(candidates)
            print(f"✅ RANSAC inlier percentage: {percent:.1f}% ({len(inliers)} of {len(candidates)} for {viewA_str}), {viewB_str}")

            # if len(inliers) < 20:
            #     continue

            all_results.extend(inliers if inliers else []) 
        
        print("Matching is done")

        # Save matches as N5
        saver = SaveMatches(all_results, self.n5_output_path, data_global)
        saver.run()
        print("Matches Saved as N5")

        print("Interest point matching is done")
    
    def run(self):
        self.match()