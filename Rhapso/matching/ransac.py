import numpy as np

class RANSAC:
    def __init__(self, iterations=1000, threshold=10.0):
        """Initialize RANSAC with iteration count and threshold"""
        self.iterations = iterations
        self.threshold = threshold

    def filter_matches(self, pointsA, pointsB, matches):
        """Filter matches using RANSAC algorithm"""
        # Check if we have any matches to process
        if len(matches) == 0:
            return [], None

        # Convert inputs to numpy arrays if they aren't already
        matches_arr = np.array(matches)
        diff_vectors = []
        
        # Calculate difference vectors between matched points
        for i, j in matches_arr:
            if i < len(pointsA) and j < len(pointsB):
                diff = pointsB[j] - pointsA[i]
                diff_vectors.append(diff)
        
        if not diff_vectors:
            return [], None

        diff_vectors = np.array(diff_vectors)
        best_inliers = []
        best_translation = None

        # RANSAC iteration
        for _ in range(self.iterations):
            if len(diff_vectors) > 0:  # Only proceed if we have vectors to work with
                rand_idx = np.random.randint(0, len(matches_arr))
                t_candidate = diff_vectors[rand_idx]
                errors = np.linalg.norm(diff_vectors - t_candidate, axis=1)
                inlier_indices = np.where(errors < self.threshold)[0]
                
                if len(inlier_indices) > len(best_inliers):
                    best_inliers = inlier_indices
                    best_translation = t_candidate

        # Return results
        if len(best_inliers) > 0:
            print(f"RANSAC found {len(best_inliers)} inliers")
            return matches_arr[best_inliers].tolist(), best_translation
        else:
            print("RANSAC found no inliers")
            return [], None
