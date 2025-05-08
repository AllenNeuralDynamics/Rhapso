import numpy as np

class RANSAC:
    def __init__(self, iterations=1000, threshold=10.0):
        self.iterations = iterations
        self.threshold = threshold

    def filter_matches(self, pointsA, pointsB, matches):
        matches_arr = np.array(matches)
        diff_vectors = [pointsB[j] - pointsA[i] for i, j in matches_arr]
        diff_vectors = np.array(diff_vectors)

        best_inliers = []
        best_translation = None

        for _ in range(self.iterations):
            rand_idx = np.random.randint(0, len(matches_arr))
            t_candidate = diff_vectors[rand_idx]
            errors = np.linalg.norm(diff_vectors - t_candidate, axis=1)
            inliers = np.where(errors < self.threshold)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_translation = t_candidate

        if best_inliers:
            return matches_arr[best_inliers].tolist(), best_translation
        return [], None
