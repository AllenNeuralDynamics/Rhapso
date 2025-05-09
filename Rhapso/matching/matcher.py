from enum import Enum
import numpy as np
from sklearn.neighbors import KDTree

class Matcher:
    """Handles all matching operations including model creation and geometric matching"""
    
    # Define model types as class attributes
    class TransformationModel(Enum):
        TRANSLATION = "TRANSLATION"
        RIGID = "RIGID" 
        AFFINE = "AFFINE"

    class RegularizationModel(Enum):
        NONE = "NONE"
        IDENTITY = "IDENTITY"
        TRANSLATION = "TRANSLATION"
        RIGID = "RIGID"
        AFFINE = "AFFINE"

    def __init__(self, 
                 transform_model=TransformationModel.AFFINE,
                 reg_model=RegularizationModel.RIGID,
                 lambda_val=0.1,
                 ransac_params=None):
        """Initialize matcher with model parameters"""
        self.transform_model = transform_model
        self.reg_model = reg_model
        self.lambda_val = lambda_val
        self.ransac_params = ransac_params or {}
        self.model = self._create_model()
        print(f"Created matcher with model type: {self.model['type']}")

    def _create_model(self):
        """Create transformation model with optional regularization"""
        print(f"Creating model: transform={self.transform_model.value}, "
              f"regularization={self.reg_model.value if self.reg_model else 'None'}")
        return {
            "type": "InterpolatedAffineModel3D",
            "transform": self.transform_model,
            "regularization": self.reg_model
        }

    def match(self, pointsA, pointsB):
        """Complete matching pipeline from candidates to RANSAC filtering"""
        # Get initial candidates using KD-tree matching
        candidates = self._get_candidates(pointsA, pointsB)
        
        # Compute initial correspondences
        correspondences = self._compute_match(candidates, pointsA, pointsB)
        print(f"Found {len(correspondences)} initial correspondences")
        
        # Filter correspondences using RANSAC
        inliers = self._compute_ransac(correspondences, pointsA, pointsB)
        print(f"RANSAC filtering retained {len(inliers)} inlier matches")
        return inliers

    def _get_candidates(self, pointsA, pointsB):
        """Find candidate matches using simple distance comparison (temporary)"""
        # Minimal debug output
        print(f"Processing match between point sets of size {len(pointsA)} and {len(pointsB)}")
        
        # For now, just return dummy data
        dummy_distances = np.array([[1.0, 2.0], [1.5, 2.5]])
        dummy_indices = np.array([[0, 1], [1, 0]])
        
        return {
            'distances': dummy_distances,
            'indices': dummy_indices
        }

    def _compute_match(self, candidates, pointsA, pointsB, ratio_threshold=0.8):
        """Filter matches using Lowe's ratio test"""
        pairs = []
        distances = candidates['distances']
        indices = candidates['indices']
        
        for i in range(len(pointsA)):
            best_dist = distances[i, 0]
            second_best = distances[i, 1]
            
            if best_dist < second_best * ratio_threshold:
                pairs.append((i, indices[i, 0]))
        
        return pairs

    def _compute_ransac(self, correspondences, pointsA, pointsB):
        """Apply RANSAC to filter correspondences"""
        if not correspondences:
            return []
            
        candidates = []
        for idx_a, idx_b in correspondences:
            candidates.append({
                'pointA': pointsA[idx_a],
                'pointB': pointsB[idx_b],
                'weight': 1.0,
                'indices': (idx_a, idx_b)
            })
            
        inliers = self._filter_with_ransac(candidates)
        return [(c['indices'][0], c['indices'][1]) for c in inliers]

    def _filter_with_ransac(self, candidates, max_iterations=1000, inlier_threshold=5.0):
        """Apply RANSAC to find best model and inlier set"""
        if not candidates:
            return []
            
        best_inliers = []
        best_model = None
        
        for _ in range(max_iterations):
            sample_idx = np.random.choice(len(candidates))
            model_candidate = candidates[sample_idx]
            
            inliers = []
            for candidate in candidates:
                error = self._compute_geometric_error(model_candidate, candidate)
                if error < inlier_threshold:
                    inliers.append(candidate)
                    
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = model_candidate
                
        return best_inliers

    def _compute_geometric_error(self, model, candidate):
        """Calculate geometric error between model and candidate match"""
        # TODO: Implement proper geometric error computation based on transformation model
        return np.linalg.norm(
            (model['pointB'] - model['pointA']) - (candidate['pointB'] - candidate['pointA'])
        )
