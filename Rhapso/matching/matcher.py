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
        try:
            # Get initial candidates using geometric hashing with Lowe's ratio test
            candidates = self._get_candidates(pointsA, pointsB)
            print(f"Found {len(candidates)} correspondence candidates after Lowe's ratio test")
        except Exception as e:
            print(f"Error in _get_candidates: {e}")
            return []

        try:
            # Filter correspondences using RANSAC
            inliers = self._compute_ransac(candidates, pointsA, pointsB)
            print(f"RANSAC filtering retained {len(inliers)} inlier matches")
            return inliers
        except Exception as e:
            print(f"Error in _compute_ransac: {e}")
            return []

    def _compute_matching(self, descriptorsA, descriptorsB, lookup_tree, difference_threshold, ratio_of_distance):
        """Compute matching using KNN search and Lowe's ratio test
        
        Uses sklearn KDTree which provides equivalent functionality to ImgLib2's Eytzinger layout KDTree:
        - Both support k-nearest neighbor queries with Euclidean distance
        - Both return sorted distances and indices
        - Both handle edge cases properly
        """
        correspondences = []
        
        print(f"\n🔍 _compute_matching() - Applying Lowe's ratio test:")
        print(f"  📊 ratio_of_distance: {ratio_of_distance}")
        print(f"  🌳 KDTree contains {lookup_tree.data.shape[0]} descriptors with {lookup_tree.data.shape[1]} dimensions")
        
        debug_count = 0
        max_debug_prints = 3
        
        # Tracking variables for summary
        processed_descriptors = 0
        ratio_test_passed = 0
        unique_pairs = set()  # Track unique (pointA_idx, pointB_idx) pairs
        final_correspondences = 0
        
        # Verify KDTree is properly constructed
        if lookup_tree.data.shape[0] < 2:
            print("⚠️ Warning: KDTree has fewer than 2 points, cannot perform 2-NN search")
            return correspondences
        
        for desc_idx, desc_a in enumerate(descriptorsA):
            processed_descriptors += 1
            try:
                # Verify descriptor dimensionality matches KDTree
                query_point = desc_a['coordinates']
                if len(query_point) != lookup_tree.data.shape[1]:
                    if debug_count < max_debug_prints:
                        print(f"❌ Dimension mismatch: query={len(query_point)}, tree={lookup_tree.data.shape[1]}")
                        debug_count += 1
                    continue
                
                # Search for 2 nearest neighbors in descriptor space
                # This is equivalent to ImgLib2's KDTree.search(point, k=2) functionality
                k_neighbors = min(2, lookup_tree.data.shape[0])  # Handle case where tree has < 2 points
                distances, indices = lookup_tree.query([query_point], k=k_neighbors)
                
                if len(distances[0]) < 2:
                    continue
                    
                best_distance = distances[0][0]
                second_best_distance = distances[0][1]
                best_match_idx = indices[0][0]
                
                # Verify distances are non-negative (sanity check)
                if best_distance < 0 or second_best_distance < 0:
                    if debug_count < max_debug_prints:
                        print(f"❌ Invalid negative distance: best={best_distance}, second={second_best_distance}")
                        debug_count += 1
                    continue
                
                # Apply Lowe's ratio test to filter ambiguous matches
                # This matches the standard computer vision approach used in ImgLib2
                ratio_test = best_distance / second_best_distance if second_best_distance > 0 else float('inf')
                
                if best_distance < difference_threshold and ratio_test <= (1.0 / ratio_of_distance):
                    # Accept the match
                    ratio_test_passed += 1
                    desc_b = descriptorsB[best_match_idx]
                    
                    # Create pair key for deduplication tracking
                    pair_key = (desc_a['point_index'], desc_b['point_index'])
                    unique_pairs.add(pair_key)
                    
                    correspondence = {
                        'pointA_idx': desc_a['point_index'],
                        'pointB_idx': desc_b['point_index'],
                        'distance': best_distance,
                        'ratio': ratio_test
                    }
                    correspondences.append(correspondence)
                    final_correspondences += 1
                    
                    if debug_count < max_debug_prints:
                        print(f"✅ ACCEPTED: PointA[{desc_a['point_index']}] ↔ PointB[{desc_b['point_index']}] | "
                              f"best={best_distance:.6f} < threshold={difference_threshold:.6f} && "
                              f"ratio_test={ratio_test:.6f} <= threshold={1.0/ratio_of_distance:.6f}")
                        debug_count += 1
                else:
                    # Reject the match
                    if debug_count < max_debug_prints:
                        print(f"❌ REJECTED (ratio test): PointA[{desc_a['point_index']}] ↔ PointB[{descriptorsB[best_match_idx]['point_index']}] | "
                              f"ratio_test={ratio_test:.6f} > threshold={1.0/ratio_of_distance:.6f}")
                        debug_count += 1
                    
            except Exception as e:
                if debug_count < max_debug_prints:
                    print(f"❌ Error in matching descriptor {desc_idx}: {e}")
                    debug_count += 1
                continue
                
        if debug_count >= max_debug_prints:
            print(f"... (showing first {max_debug_prints} ratio test results, continuing silently)")
        
        # Print comprehensive summary
        print(f"\n✅ _compute_matching() - Results:")
        print(f"  📊 Processed descriptors: {processed_descriptors}")
        print(f"  📊 Matches passed Lowe's ratio test: {ratio_test_passed}")
        print(f"  📊 Unique pairs after deduplication: {len(unique_pairs)}")
        print(f"  📊 Final correspondences added: {final_correspondences}")
        print(f"  📊 Total correspondences in list: {len(correspondences)}")
        
        return correspondences

    def _get_candidates(self, pointsA, pointsB, redundancy=4, difference_threshold=float('inf'), ratio_of_distance=3.0):
        """Find candidate matches using geometric hashing with local coordinate system descriptors.
        This method performs Lowe's ratio test to filter ambiguous matches."""
        print(f"🔍 Finding candidates between point sets of size {len(pointsA)} and {len(pointsB)}")
        
        try:
            # Create descriptors for both point sets
            descriptorsA = self._create_descriptors(pointsA, redundancy)
            descriptorsB = self._create_descriptors(pointsB, redundancy)
            
            print(f"📊 Created {len(descriptorsA)} descriptors for A, {len(descriptorsB)} descriptors for B")
            
            # Build KDTree lookup structure for descriptorsB
            if len(descriptorsB) == 0:
                print("❌ No descriptors created for point set B")
                return []
                
            descriptor_coords_B = np.array([desc['coordinates'] for desc in descriptorsB])
            
            # Verify descriptor coordinates are valid
            if descriptor_coords_B.size == 0:
                print("❌ No valid descriptor coordinates for point set B")
                return []
            
            if np.any(np.isnan(descriptor_coords_B)) or np.any(np.isinf(descriptor_coords_B)):
                print("⚠️ Warning: Found NaN or Inf values in descriptors, filtering...")
                valid_mask = np.all(np.isfinite(descriptor_coords_B), axis=1)
                descriptor_coords_B = descriptor_coords_B[valid_mask]
                descriptorsB = [descriptorsB[i] for i in range(len(descriptorsB)) if valid_mask[i]]
                print(f"📊 After filtering: {len(descriptorsB)} valid descriptors for B")
            
            if len(descriptorsB) == 0:
                print("❌ No valid descriptors remaining for point set B after filtering")
                return []
            
            # Build KDTree - this uses sklearn's implementation which is functionally equivalent
            # to ImgLib2's Eytzinger layout for our purposes
            lookup_tree = KDTree(descriptor_coords_B, metric='euclidean')
            print(f"🌳 Built KDTree with {lookup_tree.data.shape[0]} nodes, {lookup_tree.data.shape[1]} dimensions")
            
            # Find correspondences using KNN search and apply Lowe's ratio test
            correspondences = self._compute_matching(descriptorsA, descriptorsB, lookup_tree, 
                                                  difference_threshold, ratio_of_distance)
            
            print(f"\n🎯 Found {len(correspondences)} correspondence candidates after Lowe's ratio test")
            return correspondences
            
        except Exception as e:
            print(f"❌ Error in _get_candidates: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _create_descriptors(self, points, redundancy=4):
        """Create local coordinate system descriptors for each point"""
        print("🛠️ Creating descriptors...")
        descriptors = []
        points_array = np.array(points)
        
        if len(points) < 4:
            print(f"⚠️ Not enough points ({len(points)}) to create descriptors (need at least 4)")
            return descriptors

        try:
            # Build KDTree for efficient neighbor search
            kdtree = KDTree(points_array)
            
            # Constants matching Java implementation exactly
            num_neighbors_needed = 3  # We need exactly 3 neighbors for each descriptor
            max_neighbors_to_find = redundancy + 1  # +1 for the point itself
            
            print(f"🔍 Creating descriptors with {redundancy} redundancy")
            
            for basis_point_index, point in enumerate(points_array):
                try:
                    # Find nearest neighbors (including the point itself)
                    k_search = min(max_neighbors_to_find, len(points_array))
                    distances, indices = kdtree.query([point], k=k_search)
                    
                    # Remove the point itself (first result) and get neighbor indices
                    neighbor_indices = indices[0][1:]  # Skip index 0 (the point itself)
                    neighbor_distances = distances[0][1:]
                    
                    if len(neighbor_indices) < num_neighbors_needed:
                        continue  # Skip if not enough neighbors
                    
                    # Only print detailed information for first 3 basis points
                    if basis_point_index < 3:
                        print(f"🎯 Basis Point {basis_point_index}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
                        
                        # Show only the first 4 neighbors (matching Java output)
                        for i in range(min(4, len(neighbor_indices))):
                            neighbor_idx = neighbor_indices[i]
                            neighbor_point = points_array[neighbor_idx]
                            distance = neighbor_distances[i]
                            
                            print(f"  📍 Neighbor[{i}] ID:{neighbor_idx} " +
                                  f"[{neighbor_point[0]:.3f}, {neighbor_point[1]:.3f}, {neighbor_point[2]:.3f}] " +
                                  f"dist:{distance:.3f}")
                    
                    # Generate exactly 4 descriptors using combinations of first 4 neighbors
                    # This matches the Java implementation: C(4,3) = 4 combinations
                    from itertools import combinations
                    available_neighbors = min(4, len(neighbor_indices))  # Use only first 4 neighbors
                    
                    if available_neighbors >= num_neighbors_needed:
                        neighbor_combinations = list(combinations(range(available_neighbors), num_neighbors_needed))
                        
                        # Create descriptors for all combinations (should be exactly 4)
                        for combo_idx, neighbor_combo in enumerate(neighbor_combinations):
                            try:
                                # Get the 3 neighbor points for this combination
                                neighbor_points = []
                                neighbor_ids = []
                                for neighbor_local_idx in neighbor_combo:
                                    neighbor_global_idx = neighbor_indices[neighbor_local_idx]
                                    neighbor_points.append(points_array[neighbor_global_idx])
                                    neighbor_ids.append(neighbor_global_idx)
                                
                                # Create 6D descriptor from the 3 neighbors
                                descriptor_6d = self._create_6d_descriptor(
                                    point, neighbor_points[0], neighbor_points[1], neighbor_points[2]
                                )
                                
                                descriptor_entry = {
                                    'coordinates': descriptor_6d,
                                    'point_index': basis_point_index,
                                    'neighbor_indices': neighbor_ids,
                                    'base_point': point
                                }
                                
                                descriptors.append(descriptor_entry)
                                
                                # Print descriptor details for first 3 basis points
                                if basis_point_index < 3:
                                    # Convert neighbor local indices to 1-indexed format (matching Java output)
                                    neighbor_display_ids = [idx + 1 for idx in neighbor_combo]
                                    print(f"  ✅ 6D descriptor [{neighbor_display_ids[0]},{neighbor_display_ids[1]},{neighbor_display_ids[2]}]: " +
                                          f"[{descriptor_6d[0]:.6f}, {descriptor_6d[1]:.6f}, {descriptor_6d[2]:.6f}, " +
                                          f"{descriptor_6d[3]:.6f}, {descriptor_6d[4]:.6f}, {descriptor_6d[5]:.6f}]")
                            
                            except Exception as e:
                                if basis_point_index < 3:
                                    print(f"  ❌ Failed to create descriptor: {e}")
                                continue
                    
                except Exception as e:
                    if basis_point_index < 3:
                        print(f"❌ Error creating descriptors for point {basis_point_index}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error in _create_descriptors: {e}")
            import traceback
            traceback.print_exc()
            return descriptors
        
        print(f"🔢 Total descriptors created: {len(descriptors)}")
        return descriptors

    def _create_6d_descriptor(self, center, p1, p2, p3):
        """Create 6D geometric descriptor from center point and 3 neighbors"""
        # Calculate distances between all pairs
        d01 = np.linalg.norm(p1 - center)
        d02 = np.linalg.norm(p2 - center) 
        d03 = np.linalg.norm(p3 - center)
        d12 = np.linalg.norm(p2 - p1)
        d13 = np.linalg.norm(p3 - p1)
        d23 = np.linalg.norm(p3 - p2)
        
        # Sort distances to create invariant descriptor
        distances = [d01, d02, d03, d12, d13, d23]
        distances.sort()
        
        return np.array(distances, dtype=np.float64)

    def _compute_ransac(self, correspondences, pointsA, pointsB):
        """Apply RANSAC to filter correspondences"""
        if not correspondences:
            return []
            
        candidates = []
        for corr in correspondences:
            candidates.append({
                'pointA': pointsA[corr['pointA_idx']],
                'pointB': pointsB[corr['pointB_idx']],
                'weight': 1.0,
                'indices': (corr['pointA_idx'], corr['pointB_idx'])
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
