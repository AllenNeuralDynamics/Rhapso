from enum import Enum
import numpy as np
from sklearn.neighbors import KDTree
from .data_saver import save_correspondences
import sys

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
                 ransac_params=None,
                 logging=None):
        """Initialize matcher with model parameters"""
        self.transform_model = transform_model
        self.reg_model = reg_model
        self.lambda_val = lambda_val
        self.ransac_params = ransac_params or {}
        self.model = self._create_model()
        
        # Set default logging configuration if none provided
        self.logging = {
            'detailed_descriptor_breakdown': True,
            'ratio_test_output': True,
            'basis_point_details': 3,  # Number of basis points to show details for
        }
        
        # Update with user-provided logging settings
        if logging is not None:
            self.logging.update(logging)
            
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
            sys.exit(1)

        try:
            # Deduplicate correspondences before RANSAC
            unique_candidates = self._deduplicate_correspondences(candidates)
            print(f"Deduplicated to {len(unique_candidates)} unique point pairs")
            
            # Filter correspondences using RANSAC
            inliers = self._compute_ransac(unique_candidates, pointsA, pointsB)
            return inliers
        except Exception as e:
            print(f"Error in _compute_ransac: {e}")
            sys.exit(1)

    def _deduplicate_correspondences(self, correspondences):
        """Deduplicate correspondences to ensure unique point pairs"""
        unique_pairs = {}
        
        # For each correspondence, keep only the one with the smallest distance
        for corr in correspondences:
            pair_key = (corr['pointA_idx'], corr['pointB_idx'])
            
            # If the pair doesn't exist yet, or this match has a smaller distance than the existing one
            if pair_key not in unique_pairs or corr['distance'] < unique_pairs[pair_key]['distance']:
                unique_pairs[pair_key] = corr
                
        return list(unique_pairs.values())

    def _compute_matching(self, descriptorsA, descriptorsB, lookup_tree, difference_threshold, ratio_of_distance):
        """Compute matching using KNN search and Lowe's ratio test
        
        Uses sklearn KDTree which provides equivalent functionality to ImgLib2's Eytzinger layout KDTree
        """
        correspondences = []
        
        print(f"\n🔍 _compute_matching() - Input Parameters:")
        print(f"  📊 descriptors_A size: {len(descriptorsA)}")
        print(f"  📊 descriptors_B size: {lookup_tree.data.shape[0]}")
        print(f"  📊 difference_threshold: {difference_threshold}")
        print(f"  📊 ratio_of_distance: {ratio_of_distance}")
        
        debug_count = 0
        # Use the logging configuration to determine how many ratio tests to print
        max_debug_prints = self.logging.get('ratio_test_output', True)
        if max_debug_prints is True:
            max_debug_prints = float('inf')  # Print all
        elif max_debug_prints is False:
            max_debug_prints = 0  # Print none
        else:
            max_debug_prints = int(max_debug_prints)  # Print specific number
        
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
                # EXACTLY matching BigStitcherSpark's implementation:
                # best * ratioOfDistance <= secondBest
                ratio_test_value = best_distance * ratio_of_distance
                
                # Print detailed debugging info for understanding the ratio test
                if debug_count < max_debug_prints:
                    point_a_id = desc_a['point_index']
                    point_b_id = descriptorsB[best_match_idx]['point_index']
                    
                    if best_distance < difference_threshold and ratio_test_value <= second_best_distance:
                        print(f"✅ ACCEPTED: PointA[{point_a_id}] ↔ PointB[{point_b_id}] | " +
                              f"best={best_distance:.6f} < threshold={difference_threshold:.6f} && " +
                              f"ratio_test={ratio_test_value:.6f} <= second_best={second_best_distance:.6f}")
                        print(f"   📊 Detailed: best={best_distance:.6f}, secondBest={second_best_distance:.6f}, " +
                              f"best/secondBest={best_distance/second_best_distance:.6f}, " +
                              f"1/ratio={1.0/ratio_of_distance:.6f}")
                        debug_count += 1
                    elif best_distance >= difference_threshold:
                        print(f"❌ REJECTED (distance threshold): PointA[{point_a_id}] ↔ PointB[{point_b_id}] | " +
                              f"best={best_distance:.6f} >= threshold={difference_threshold:.6f}")
                        debug_count += 1
                    else:
                        print(f"❌ REJECTED (ratio test): PointA[{point_a_id}] ↔ PointB[{point_b_id}] | " +
                              f"ratio_test={ratio_test_value:.6f} > second_best={second_best_distance:.6f}")
                        print(f"   📊 Detailed: best={best_distance:.6f}, secondBest={second_best_distance:.6f}, " +
                              f"best/secondBest={best_distance/second_best_distance:.6f}, " +
                              f"1/ratio={1.0/ratio_of_distance:.6f}")
                        debug_count += 1
                
                # BigStitcherSpark implementation: best * ratioOfDistance <= secondBest
                if best_distance < difference_threshold and ratio_test_value <= second_best_distance:
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
                        'ratio': best_distance / second_best_distance if second_best_distance > 0 else float('inf')
                    }
                    correspondences.append(correspondence)
                    final_correspondences += 1
                    
            except Exception as e:
                if debug_count < max_debug_prints:
                    print(f"❌ Error in matching descriptor {desc_idx}: {e}")
                    debug_count += 1
                continue
                
        if debug_count >= max_debug_prints and max_debug_prints > 0:
            print(f"... (showing first {max_debug_prints} ratio test results, continuing silently)")
        
        # Lowe's ratio test summary line (with emoji)
        ratio_test_failed = processed_descriptors - ratio_test_passed
        percent_passed = (ratio_test_passed / processed_descriptors * 100) if processed_descriptors > 0 else 0.0
        print(f"\n🟢 Lowe's ratio test results: {ratio_test_passed}/{processed_descriptors} passed ({percent_passed:.1f}%) {ratio_test_failed} failed")

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
            descriptorsA = self._create_descriptors(pointsA, redundancy, "A")
            descriptorsB = self._create_descriptors(pointsB, redundancy, "B")
            
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
            correspondences = self._compute_matching(descriptorsA, descriptorsB, lookup_tree, difference_threshold, ratio_of_distance)
            
            print(f"\n🎯 Found {len(correspondences)} correspondence candidates after Lowe's ratio test")
            return correspondences
            
        except Exception as e:
            print(f"❌ Error in _get_candidates: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _create_descriptors(self, points, redundancy=4, view_label=""):
        """Create local coordinate system descriptors for each point"""
        print(f"🛠️ Creating descriptors for View {view_label}...")
        descriptors = []
        points_array = np.array(points)
        total_basis_points = len(points)
        
        if len(points) < 4:
            print(f"⚠️ Not enough points ({len(points)}) to create descriptors (need at least 4)")
            return descriptors

        try:
            # Build KDTree for efficient neighbor search
            kdtree = KDTree(points_array)
            
            # Constants matching Java implementation exactly
            num_neighbors_needed = 3  # We need exactly 3 neighbors for each descriptor
            max_neighbors_to_find = redundancy + 1  # +1 for the point itself
            
            print(f"🔍 Creating descriptors with {redundancy} redundancy for {total_basis_points} basis points")
            
            # Determine how many basis points to show details for - check both parameter names
            basis_point_details = self.logging.get('basis_points_details', 
                              self.logging.get('basis_point_details', 3))
            
            if basis_point_details is True:
                basis_point_details = float('inf')  # Show all
            elif basis_point_details is False:
                basis_point_details = 0  # Show none
            else:
                basis_point_details = int(basis_point_details)  # Show specific number
            
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
                    
                    # Only print detailed information for the first N basis points (configurable)
                    if basis_point_index < basis_point_details:
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
                                
                                # Create 6D descriptor with explanations
                                descriptor_6d = self._create_6d_descriptor(
                                    point, neighbor_points[0], neighbor_points[1], neighbor_points[2],
                                    print_details=(basis_point_index < basis_point_details and 
                                                  self.logging.get('detailed_descriptor_breakdown', True))
                                )
                                
                                descriptor_entry = {
                                    'coordinates': descriptor_6d,
                                    'point_index': basis_point_index,
                                    'neighbor_indices': neighbor_ids,
                                    'base_point': point
                                }
                                
                                # Print detailed descriptor entry information
                                if basis_point_index < basis_point_details:
                                    print(f"  📝 Adding descriptor to list with details:")
                                    print(f"    🔹 Basis point index: {descriptor_entry['point_index']}")
                                    print(f"    🔹 Base point coordinates: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
                                    print(f"    🔹 Neighbor indices: {descriptor_entry['neighbor_indices']}")
                                    print(f"    🔹 6D descriptor: [{descriptor_6d[0]:.6f}, {descriptor_6d[1]:.6f}, {descriptor_6d[2]:.6f}, " +
                                          f"{descriptor_6d[3]:.6f}, {descriptor_6d[4]:.6f}, {descriptor_6d[5]:.6f}]")
                                
                                descriptors.append(descriptor_entry)
                                
                                # Print descriptor details for configured number of basis points
                                if basis_point_index < basis_point_details:
                                    # Convert neighbor local indices to 1-indexed format (matching Java output)
                                    neighbor_display_ids = [idx + 1 for idx in neighbor_combo]
                                    print(f"  ✅ 6D descriptor [{neighbor_display_ids[0]},{neighbor_display_ids[1]},{neighbor_display_ids[2]}]: " +
                                          f"[{descriptor_6d[0]:.6f}, {descriptor_6d[1]:.6f}, {descriptor_6d[2]:.6f}, " +
                                          f"{descriptor_6d[3]:.6f}, {descriptor_6d[4]:.6f}, {descriptor_6d[5]:.6f}]")
                            
                            except Exception as e:
                                if basis_point_index < basis_point_details:
                                    print(f"  ❌ Failed to create descriptor: {e}")
                                continue
                    
                except Exception as e:
                    if basis_point_index < basis_point_details:
                        print(f"❌ Error creating descriptors for point {basis_point_index}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error in _create_descriptors: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        print(f"🔢 Total descriptors created: {len(descriptors)}")
        return descriptors

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
            
        inliers, avg_error = self._filter_with_ransac(candidates)
        
        # Calculate percentage of inliers
        inlier_percentage = (len(inliers) / len(candidates) * 100) if candidates else 0
        
        # Format the output message with inlier statistics
        print(f"Remaining inliers after RANSAC: {len(inliers)} of {len(candidates)} " +
              f"({inlier_percentage:.0f}%) with average error {avg_error:.16f}")
        
        return [(c['indices'][0], c['indices'][1]) for c in inliers]

    def _filter_with_ransac(self, candidates, max_iterations=1000, inlier_threshold=5.0):
        """Apply RANSAC to find best model and inlier set"""
        if not candidates:
            return [], 0.0
            
        best_inliers = []
        best_model = None
        best_avg_error = float('inf')
        
        for _ in range(max_iterations):
            sample_idx = np.random.choice(len(candidates))
            model_candidate = candidates[sample_idx]
            
            inliers = []
            total_error = 0.0
            
            for candidate in candidates:
                error = self._compute_geometric_error(model_candidate, candidate)
                if error < inlier_threshold:
                    inliers.append(candidate)
                    total_error += error
                    
            # Calculate average error for this model
            avg_error = total_error / len(inliers) if inliers else float('inf')
            
            # Update best model based on number of inliers (primary) and average error (secondary)
            if len(inliers) > len(best_inliers) or (len(inliers) == len(best_inliers) and avg_error < best_avg_error):
                best_inliers = inliers
                best_model = model_candidate
                best_avg_error = avg_error
    
        # Recalculate average error for the best model's inliers
        if best_inliers:
            total_error = sum(self._compute_geometric_error(best_model, inlier) for inlier in best_inliers)
            best_avg_error = total_error / len(best_inliers)
        else:
            best_avg_error = 0.0
                
        return best_inliers, best_avg_error

    def _compute_geometric_error(self, model, candidate):
        """Calculate geometric error between model and candidate match"""
        # TODO: Implement proper geometric error computation based on transformation model
        return np.linalg.norm(
            (model['pointB'] - model['pointA']) - (candidate['pointB'] - candidate['pointA'])
        )

    def _create_6d_descriptor(self, center, p1, p2, p3, print_details=True):
        """Compute 6D descriptor following BigStitcherSpark algorithm."""
        # compute relative vectors
        v1 = p1 - center
        v2 = p2 - center
        v3 = p3 - center
        
        # distances to basis
        d1 = np.linalg.norm(v1); d2 = np.linalg.norm(v2); d3 = np.linalg.norm(v3)
        dist_list = [d1, d2, d3]
        
        # identify farthest neighbor
        far_idx = int(np.argmax(dist_list))
        dist_f = dist_list[far_idx]
        
        # reorder vectors: farthest, second, third
        vectors = [v1, v2, v3]
        others = [i for i in (0,1,2) if i != far_idx]
        sec_idx, thr_idx = others
        
        v_f = vectors[far_idx]
        v_sec = vectors[sec_idx]
        v_thr = vectors[thr_idx]
        
        # build local coordinate axes
        X = v_f / dist_f
        
        # Calculate cross product for Z-axis
        Z = np.cross(v_f, v_sec)
        
        # Check for zero or near-zero norm to avoid divide-by-zero
        z_norm = np.linalg.norm(Z)
        if z_norm < 1e-10:  # Use a small epsilon value
            # Vectors are nearly parallel, try with third vector instead
            Z = np.cross(v_f, v_thr)
            z_norm = np.linalg.norm(Z)
            
            # If still problematic, create an arbitrary perpendicular vector
            if z_norm < 1e-10:
                # Find non-zero component in v_f to create perpendicular vector
                if abs(X[0]) > 1e-10:
                    Z = np.array([0, 1, 0])
                else:
                    Z = np.array([1, 0, 0])
                Z = Z - X * np.dot(Z, X)  # Make it perpendicular to X
                z_norm = np.linalg.norm(Z)
        
        # Now normalize Z safely
        Z = Z / z_norm if z_norm > 1e-10 else np.array([0.0, 0.0, 1.0])
        
        # Continue with Y calculation
        Y = np.cross(Z, X)
        
        # project neighbors into local frame
        x2, y2 = np.dot(v_sec, X), np.dot(v_sec, Y)
        x3, y3, z3 = np.dot(v_thr, X), np.dot(v_thr, Y), np.dot(v_thr, Z)

        # Apply sign conventions to match BigStitcherSpark
        y2 = -y2   # Negate the Y-coordinate of the second neighbor
        y3 = abs(y3)  # Use absolute value for the Y-coordinate of the third neighbor
        z3 = abs(z3)  # Use absolute value for the Z-coordinate of the third neighbor

        # Print detailed breakdown if enabled
        if print_details:
            print("🧮 Mathematical breakdown for this 6D descriptor:")
            print(f"      📍 Farthest neighbor index: {far_idx} at distance {dist_f:.6f}")
            print(f"      🔄 X-axis = {X.tolist()}")
            print(f"      🧭 Y-axis = {Y.tolist()}")
            print(f"      ⚓ Z-axis = {Z.tolist()}")
            print(f"      📏 Local coords of second neighbor: [x2={x2:.6f}, y2={y2:.6f}]")
            print(f"      📏 Local coords of third  neighbor: [x3={x3:.6f}, y3={y3:.6f}, z3={z3:.6f}]")
            print(f"      📐 Original vectors:")
            print(f"         v1 = {v1.tolist()} (length={d1:.6f})")
            print(f"         v2 = {v2.tolist()} (length={d2:.6f})")
            print(f"         v3 = {v3.tolist()} (length={d3:.6f})")
        
        # assemble final descriptor
        return np.array([dist_f, x2, y2, x3, y3, z3], dtype=np.float64)
