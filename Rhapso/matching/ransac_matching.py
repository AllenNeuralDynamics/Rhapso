import numpy as np
from sklearn.neighbors import KDTree
import itertools
import random
from scipy.linalg import eigh

"""Handles all matching operations including model creation and geometric matching"""

class RansacMatching:
    def __init__(self, process_pairs, num_neighbors, redundancy, significance, num_required_neighbors):
        self.process_pairs = process_pairs
        self.num_neighbors = num_neighbors
        self.redundancy = redundancy
        self.significance = significance
        self.num_required_neighbors = num_required_neighbors
    
    def fit_rigid_model(self, matches):
        matches = np.array(matches)    # shape (N, 2, 3)
        P = matches[:, 0]              # source points
        Q = matches[:, 1]              # target points
        weights = np.ones(P.shape[0])  # uniform weights for now

        sum_w = np.sum(weights)

        # Weighted centroids
        pc = np.average(P, axis=0, weights=weights)
        qc = np.average(Q, axis=0, weights=weights)

        # Centered and weighted coordinates
        P_centered = (P - pc) * weights[:, None]
        Q_centered = Q - qc

        # Cross-covariance matrix S
        S = P_centered.T @ Q_centered  # shape: (3, 3)
        Sxx, Sxy, Sxz = S[0]
        Syx, Syy, Syz = S[1]
        Szx, Szy, Szz = S[2]

        # Build 4x4 N matrix for quaternion extraction
        N = np.array([
            [Sxx + Syy + Szz, Syz - Szy,       Szx - Sxz,       Sxy - Syx],
            [Syz - Szy,       Sxx - Syy - Szz, Sxy + Syx,       Szx + Sxz],
            [Szx - Sxz,       Sxy + Syx,      -Sxx + Syy - Szz, Syz + Szy],
            [Sxy - Syx,       Szx + Sxz,       Syz + Szy,      -Sxx - Syy + Szz]
        ])

        # Find eigenvector with largest eigenvalue
        eigenvalues, eigenvectors = eigh(N)
        q = eigenvectors[:, np.argmax(eigenvalues)]  # q = [q0, qx, qy, qz]
        q0, qx, qy, qz = q

        # Convert quaternion to rotation matrix
        R = np.array([
            [q0*q0 + qx*qx - qy*qy - qz*qz,     2*(qx*qy - q0*qz),           2*(qx*qz + q0*qy)],
            [2*(qy*qx + q0*qz),                 q0*q0 - qx*qx + qy*qy - qz*qz, 2*(qy*qz - q0*qx)],
            [2*(qz*qx - q0*qy),                 2*(qz*qy + q0*qx),           q0*q0 - qx*qx - qy*qy + qz*qz]
        ])

        # Compute translation
        t = qc - R @ pc

        # Combine into 4x4 rigid transformation matrix
        rigid_matrix = np.eye(4)
        rigid_matrix[:3, :3] = R
        rigid_matrix[:3, 3] = t

        return rigid_matrix

    def fit_affine_model(self, matches):
        matches = np.array(matches)    # shape (N, 2, 3)
        P = matches[:, 0]              # source points
        Q = matches[:, 1]              # target points
        weights = np.ones(P.shape[0])  # uniform weights

        ws = np.sum(weights)

        pc = np.average(P, axis=0, weights=weights)
        qc = np.average(Q, axis=0, weights=weights)

        P_centered = P - pc
        Q_centered = Q - qc

        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        
        for i in range(P.shape[0]):
            w = weights[i]
            p = P_centered[i]
            q = Q_centered[i]

            A += w * np.outer(p, p)
            B += w * np.outer(p, q)

        det = np.linalg.det(A)
        if det == 0:
            raise ValueError("Ill-defined data points (det=0)")

        A_inv = np.linalg.inv(A)
        M = A_inv @ B  # 3x3 transformation matrix

        t = qc - M @ pc  # translation

        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = M
        affine_matrix[:3, 3] = t

        return affine_matrix
    
    def test(self, candidates, model, max_epsilon, min_inlier_ratio, min_num_inliers):
        inliers = []
        for idxA, pointA, view_a, idxB, pointB, view_b in candidates:
            p1_hom = np.append(pointA, 1.0)            
            transformed = model @ p1_hom                       
            distance = np.linalg.norm(transformed[:3] - pointB)

            if distance < max_epsilon:
                inliers.append((idxA, pointA, view_a, idxB, pointB, view_b))
        
        ir = len(inliers) / len(candidates)
        # cost = max(0.0, min(1.0, 1.0 - ir))
        is_good = len(inliers) >= min_num_inliers and ir > min_inlier_ratio
        
        return is_good, inliers
    
    def compute_ransac(self, candidates):
        max_epsilon = 5.0
        min_inlier_ratio = 0.1
        num_iterations = 1000
        model_min_matches = 4
        regularization_weight = 1.0

        best_inliers = []
        max_inliers = 0

        for _ in range(num_iterations):
            indices = random.sample(range(len(candidates)), model_min_matches)
            min_matches = [candidates[i] for i in indices]

            try:
                point_pairs = [(m[1], m[4]) for m in min_matches]
                affine_model = self.fit_affine_model(point_pairs)
                rigid_model = self.fit_rigid_model(point_pairs)
                regularized_model = (1 - regularization_weight) * affine_model + regularization_weight * rigid_model
            except Exception:
                continue  # skip degenerate samples

            num_inliers = 0
            is_good, tmp_inliers = self.test(candidates, regularized_model, max_epsilon, min_inlier_ratio, model_min_matches)

            while is_good and num_inliers < len(tmp_inliers):
                num_inliers = len(tmp_inliers)
                point_pairs = [(i[1], i[4]) for i in tmp_inliers]
                affine_model = self.fit_affine_model([(i[0], i[1]) for i in point_pairs])
                rigid_model = self.fit_rigid_model([(i[0], i[1]) for i in point_pairs])
                regularized_model = (1 - regularization_weight) * affine_model + regularization_weight * rigid_model
                is_good, tmp_inliers = self.test(candidates, regularized_model, max_epsilon, min_inlier_ratio, model_min_matches)

            if len(tmp_inliers) > max_inliers:
                best_inliers = tmp_inliers
                max_inliers = len(tmp_inliers)

        return best_inliers

    # This function defaults to 1 given current params but likely not if we apply a transformation from below
    # The function below is input to this function
    def normalization_factor(self, matches):
        return 1
    
    # This seems to be the place where we apply the type of transformation (rigid, affine)
    def fit_matches(self):
        pass

    def descriptor_distance(self, desc_a, desc_b):
        subsets_a = desc_a["subsets"]  # (20, 3, D)
        subsets_b = desc_b["subsets"]  # (20, 3, D)

        # TODO - fit_matches would go here to transform points for rigid, affine
        # subsets_a, subsets_b = self.fit_matches(subsets_a, subsets_b)

        # Broadcast all pairwise subset differences
        diffs = subsets_a[:, None, :, :] - subsets_b[None, :, :, :]  # (20, 20, 3, D)
        sq_diffs = np.sum(diffs ** 2, axis=-1)                       # (20, 20, 3)
        similarities = np.mean(sq_diffs, axis=-1)                    # (20, 20)

        return np.min(similarities)
    
    def create_simple_point_descriptors(self, tree, points_array, num_required_neighbors, matcher):
        k = num_required_neighbors + 1  # +1 to skip self
        distances, indices = tree.query(points_array, k=k)

        descriptors = []
        for i, basis_point in enumerate(points_array):
            try:
                neighbor_idxs = indices[i][1:]
                neighbors = points_array[neighbor_idxs]

                # Validate dimensionality
                # num_dimensions = basis_point.shape[0]
                # for n in neighbors:
                #     if n.shape[0] != num_dimensions:
                #         raise ValueError("Neighbor has different dimensionality than basis point.")

                # # Compute relative coordinates
                # descriptor_points = []
                # for neighbor in neighbors:
                #     relative = neighbor - basis_point  # relative local coordinates
                #     descriptor_points.append({
                #         "relative": relative,
                #         "absolute": neighbor
                #     })

                relative_vectors = neighbors - basis_point     

                # Final descriptor representation (as dict)
                descriptor = {
                    "point_index": i,
                    "point": basis_point,
                    "neighbors": neighbors,
                    "relative_descriptors": relative_vectors,
                    "matcher": matcher,
                    "subsets": np.stack([relative_vectors[list(combo)] for combo in matcher["neighbors"]])
                }

                descriptors.append(descriptor)

            except Exception as e:
                print(f"⚠️ Failed to create descriptor for point {i}: {e}")

        return descriptors

    def get_candidates(self, points_a, points_b, view_a, view_b):
        difference_threshold = 3.4028235e+38
        max_value = float("inf")

        # --- KD Trees ---
        points_a_array = np.array(points_a)
        points_b_array = np.array(points_b)
        tree_a = KDTree(points_a_array)
        tree_b = KDTree(points_b_array)

        # --- SubsetMatcher ---
        subset_size = self.num_neighbors
        total_neighbors = self.num_neighbors + self.redundancy  
        neighbor_indices_combinations = list(itertools.combinations(range(total_neighbors), subset_size))
        num_combinations = len(neighbor_indices_combinations)
        num_matchings = num_combinations * num_combinations
        matcher = {
            "subset_size": subset_size,
            "num_neighbors": total_neighbors,
            "neighbors": neighbor_indices_combinations,
            "num_combinations": num_combinations,
            "num_matchings": num_matchings
        }

        # --- Descriptors ---
        descriptors_a = self.create_simple_point_descriptors(tree_a, points_a_array, self.num_required_neighbors, matcher)
        descriptors_b = self.create_simple_point_descriptors(tree_b, points_b_array, self.num_required_neighbors, matcher)

        # --- Descriptor Matching ---
        correspondence_candidates = []
        
        for desc_a in descriptors_a:  
            best_difference = float("inf")
            second_best_difference = float("inf")  
            best_match = None
            second_best_match = None

            for desc_b in descriptors_b:
                difference = self.descriptor_distance(desc_a, desc_b)

                if difference < second_best_difference:
                    second_best_difference = difference
                    second_best_match = desc_b

                    if second_best_difference < best_difference:
                        tmp_diff = second_best_difference
                        tmp_match = second_best_match
                        second_best_difference = best_difference
                        second_best_match = best_match
                        best_difference = tmp_diff
                        best_match = tmp_match
            
            # --- Lowe's Test ---
            if best_difference < difference_threshold and best_difference * self.significance < second_best_difference and second_best_difference is not max_value:
                correspondence_candidates.append((
                    desc_a['point_index'],        
                    desc_a['point'],               
                    view_a,
                    best_match['point_index'],    
                    best_match['point'],            
                    view_b
                ))

        return correspondence_candidates
    
    def run(self):
        all_results = []
        for pointsA, pointsB, viewA_str, viewB_str in self.process_pairs:
            candidates = self.get_candidates(pointsA, pointsB, viewA_str, viewB_str)
            inliers = self.compute_ransac(candidates)
            
            percent = 100.0 * len(inliers) / len(candidates)
            print(f"✅ RANSAC inlier percentage: {percent:.1f}% ({len(inliers)} of {len(candidates)})")

            all_results.extend(inliers if inliers else [])
        
        return all_results

    # --- DEBUG --- This function was replaced below

    # def create_candidates(self, desc_a, desc_b, neighbors):
    #     match_list = []

    #     for a in range(4):
    #         for b in range(4):
    #             matches = []
    #             subset_size = 3

    #             for i in range(subset_size):
    #                 index_a = neighbors[a][i]
    #                 index_b = neighbors[b][i]

    #                 point_match = (desc_a['relative_descriptors'][index_a]['relative'], desc_b['relative_descriptors'][index_b]['relative'])
    #                 matches.append(point_match)

    #             match_list.append(matches)
        
    #     return match_list


    # --- DEBUG --- These three functions were consolidated into one above

    # def similarity_measure(self, matches):
    #     num_dims = 3
    #     diff = 0

    #     for p1, p2 in matches:
    #         diff += np.sum((p1 - p2) ** 2)

    #     return diff / num_dims
    
    # def create_candidates(self, desc_a, desc_b):
    #     match_list = []

    #     for subset_a in desc_a["subsets"]:
    #         for subset_b in desc_b["subsets"]:
    #             match = list(zip(subset_a, subset_b))  # Each is a (vec_a, vec_b) pair
    #             match_list.append(match)

    #     return match_list
    
    # def descriptor_distance(self, desc_a, desc_b):
    #     match_list = self.create_candidates(desc_a, desc_b)
    #     best_similarity = float("inf")
    #     # best_point_match_set = None

    #     for matches in match_list:
    #         # TODO - Add fit matches here for transformations (rigid, affine) and pass it into normalization factor
    #         similarity_measure = self.similarity_measure(matches)
    #         normalization_factor = self.normalization_factor(matches) 
    #         similarity = similarity_measure * normalization_factor

    #         if similarity < best_similarity:
    #             best_similarity = similarity
    #             # best_point_match_set = matches
        
    #     return best_similarity