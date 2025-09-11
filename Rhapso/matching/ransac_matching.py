import numpy as np
from sklearn.neighbors import KDTree
import itertools
import random
from scipy.linalg import eigh
import zarr
from bioio import BioImage
import bioio_tifffile
import dask.array as da
import s3fs
import copy
import re

"""
Utility class to find interest point match candidates and filter with ransac 
"""

class CustomBioImage(BioImage):
    def standard_metadata(self):
        pass
    
    def scale(self):
        pass
    
    def time_interval(self):
        pass

class RansacMatching:
    def __init__(self, data_global, num_neighbors, redundancy, significance, num_required_neighbors, match_type, 
                 max_epsilon, min_inlier_ratio, num_iterations, model_min_matches, regularization_weight, 
                 search_radius, view_registrations, input_type, image_file_prefix):
        self.data_global = data_global
        self.num_neighbors = num_neighbors
        self.redundancy = redundancy
        self.significance = significance
        self.num_required_neighbors = num_required_neighbors
        self.match_type = match_type
        self.max_epsilon = max_epsilon
        self.min_inlier_ratio = min_inlier_ratio
        self.num_iterations = num_iterations
        self.model_min_matches = model_min_matches
        self.regularization_weight = regularization_weight
        self.search_radius = search_radius
        self.view_registrations = view_registrations
        self.input_type = input_type
        self.image_file_prefix = image_file_prefix
    
    def filter_inliers(self, candidates, initial_model):
        max_trust = 4.0
            
        if len(candidates) < self.model_min_matches:
            return []
        
        model_copy = copy.deepcopy(initial_model)
        inliers = candidates[:]
        temp = []
        
        while True:
            temp = copy.deepcopy(inliers)
            num_inliers = len(inliers)
    
            point_pairs = [(m[1], m[5]) for m in inliers]
            model_copy = self.model_regularization(point_pairs)
            
            # Apply model and collect errors
            errors = []
            for match in temp:
                p1 = np.array(match[1])
                p2 = np.array(match[4])
                p1_h = np.append(p1, 1.0)
                p1_trans = model_copy @ p1_h
                error = np.linalg.norm(p1_trans[:3] - p2)
                errors.append(error)
            
            median_error = np.median(errors)
            threshold = median_error * max_trust
            
            # Filter based on threshold
            inliers = [m for m, err in zip(temp, errors) if err <= threshold]
            
            if num_inliers <= len(inliers):
                break
        
        if num_inliers < self.model_min_matches:
            return []

        return inliers 
    
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

        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            # If A is not invertible, use the pseudo-inverse
            A_inv = np.linalg.pinv(A)

        M = A_inv @ B  # 3x3 transformation matrix

        t = qc - M @ pc  # translation

        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = M
        affine_matrix[:3, 3] = t

        return affine_matrix
    
    def test(self, candidates, model, max_epsilon, min_inlier_ratio, min_num_inliers):
        inliers = []
        for idxA, pointA, view_a, label_a, idxB, pointB, view_b, label_b in candidates:
            p1_hom = np.append(pointA, 1.0)            
            transformed = model @ p1_hom                       
            distance = np.linalg.norm(transformed[:3] - pointB)

            if distance < max_epsilon:
                inliers.append((idxA, pointA, view_a, label_a, idxB, pointB, view_b, label_b))
        
        ir = len(inliers) / len(candidates)
        is_good = len(inliers) >= min_num_inliers and ir > min_inlier_ratio
        
        return is_good, inliers
    
    def regularize_models(self, affine, rigid):
        alpha=0.1
        l1 = 1.0 - alpha

        def to_array(model):
            return [
                model['m00'], model['m01'], model['m02'], model['m03'], 
                model['m10'], model['m11'], model['m12'], model['m13'],  
                model['m20'], model['m21'], model['m22'], model['m23'], 
            ]

        afs = to_array(affine)
        bfs = to_array(rigid)
        rfs = [l1 * a + alpha * b for a, b in zip(afs, bfs)]

        keys = [
            'm00', 'm01', 'm02', 'm03',
            'm10', 'm11', 'm12', 'm13',
            'm20', 'm21', 'm22', 'm23',
        ]
        regularized = dict(zip(keys, rfs))

        return regularized
    
    def model_regularization(self, point_pairs):
        if self.match_type == "rigid":
            regularized_model = self.fit_rigid_model(point_pairs)
        elif self.match_type == "affine" or self.match_type == "split-affine":
            rigid_model = self.fit_rigid_model(point_pairs)
            affine_model = self.fit_affine_model(point_pairs)
            regularized_model = (1 - self.regularization_weight) * affine_model + self.regularization_weight * rigid_model
        else:
            raise SystemExit(f"Unsupported match type: {self.match_type}")
        
        return regularized_model
    
    def compute_ransac(self, candidates):
        best_inliers = []
        max_inliers = 0
        best_model = None

        if len(candidates) < self.model_min_matches:
            return [], None
        
        for _ in range(self.num_iterations):
            indices = random.sample(range(len(candidates)), self.model_min_matches)
            min_matches = [candidates[i] for i in indices]

            try:
                point_pairs = [(m[1], m[5]) for m in min_matches]
                regularized_model = self.model_regularization(point_pairs)
            except Exception as e:
                print(e)

            num_inliers = 0
            is_good, tmp_inliers = self.test(candidates, regularized_model, self.max_epsilon, self.min_inlier_ratio, self.model_min_matches)

            while is_good and num_inliers < len(tmp_inliers):
                num_inliers = len(tmp_inliers)
                point_pairs = [(i[1], i[5]) for i in tmp_inliers]
                regularized_model = self.model_regularization(point_pairs)
                is_good, tmp_inliers = self.test(candidates, regularized_model, self.max_epsilon, self.min_inlier_ratio, self.model_min_matches)

            if len(tmp_inliers) > max_inliers:
                best_inliers = tmp_inliers
                max_inliers = len(tmp_inliers)
                best_model = regularized_model

        return best_inliers, best_model

    def create_candidates(self, desc_a, desc_b):
        match_list = []
        
        for a in range(1):
            for b in range(1):

                matches = []
                for i in range(3):
                    point_match = (desc_a['relative_descriptors'][i], desc_b['relative_descriptors'][i])
                    matches.append(point_match)

                match_list.append(matches)
        
        return match_list
    
    def descriptor_distance(self, desc_a, desc_b):
        matches_list = self.create_candidates(desc_a, desc_b)

        best_similarity = float("inf")
        best_match_set = None

        for matches in matches_list:
            try:
                points_a = np.array([pa for pa, _ in matches])
                points_b = np.array([pb for _, pb in matches])

                squared_diff_sum = np.sum((points_a - points_b) ** 2)
                similarity = squared_diff_sum / points_a.shape[1] 

                if similarity < best_similarity:
                    best_similarity = similarity
                    best_match_set = matches

            except Exception:
                continue

        return best_similarity
    
    def create_simple_point_descriptors(self, tree, points_array, idx, num_required_neighbors, matcher):
        k = num_required_neighbors + 1 
        if len(points_array) < k:
            return []
        
        _, indices = tree.query(points_array, k=k)

        descriptors = []
        for i, basis_point in enumerate(points_array):
            try:
                neighbor_idxs = indices[i][1:]
                neighbors = points_array[neighbor_idxs]
                
                if len(neighbors) == num_required_neighbors:
                    idx_sets = [tuple(range(num_required_neighbors))]   
                elif len(neighbors) > num_required_neighbors:
                    idx_sets = matcher["neighbors"] 

                relative_vectors = neighbors - basis_point     

                # Final descriptor representation (as dict)
                descriptor = {
                    "point_index": idx[i],
                    "point": basis_point,
                    "neighbors": neighbors,
                    "relative_descriptors": relative_vectors,
                    "matcher": matcher,
                    "subsets": np.stack([neighbors[list(c)] for c in idx_sets])
                }

                descriptors.append(descriptor)

            except Exception as e:
                raise

        return descriptors

    def get_candidates(self, points_a, points_b, view_a, view_b, label):
        difference_threshold = 3.4028235e+38
        max_value = float("inf")
        
        # -- Get Points and Indexes
        idx_a, coords_a = zip(*points_a)
        idx_b, coords_b = zip(*points_b)
        points_a_array = np.array(coords_a)
        points_b_array = np.array(coords_b)
        
        # --- KD Trees ---
        tree_a = KDTree(points_a_array)
        tree_b = KDTree(points_b_array)

        # --- Subset Matcher ---
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
        descriptors_a = self.create_simple_point_descriptors(tree_a, points_a_array, idx_a, self.num_required_neighbors, matcher)
        descriptors_b = self.create_simple_point_descriptors(tree_b, points_b_array, idx_b, self.num_required_neighbors, matcher)

        # --- Descriptor Matching ---
        correspondence_candidates = []

        out_of_radius = 0
        passed_lowes = 0
        first_if = 0
        second_if = 0
        
        for desc_a in descriptors_a:  
            best_difference = float("inf")
            second_best_difference = float("inf")  
            best_match = None
            second_best_match = None

            for desc_b in descriptors_b:

                if np.linalg.norm(desc_a['point'] - desc_b['point']) > self.search_radius:
                    out_of_radius += 1
                    continue
                
                difference = self.descriptor_distance(desc_a, desc_b)

                if difference < second_best_difference:
                    second_best_difference = difference
                    second_best_match = desc_b
                    first_if += 1

                    if second_best_difference < best_difference:
                        tmp_diff = second_best_difference
                        tmp_match = second_best_match
                        second_best_difference = best_difference
                        second_best_match = best_match
                        best_difference = tmp_diff
                        best_match = tmp_match
                        second_if += 1
            
            # --- Lowe's Test ---
            if best_difference < difference_threshold and best_difference * self.significance < second_best_difference and second_best_difference != max_value:
                correspondence_candidates.append((
                    desc_a['point_index'],        
                    desc_a['point'],               
                    view_a,
                    label,
                    best_match['point_index'],    
                    best_match['point'],            
                    view_b,
                    label
                ))
                passed_lowes += 1

        # print(f"out of range: {out_of_radius}, first if: {first_if}, second if: {second_if}, passed lowes: {passed_lowes}")
        return correspondence_candidates
    
    def get_tile_dims(self, view1):
        stripped = view1.strip("()")
        parts = stripped.split(", ")
        tp_id = int(parts[0].split("=")[1])
        setup_id = int(parts[1].split("=")[1])
        
        image_loader = self.data_global.get('imageLoader', {})

        # Loop through all view entries in the image loader
        for entry in image_loader:
            entry_setup = int(entry.get('view_setup', -1))
            entry_tp = int(entry.get('timepoint', -1))

            if entry_setup == setup_id and entry_tp == tp_id:
                file_path = self.image_file_prefix + entry.get('file_path')
                if self.input_type == "tiff":
                    img = CustomBioImage(file_path, reader=bioio_tifffile.Reader)
                    dask_array = img.get_dask_stack()[0, 0, 0, :, :, :]
                    shape = dask_array.shape
                
                elif self.input_type == "zarr":
                    s3 = s3fs.S3FileSystem(anon=False)  
                    full_path = f"{file_path}/0"
                    store = s3fs.S3Map(root=full_path, s3=s3)
                    zarr_array = zarr.open(store, mode='r')
                    dask_array = da.from_zarr(zarr_array)[0, 0, :, :, :]
                    shape = dask_array.shape
        
                return shape[::-1]  
         
    def invert_transformation_matrix(self, view_2):
        """
        Compose and invert all ViewTransforms for the given view key (timepoint, setup).
        """
        stripped = view_2.strip("()")
        parts = stripped.split(", ")
        tp_id = int(parts[0].split("=")[1])
        setup_id = int(parts[1].split("=")[1])
        view_key = (tp_id, setup_id)

        # Get all transforms for this view
        transforms = self.view_registrations.get(view_key, [])
        if not transforms:
            raise ValueError(f"No transforms found for view {view_key}")

        final_matrix = np.eye(4)

        for i, transform in enumerate(transforms):
            affine_str = transform.get("affine")
            if not affine_str:
                continue

            values = [float(x) for x in affine_str.strip().split()]
            if len(values) != 12:
                raise ValueError(f"Transform {i+1} in view {view_key} has {len(values)} values, expected 12.")

            matrix3x4 = np.array(values).reshape(3, 4)
            matrix4x4 = np.eye(4)
            matrix4x4[:3, :4] = matrix3x4

            # Combine with running matrix
            final_matrix = final_matrix @ matrix4x4

        # Return the inverse
        return np.linalg.inv(final_matrix)

    def filter_for_overlapping_points(self, points_a, points_b, view_a, view_b):
        points_a = list(enumerate(points_a))  
        points_b = list(enumerate(points_b))

        if not points_a or not points_b:
            return [], []

        # Check points_a against view_b's interval
        overlapping_a = []
        tinv_b = self.invert_transformation_matrix(view_b)

        view_b_key = tuple(map(int, re.findall(r'\d+', view_b)))
        dim_b = self.data_global['viewSetup']['byId'][view_b_key[1]]
        interval_b = {'min': (0, 0, 0), 'max': dim_b['size']}

        for i in reversed(range(len(points_a))):
            idx, point = points_a[i]
            p_h = np.append(point, 1.0)
            transformed = tinv_b @ p_h
            x, y, z = transformed[:3]
            x_min, y_min, z_min = interval_b['min']
            x_max, y_max, z_max = interval_b['max']

            if x_min <= x < x_max and y_min <= y < y_max and z_min <= z < z_max:
                overlapping_a.append((idx, point))
                del points_a[i]

        # Check points_b against view_a's interval
        overlapping_b = []
        tinv_a = self.invert_transformation_matrix(view_a)

        view_a_key = tuple(map(int, re.findall(r'\d+', view_a)))
        dim_a = self.data_global['viewSetup']['byId'][view_a_key[1]]
        interval_a = {'min': (0, 0, 0), 'max': dim_a['size']}

        for i in reversed(range(len(points_b))):
            idx, point = points_b[i]
            p_h = np.append(point, 1.0)
            transformed = tinv_a @ p_h
            x, y, z = transformed[:3]
            x_min, y_min, z_min = interval_a['min']
            x_max, y_max, z_max = interval_a['max']

            if x_min <= x < x_max and y_min <= y < y_max and z_min <= z < z_max:
                overlapping_b.append((idx, point))
                del points_b[i]

        return overlapping_a, overlapping_b