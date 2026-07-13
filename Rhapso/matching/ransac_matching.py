import numpy as np
from sklearn.neighbors import KDTree
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
                 inlier_threshold, min_inlier_ratio, num_iterations, model_min_inliers, regularization_weight, 
                 search_radius, view_registrations, input_type, image_file_prefix, ransac_sample_size):
        self.data_global = data_global
        self.num_neighbors = num_neighbors
        self.redundancy = redundancy
        self.significance = significance
        self.num_required_neighbors = num_required_neighbors
        self.match_type = match_type
        self.inlier_threshold = inlier_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.num_iterations = num_iterations
        self.model_min_inliers = model_min_inliers
        self.regularization_weight = regularization_weight
        self.search_radius = search_radius
        self.view_registrations = view_registrations
        self.input_type = input_type
        self.image_file_prefix = image_file_prefix
        self.ransac_sample_size = ransac_sample_size
    
    def filter_inliers(self, candidates, initial_model):
        max_trust = 4.0
            
        if len(candidates) < self.model_min_inliers:
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
                p2 = np.array(match[5])
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
        
        if num_inliers < self.model_min_inliers:
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
        """
        Fit a 3x4 affine transform such that:
            Q ≈ M @ P + t
        where P, Q are 3D column vectors (but stored here as row vectors).
        """
        matches = np.asarray(matches)          # shape (N, 2, 3)
        P = matches[:, 0]                      # source points, shape (N, 3)
        Q = matches[:, 1]                      # target points, shape (N, 3)

        # Uniform weights for now (kept in case you add non-uniform later)
        weights = np.ones(P.shape[0], dtype=float)

        # Weighted centroids
        pc = np.average(P, axis=0, weights=weights)
        qc = np.average(Q, axis=0, weights=weights)

        # Centered coordinates
        P_centered = P - pc
        Q_centered = Q - qc

        # Weighted least squares: scale rows by sqrt(weight)
        sqrt_w = np.sqrt(weights)[:, None]           # (N, 1)
        P_w = P_centered * sqrt_w                    # (N, 3)
        Q_w = Q_centered * sqrt_w                    # (N, 3)

        # Solve P_w @ M^T ≈ Q_w  → M_T is 3x3, then transpose
        M_T, *_ = np.linalg.lstsq(P_w, Q_w, rcond=None)
        M = M_T.T

        # Translation so that M @ pc ≈ qc
        t = qc - M @ pc

        # Pack into 4x4 affine matrix
        affine_matrix = np.eye(4, dtype=float)
        affine_matrix[:3, :3] = M
        affine_matrix[:3, 3] = t

        return affine_matrix
    
    def test(self, candidates, model, inlier_threshold, min_inlier_ratio, min_num_inliers):
        inliers = []
        for idxA, pointA, view_a, label_a, idxB, pointB, view_b, label_b in candidates:
            p1_hom = np.append(pointA, 1.0)            
            transformed = model @ p1_hom                       
            distance = np.linalg.norm(transformed[:3] - pointB)

            if distance < inlier_threshold:
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
            return self.fit_rigid_model(point_pairs)

        if self.match_type == "affine" or self.match_type == "split-affine":
            rigid_model = self.fit_rigid_model(point_pairs)
            affine_model = self.fit_affine_model(point_pairs)
            w = self.regularization_weight
            return (1.0 - w) * affine_model + w * rigid_model

        raise SystemExit(f"Unsupported match type: {self.match_type}")

    def compute_ransac(self, candidates):
        best_inliers = []
        best_model = None
        max_inliers = 0

        if len(candidates) < self.model_min_inliers:
            return [], None

        rng = random

        for _ in range(self.num_iterations):
            indices = rng.sample(
                range(len(candidates)),
                self.ransac_sample_size,
            )
            sample = [candidates[i] for i in indices]

            try:
                point_pairs = [
                    (match[1], match[5])
                    for match in sample
                ]
                model = self.model_regularization(point_pairs)
            except Exception:
                continue

            is_good, inliers = self.test(
                candidates,
                model,
                self.inlier_threshold,
                self.min_inlier_ratio,
                self.model_min_inliers,
            )

            # The initial model did not satisfy the configured RANSAC gates.
            if not is_good:
                continue

            previous_count = 0

            # Refit using all current inliers until the inlier set stops growing.
            while is_good and len(inliers) > previous_count:
                previous_count = len(inliers)

                try:
                    point_pairs = [
                        (match[1], match[5])
                        for match in inliers
                    ]
                    model = self.model_regularization(point_pairs)
                except Exception:
                    is_good = False
                    break

                is_good, refined_inliers = self.test(
                    candidates,
                    model,
                    self.inlier_threshold,
                    self.min_inlier_ratio,
                    self.model_min_inliers,
                )

                if not is_good:
                    break

                inliers = refined_inliers

            # Never save a model that failed after refinement.
            if not is_good:
                continue

            if len(inliers) > max_inliers:
                best_inliers = inliers
                max_inliers = len(inliers)
                best_model = model

        return best_inliers, best_model
    
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
        if len(points_a) == 0 or len(points_b) == 0:
            return []

        idx_a, coords_a = zip(*points_a)
        idx_b, coords_b = zip(*points_b)

        points_a_array = np.asarray(coords_a, dtype=np.float64)
        points_b_array = np.asarray(coords_b, dtype=np.float64)

        k = self.num_required_neighbors + 1

        if len(points_a_array) < k or len(points_b_array) < k:
            return []

        tree_a = KDTree(points_a_array)
        tree_b = KDTree(points_b_array)

        # Build the same local nearest-neighbor descriptors as before.
        _, nbr_a = tree_a.query(
            points_a_array,
            k=k,
            return_distance=True,
        )
        _, nbr_b = tree_b.query(
            points_b_array,
            k=k,
            return_distance=True,
        )

        relative_a = (
            points_a_array[nbr_a[:, 1:]]
            - points_a_array[:, None, :]
        )
        relative_b = (
            points_b_array[nbr_b[:, 1:]]
            - points_b_array[:, None, :]
        )

        # Only compare A points to B points inside the existing search radius.
        nearby_b_for_a = tree_b.query_radius(
            points_a_array,
            r=self.search_radius,
            return_distance=False,
        )

        correspondence_candidates = []

        for a_pos, b_positions in enumerate(nearby_b_for_a):
            # Lowe's test requires a best and second-best candidate.
            if b_positions.size < 2:
                continue

            # Match original B traversal order and deterministic tie behavior.
            b_positions = np.sort(b_positions)

            # Compare this A descriptor against all nearby B descriptors.
            delta = relative_b[b_positions] - relative_a[a_pos]

            differences = (
                np.einsum(
                    "nkd,nkd->n",
                    delta,
                    delta,
                )
                / 3.0
            )

            # Find the best and second-best without sorting every score.
            best_local = int(np.argmin(differences))

            second_differences = differences.copy()
            second_differences[best_local] = np.inf
            second_local = int(np.argmin(second_differences))

            best_difference = float(differences[best_local])
            second_best_difference = float(
                differences[second_local]
            )

            if not (
                np.isfinite(best_difference)
                and np.isfinite(second_best_difference)
            ):
                continue

            # Same Lowe/significance gate as the original matcher.
            if (
                best_difference * self.significance
                < second_best_difference
            ):
                b_pos = int(b_positions[best_local])

                correspondence_candidates.append(
                    (
                        idx_a[a_pos],
                        points_a_array[a_pos],
                        view_a,
                        label,
                        idx_b[b_pos],
                        points_b_array[b_pos],
                        view_b,
                        label,
                    )
                )

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