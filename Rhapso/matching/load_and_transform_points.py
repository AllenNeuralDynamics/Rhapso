import numpy as np
import json
import fsspec
import pandas as pd

"""
Load and Transform Points loads interest points from n5 and transforms them into global space
"""

class LoadAndTransformPoints:
    def __init__(self, data_global, view_registrations, label, n5_output_path):
        self.data_global = data_global
        self.view_registrations = view_registrations
        self.label = label
        self.n5_output_path = n5_output_path
    
    def transform_interest_points(self, points, transformation_matrix):
        """
        Transform interest points using the given transformation matrix
        """
        if len(points) == 0: return []
        
        # Convert points to homogeneous coordinates (add 1 as 4th coordinate)
        homogeneous_points = np.column_stack([points, np.ones(len(points))])
        
        # Apply transformation: result = matrix @ points.T, then transpose back
        transformed_homogeneous = (transformation_matrix @ homogeneous_points.T).T
        
        # Convert back to 3D coordinates (remove homogeneous coordinate)
        transformed_points = transformed_homogeneous[:, :3]

        return transformed_points.astype(np.float64)

    def _parse_affine_matrix(self, affine_text):
        """
        Parse affine transformation matrix from text string
        """
        try:
            # Split the affine text into float values
            values = [float(x) for x in affine_text.strip().split()]
            
            if len(values) != 12:
                raise ValueError(f"Expected 12 values for 3x4 affine matrix, got {len(values)}")
            
            # Reshape into 3x4 matrix (row-major order)
            matrix_3x4 = np.array(values).reshape(3, 4)
            
            # Convert to 4x4 homogeneous matrix by adding bottom row [0, 0, 0, 1]
            matrix_4x4 = np.eye(4)
            matrix_4x4[:3, :] = matrix_3x4
            
            return matrix_4x4
            
        except Exception as e:
            print(f"❌ Error parsing affine matrix from '{affine_text}': {e}")
            # Return identity matrix as fallback
            return np.eye(4)
        
    def get_transformation_matrix(self, view_id):
        """
        Compose all affine ViewTransforms for a given view (timepoint, setup)
        """
        try:
            transforms = self.view_registrations.get(view_id, [])
            if not transforms:
                print(f"⚠️ No transforms found for view {view_id}, using identity matrix")
                return np.eye(4)

            final_matrix = np.eye(4)

            for i, transform in enumerate(transforms):
                affine_str = transform.get("affine")
                if not affine_str:
                    print(f"⚠️ No affine string in transform {i+1} for view {view_id}")
                    continue

                values = [float(x) for x in affine_str.strip().split()]
                if len(values) != 12:
                    raise ValueError(f"Transform {i+1} in view {view_id} has {len(values)} values, expected 12.")

                matrix3x4 = np.array(values).reshape(3, 4)
                matrix4x4 = np.eye(4)
                matrix4x4[:3, :4] = matrix3x4

                final_matrix = final_matrix @ matrix4x4

            return final_matrix

        except Exception as e:
            print(f"❌ Error in get_transformation_matrix for view {view_id}: {e}")
            raise

    def get_bounding_boxes(self, M, dims):
        """
        Compute world-space AABB (min/max corners) of a voxel-aligned box
        """
        M = np.asarray(M, float)
        if M.shape == (3, 4):
            M = np.vstack([M, [0.0, 0.0, 0.0, 1.0]])

        # interval mins/maxes
        t0 = 0.0; t1 = 0.0; t2 = 0.0
        s0 = float(dims[0]) - 1.0
        s1 = float(dims[1]) - 1.0
        s2 = float(dims[2]) - 1.0

        A = M[:3, :3]
        tx, ty, tz = M[0, 3], M[1, 3], M[2, 3]

        # row 0
        tt0 = A[0,0]*t0 + A[0,1]*t1 + A[0,2]*t2 + tx
        rMin0 = rMax0 = tt0
        rMin0 += s0*A[0,0] if A[0,0] < 0 else 0.0; rMax0 += 0.0 if A[0,0] < 0 else s0*A[0,0]
        rMin0 += s1*A[0,1] if A[0,1] < 0 else 0.0; rMax0 += 0.0 if A[0,1] < 0 else s1*A[0,1]
        rMin0 += s2*A[0,2] if A[0,2] < 0 else 0.0; rMax0 += 0.0 if A[0,2] < 0 else s2*A[0,2]

        # row 1
        tt1 = A[1,0]*t0 + A[1,1]*t1 + A[1,2]*t2 + ty
        rMin1 = rMax1 = tt1
        rMin1 += s0*A[1,0] if A[1,0] < 0 else 0.0; rMax1 += 0.0 if A[1,0] < 0 else s0*A[1,0]
        rMin1 += s1*A[1,1] if A[1,1] < 0 else 0.0; rMax1 += 0.0 if A[1,1] < 0 else s1*A[1,1]
        rMin1 += s2*A[1,2] if A[1,2] < 0 else 0.0; rMax1 += 0.0 if A[1,2] < 0 else s2*A[1,2]

        # row 2
        tt2 = A[2,0]*t0 + A[2,1]*t1 + A[2,2]*t2 + tz
        rMin2 = rMax2 = tt2
        rMin2 += s0*A[2,0] if A[2,0] < 0 else 0.0; rMax2 += 0.0 if A[2,0] < 0 else s0*A[2,0]
        rMin2 += s1*A[2,1] if A[2,1] < 0 else 0.0; rMax2 += 0.0 if A[2,1] < 0 else s1*A[2,1]
        rMin2 += s2*A[2,2] if A[2,2] < 0 else 0.0; rMax2 += 0.0 if A[2,2] < 0 else s2*A[2,2]

        rMin = np.array([rMin0, rMin1, rMin2], float)
        rMax = np.array([rMax0, rMax1, rMax2], float)
        return rMin, rMax

    def bounding_boxes(self, M, dims):
        """
        Compute an integer, padded axis-aligned bounding box from the real-valued bounds
        """
        rMin, rMax = self.get_bounding_boxes(M, dims['size'])
        min_i = np.rint(rMin).astype(int) - 1
        max_i = np.rint(rMax).astype(int) + 1
        return (min_i.tolist(), max_i.tolist())
    
    def transform_matrices(self, view): 
        """
        Compose the per-view 4x4 world transform by chaining all affine models in order
        """
        M = np.eye(4, dtype=float)   
        for model in self.data_global['viewRegistrations'][view]:
            vals = np.fromstring(str(model['affine']).replace(',', ' '), sep=' ', dtype=float)
            T = np.eye(4, dtype=float); T[:3, :4] = vals.reshape(3, 4)  
            M = M @ T
        return M
    
    def overlaps(self, bba, bbb):
        """
        Boolean check if two axis-aligned boxes overlap in every dimension
        """
        (minA, maxA) = bba
        (minB, maxB) = bbb
        for d in range(len(minA)):  
            if ((minA[d] <= minB[d] and maxA[d] <= minB[d]) or
                (minA[d] >= maxB[d] and maxA[d] >= maxB[d])):
                return False
        return True

    def overlap(self, view_a, dims_a, view_b, dims_b):
        """
        Build each view's transform, derive their axis-aligned bounding boxes, then test for intersection
        """
        ma = self.transform_matrices(view_a)
        mb = self.transform_matrices(view_b)

        bba = self.bounding_boxes(ma, dims_a)
        bbb = self.bounding_boxes(mb, dims_b)

        return self.overlaps(bba, bbb)   
    
    def load_interest_points_from_path(self, base_path, point_key):
        """
        Load interest points from the Parquet/JSON alignment store.

        point_key format:
            timepoint/setup/label
        """
        try:
            base_path = str(base_path).rstrip("/")

            if not hasattr(self, "_manifest_points"):
                manifest_path = f"{base_path}/manifest.json"

                with fsspec.open(manifest_path, "r") as f:
                    manifest = json.load(f)

                self._manifest_points = manifest["points"]

            if point_key not in self._manifest_points:
                print(f"⚠️ No interest points found in manifest for {point_key}")
                return []

            point_path = f"{base_path}/{self._manifest_points[point_key]}"

            df = pd.read_parquet(point_path, engine="pyarrow")

            if len(df) == 0:
                return []

            return df[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=False)

        except Exception as e:
            print(f"❌ Failed to load interest points from parquet for {point_key}: {e}")
            return []

    def get_transformed_points(self, view_id):
        """
        Retrieve and transform interest points for a given view
        """
        timepoint, setup = view_id

        point_key = f"{int(timepoint)}/{int(setup)}/{self.label}"
        full_path = self.n5_output_path

        raw_points = self.load_interest_points_from_path(full_path, point_key)

        if len(raw_points) == 0:
            return []

        transform = self.get_transformation_matrix(view_id)
        transformed_points = self.transform_interest_points(raw_points, transform)

        return transformed_points
    
    def load_and_transform_points(self, viewA, viewB):
        """
        Process a single matching task
        """
        try:
            # Retrieve and transform interest points for both views
            viewA_str = f"(tpId={viewA[0]}, setupId={viewA[1]})"
            viewB_str = f"(tpId={viewB[0]}, setupId={viewB[1]})"   
            pointsA = self.get_transformed_points(viewA)
            pointsB = self.get_transformed_points(viewB)
            return pointsA, pointsB, viewA_str, viewB_str
            
        except Exception:
            print(f"❌ ERROR: Failed in process_matching_task for views {viewA} and {viewB}")
            return []
    
    def run(self, viewA, viewB):
        """
        Executes the entry point of the script.
        """
        return self.load_and_transform_points(viewA, viewB)
