import zarr
import numpy as np
import s3fs
from itertools import combinations
import ray

"""
Load and Transform Points loads interest points from n5 and transforms them into global space
"""

class LoadAndTransformPoints:
    def __init__(self, data_global, xml_input_path, n5_output_path, match_type):
        self.data_global = data_global
        self.xml_input_path = xml_input_path
        self.n5_output_path = n5_output_path
        self.match_type = match_type
    
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
        
    def get_transformation_matrix(self, view_id, view_registrations):
        """
        Compose all affine ViewTransforms for a given view (timepoint, setup)
        """
        try:
            transforms = view_registrations.get(view_id, [])
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
    
    def load_interest_points_from_path(self, base_path, loc_path):
        """
        Load data from any N5 dataset path
        """
        try:      
            if self.n5_output_path.startswith("s3://"):
                path = base_path.rstrip("/")
                s3 = s3fs.S3FileSystem(anon=False)
                store = s3fs.S3Map(root=path, s3=s3, check=False)
                root = zarr.open(store, mode="r")
                group = root[loc_path]
                data = group[:]
                return data.astype(np.float64)
            
            else:
                store = zarr.N5Store(base_path)
                root = zarr.open(store, mode="r")
                group = root[loc_path]
                data = group[:]
                return data.astype(np.float64)
            
        except Exception as e:
            return []
    
    def get_transformed_points(self, view_id, view_data, view_registrations, label):
        """
        Retrieve and transform interest points for a given view
        """
        view_info = view_data[view_id]
        path = view_info['path']
        loc_path = f"{path}/{label}/interestpoints/loc"
        full_path = self.n5_output_path + "interestpoints.n5"
        
        raw_points = self.load_interest_points_from_path(full_path, loc_path)

        if len(raw_points) == 0:
            return []
        
        transform = self.get_transformation_matrix(view_id, view_registrations)
        transformed_points = self.transform_interest_points(raw_points, transform)
            
        return transformed_points
    
    def load_and_transform_points(self, pair, view_data, view_registrations, label):
        """
        Process a single matching task
        """
        viewA, viewB = pair
        try:
            # Retrieve and transform interest points for both views
            if isinstance(viewA, tuple) and len(viewA) == 2:
                tpA, setupA = viewA
                viewA_str = f"(tpId={tpA}, setupId={setupA})"
            else:
                viewA_str = str(viewA)
            if isinstance(viewB, tuple) and len(viewB) == 2:
                tpB, setupB = viewB
                viewB_str = f"(tpId={tpB}, setupId={setupB})"
            else:
                viewB_str = str(viewB)
            
            pointsA = self.get_transformed_points(viewA, view_data, view_registrations, label)
            pointsB = self.get_transformed_points(viewB, view_data, view_registrations, label)

            return pointsA, pointsB, viewA_str, viewB_str
            
        except Exception as e:
            print(f"❌ ERROR: Failed in process_matching_task for views {viewA} and {viewB}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception details: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    # TODO - eventually handle if more than 1 timepoint
    def merge_sets(self, v_sets, pair_sets, i1, i2):
        return [], []
    
    def set_id(self, v1, v_sets):
        """
        Find the index of the component in `v_sets` that contains `v1`
        """
        i = -1
        for j in range(len(v_sets)):
            if v1 in v_sets[j]:
                i = j
        
        return i
    
    def subsets(self, pairs):
        """
        Cluster views into connected components based on the given pairs
        """
        views = list(self.data_global['viewsInterestPoints'].keys())
        v_sets: list[set] = []
        pair_sets: list[list[tuple]] = []       
        groups = None

        counter = 0

        for pair_a, pair_b in pairs:
            
            counter += 1
            if counter == 100:
                break
            
            i1 = self.set_id(pair_a, v_sets)
            i2 = self.set_id(pair_b, v_sets)

            if i1 == -1 and i2 == -1:
                v_set: list[set] = []
                pair_set: list[set] = []
                pair_set.append((pair_a, pair_b))
                v_set.append(pair_a)
                v_set.append(pair_b)

                v_sets.append(v_set)
                pair_sets.append(pair_set)
            
            elif i1 >= 0 and i2 == 0:
                v_sets[i2].append(pair_a)
                pair_sets[i2].append((pair_a, pair_b))
            
            elif i1 >= 0 and i2 == -1:
                v_sets[i1].append(pair_b)
                pair_sets[i1].append((pair_a, pair_b))
            
            elif i1 == i2:
                pair_sets[i1].append((pair_a, pair_b))
            
            else:
                pair_sets, v_sets = self.merge_sets(v_sets, pair_sets, i1, i2)
        
        for view in views:
            is_present = False

            for subset_precursor in v_sets:
                if view in subset_precursor:
                    is_present = True

            if not is_present:
                v_set = []
                pair_set = []

                v_set.append(view)
                v_sets.append(v_set)
                pair_sets.append(pair_set)

        subsets = []

        for i in range(len(v_sets)):
            set_pairs = pair_sets[i]
            set_views = v_sets[i]
            subsets.append((set_views, set_pairs, groups))
        
        return {
            'groups': None,
            'pairs': pairs,
            'rangeComparator': None,
            'subsets': subsets,
            'views': views 
        }

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

    def setup_groups_split(self):
        """
        Generate all unique view pairs and keep only those whose setups overlap
        """
        views = list(self.data_global['viewsInterestPoints'].keys())
        pairs = list(combinations(views, 2))
        final_pairs = []

        for view_a, view_b in pairs:    
            dims_a = self.data_global['viewSetup']['byId'][view_a[1]]
            dims_b = self.data_global['viewSetup']['byId'][view_b[1]]
            
            if self.overlap(view_a, dims_a, view_b, dims_b):
                view_a = (view_a[0], view_a[1])
                view_b = (view_b[0], view_b[1])
                final_pairs.append((view_a, view_b))
        
        return final_pairs
    
    def setup_groups(self):
        """
        Group views by timepoint and generate all unique unordered intra-timepoint pairs
        """
        views = list(self.data_global['viewsInterestPoints'].keys())

        # Group views by timepoint
        timepoint_groups = {}
        for view in views:
            timepoint, _ = view
            if timepoint not in timepoint_groups:
                timepoint_groups[timepoint] = []
            timepoint_groups[timepoint].append(view)

        # Create pairs within each timepoint
        pairs = []
        for timepoint, timepoint_views in timepoint_groups.items():
            for i in range(len(timepoint_views)):
                for j in range(i + 1, len(timepoint_views)):
                    pairs.append((timepoint_views[i], timepoint_views[j]))

        return {
            'groups': timepoint_groups,
            'pairs': pairs,
            'rangeComparator': None,
            'subsets': None,
            'views': views
        }
    
    def as_list(self, x):
        return x if isinstance(x, list) else [x]

    def expand_pairs_with_labels(self, base_pairs, view_ids_global):
        """
        Add a label for each pair
        """
        out = []
        for va, vb in base_pairs:
            la = self.as_list(view_ids_global[va].get('label', []))
            lb = self.as_list(view_ids_global[vb].get('label', []))

            if not la or not lb:
                continue

            lb_set = set(lb)
            common = [l for l in la if l in lb_set]

            for l in common:
                out.append(((va[0], va[1]), (vb[0], vb[1]), l))

        return out
    
    def run(self):
        """
        Executes the entry point of the script.
        """
        view_ids_global = self.data_global['viewsInterestPoints']
        view_registrations = self.data_global['viewRegistrations']

        # Set up view groups using complete dataset info
        if self.match_type == "split-affine":
            setup = self.setup_groups_split()
            setup = self.subsets(setup)
        else:
            setup = self.setup_groups()
        
        # Distribute points loading (very helpful with split-affine)
        @ray.remote
        def process_pair(view_a, view_b, label, view_ids_global, view_registrations):
            if isinstance(view_a, tuple) and len(view_a) == 2:
                tpA, setupA = view_a
                viewA_str = f"(tpId={tpA}, setupId={setupA})"
            else:
                viewA_str = str(view_a)

            if isinstance(view_b, tuple) and len(view_b) == 2:
                tpB, setupB = view_b
                viewB_str = f"(tpId={tpB}, setupId={setupB})"
            else:
                viewB_str = str(view_b)

            pointsA, pointsB, viewA_str, viewB_str = self.load_and_transform_points(
                (view_a, view_b), view_ids_global, view_registrations, label
            )
            return pointsA, pointsB, viewA_str, viewB_str, label

        setup['pairs'] = self.expand_pairs_with_labels(setup['pairs'], view_ids_global)

        # launch Ray tasks
        futures = [
            process_pair.remote(view_a, view_b, label, view_ids_global, view_registrations)
            for view_a, view_b, label in setup['pairs']
        ]

        process_pairs = ray.get(futures)

        return process_pairs, view_registrations