import zarr
import numpy as np
import os
import s3fs
from tqdm import tqdm
import tempfile
from urllib.parse import urlparse

class LoadAndTransformPoints:
    def __init__(self, data_global, xml_input_path, big_stitcher_input):
        """Initialize data loader with base path"""
        self.data_global = data_global
        self.xml_input_path = xml_input_path
        self.big_stitcher_input = big_stitcher_input
    
    def transform_interest_points(self, points, transformation_matrix):
        """Transform interest points using the given transformation matrix"""
        if len(points) == 0: return []
        
        # Convert points to homogeneous coordinates (add 1 as 4th coordinate)
        homogeneous_points = np.column_stack([points, np.ones(len(points))])
        
        # Apply transformation: result = matrix @ points.T, then transpose back
        transformed_homogeneous = (transformation_matrix @ homogeneous_points.T).T
        
        # Convert back to 3D coordinates (remove homogeneous coordinate)
        transformed_points = transformed_homogeneous[:, :3]

        return transformed_points.astype(np.float64)

    def _parse_affine_matrix(self, affine_text):
        """Parse affine transformation matrix from text string"""
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
        """Get the complete transformation matrix for a view by composing all ViewTransforms"""
        try:       
            # Get all transforms for this view
            transforms = view_registrations.get(view_id, [])
            if not transforms:
                print(f"⚠️ No transforms found for view {view_id}, using identity matrix")
                return np.eye(4)
            
            # Start with identity matrix
            final_matrix = np.eye(4)
            
            # Compose all transforms in order
            for i, transform in enumerate(transforms):
                # Get the matrix data - could be stored as 'matrix' or 'affine'
                matrix_data = transform.get('matrix') or transform.get('affine')
                if matrix_data is None:
                    print(f"  ⚠️ No matrix data found in transform {i+1}, skipping")
                    continue
                
                # Parse the affine transform
                if isinstance(matrix_data, str):
                    matrix = self._parse_affine_matrix(matrix_data)
                else:
                    # Assume it's already a matrix
                    matrix = np.array(matrix_data)
                    if matrix.shape == (3, 4):
                        # Convert 3x4 to 4x4
                        matrix = np.vstack([matrix, [0, 0, 0, 1]])
                
                # Compose with previous transforms (matrix multiplication)
                final_matrix = final_matrix @ matrix
            
            return final_matrix
        except Exception as e:
            print(f"❌ Error in get_transformation_matrix for view {view_id}: {e}")
            print(f"Error type: {type(e).__name__}")
            raise
    
    def download_n5_from_s3_to_local(self, s3_uri, local_dir):
        """
        Recursively download an N5 dataset from S3 to a local directory.
        """
        s3 = s3fs.S3FileSystem(anon=False)
        s3_path = s3_uri.replace("s3://", "")
        all_keys = s3.find(s3_path, detail=True)

        for key, obj in tqdm(all_keys.items(), desc="Downloading N5 files from S3"):
            if obj["type"] == "file":
                rel_path = key.replace(s3_path + "/", "")
                local_file_path = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3.get(key, local_file_path)
    
    def load_interest_points_from_path(self, base_path, loc_path):
        """Load data from any N5 dataset path"""
        try:
            if self.xml_input_path.startswith("s3://") and self.big_stitcher_input == True:
                xml_stripped = self.xml_input_path.replace("s3://", "")
                parts = xml_stripped.split("/", 1)
                bucket_prefix = parts[1].rsplit("/", 1)[0]  
                s3_n5_uri = f"s3://{parts[0]}/{bucket_prefix}/"

                parsed = urlparse(self.xml_input_path)
                bucket_and_key = parsed.netloc + parsed.path
                parent_prefix = os.path.dirname(bucket_and_key)

                with tempfile.TemporaryDirectory() as tmpdir:
                    self.download_n5_from_s3_to_local(s3_n5_uri, tmpdir)          
                    new_path = tmpdir + "/" + parent_prefix + base_path
                    store = zarr.N5Store(new_path)
                    root = zarr.open(store, mode="r")
                    group = root[loc_path]
                    data = group[:]
                    return data.astype(np.float64)
                
            elif self.xml_input_path.startswith("s3://"):
                s3 = s3fs.S3FileSystem(anon=False)
                xml_stripped = self.xml_input_path.replace("s3://", "")
                parts = xml_stripped.split("/", 1)
                bucket_prefix = parts[1].rsplit("/", 1)[0]  
                base_path = f"{parts[0]}/{bucket_prefix}/interestpoints.n5"
                store = s3fs.S3Map(root=base_path, s3=s3, check=False)
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
            print(f"ERROR: Failed to load data from {loc_path}")
            print(f"Error details: {e}")
            raise e
    
    def get_transformed_points(self, view_id, view_data, view_registrations, xml_base_path):
        """Retrieve and transform interest points for a given view."""
        view_info = view_data[view_id]
        loc_path = f"{view_info['path']}/{view_info['label']}/interestpoints/loc"
        full_path = xml_base_path + "/interestpoints.n5"
        raw_points = self.load_interest_points_from_path(full_path, loc_path)
        transform = self.get_transformation_matrix(view_id, view_registrations)
        transformed_points = self.transform_interest_points(raw_points, transform)
        
        return transformed_points
    
    def load_and_transform_points(self, pair, view_data, view_registrations, xml_base_path):
        """Process a single matching task"""
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
            
            pointsA = self.get_transformed_points(viewA, view_data, view_registrations, xml_base_path)
            pointsB = self.get_transformed_points(viewB, view_data, view_registrations, xml_base_path)

            return pointsA, pointsB, viewA_str, viewB_str
            
        except Exception as e:
            print(f"❌ ERROR: Failed in process_matching_task for views {viewA} and {viewB}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception details: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def setup_groups(self, data_global):
        """Set up view groups for pairwise matching"""
        # Get all views from viewsInterestPoints
        views = list(data_global['viewsInterestPoints'].keys())
        
        # Group views by timepoint
        timepoint_groups = {}
        for view in views:
            timepoint, setup_id = view
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
    
    def run(self):
        # Use data_global for all subsequent operations
        view_ids_global = self.data_global['viewsInterestPoints']
        view_registrations = self.data_global['viewRegistrations']

        # Set up view groups using complete dataset info
        setup = self.setup_groups(self.data_global)

        # Initialize data loader
        xml_dir = os.path.dirname(self.xml_input_path) if not self.xml_input_path.startswith('s3://') else ""

        # Iterate through each pair of views to perform matching
        process_pairs = []
        for idx, pair in enumerate(setup['pairs'], 1):
            viewA, viewB = pair  # Unpack the current pair of view IDs

            # Unpack for clarity
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
            
            # Run the matching task for the current pair and get results
            pointsA, pointsB, viewA_str, viewB_str = self.load_and_transform_points(pair, view_ids_global, view_registrations, xml_dir)
            process_pairs.append((pointsA, pointsB, viewA_str, viewB_str))

        return process_pairs