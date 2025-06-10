import zarr
import s3fs
import numpy as np
import os

class DataLoader:
    def __init__(self, base_path, logging=None):
        """Initialize data loader with base path"""
        self.base_path = base_path
        self.store = self._initialize_store()
        
        # Set default logging configuration if none provided
        self.logging = {
            'interest_point_transformation': True,  # Show all by default
        }
        
        # Update with user-provided logging settings
        if logging is not None:
            self.logging.update(logging)

    def _initialize_store(self):
        """Initialize appropriate storage backend (S3 or local)"""
        if self.base_path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            return s3fs.S3Map(root=self.base_path, s3=s3, check=False)
        else:
            return zarr.N5Store(self.base_path)

    def load_dataset(self, dataset_path):
        """Load dataset from specified path"""
        try:
            root = zarr.open(self.store, mode='r')
            if dataset_path not in root:
                print(f"ERROR: Dataset path {dataset_path} not found.")
                return None
            return root[dataset_path]
        except Exception as e:
            print(f"ERROR: Failed to load dataset from {dataset_path}")
            print(f"Error details: {e}")
            print(f"Error type: {type(e).__name__}")
            return None

    def load_interest_points(self, dataset_path):
        """Load interest points from dataset"""
        loc_path = f"{dataset_path}/interestpoints/loc"
        dataset = self.load_dataset(loc_path)
        return dataset[:]

    def get_interest_points_path(self, view_id, view_data):
        """Construct the complete N5 path for interest points"""
        try:
            view_info = view_data[view_id]
            relative_path = f"{view_info['path']}/{view_info['label']}/interestpoints/loc"
            
            # Use xml_base_path if available, otherwise use base_path directory
            if hasattr(self, 'xml_base_path'):
                # Combine xml_base_path + interestpoints.n5 + relative_path
                complete_path = os.path.join(self.xml_base_path, 'interestpoints.n5', relative_path)
            else:
                # Use the base_path which should already include interestpoints.n5
                complete_path = os.path.join(self.base_path, relative_path)
            
            print(f"Constructed complete path: {complete_path}")
            return complete_path
        except Exception as e:
            print(f"ERROR: Failed to construct interest points path for view_id {view_id}")
            print(f"Error details: {e}")
            print(f"Error type: {type(e).__name__}")
            raise e

    def load_interest_points_from_path(self, dataset_path):
        """Load data from any N5 dataset path"""
        try:
            # Navigate to the dataset step by step
            root = zarr.open(self.store, mode='r')
            
            # Calculate relative path from the store base
            if dataset_path.startswith(self.base_path):
                relative_path = dataset_path.replace(self.base_path, '').lstrip('/')
            else:
                relative_path = dataset_path.lstrip('/')
            
            # Navigate to the dataset
            if relative_path:
                path_parts = relative_path.split('/')
                current_group = root
                for part in path_parts:
                    current_group = current_group[part]
            else:
                current_group = root
            
            # Load the actual data
            data_array = current_group[:]
            return data_array.astype(np.float64)
            
        except Exception as e:
            print(f"ERROR: Failed to load data from {dataset_path}")
            print(f"Error details: {e}")
            raise e

    def get_transformation_matrix(self, view_id, view_data, view_registrations):
        """Get the complete transformation matrix for a view by composing all ViewTransforms"""
        try:
            print(f"📐 Computing transformation matrix for view {view_id}")
            
            # Get all transforms for this view
            transforms = view_registrations.get(view_id, [])
            if not transforms:
                print(f"⚠️ No transforms found for view {view_id}, using identity matrix")
                return np.eye(4)
            
            print(f"🔗 Found {len(transforms)} transforms to compose:")
            
            # Start with identity matrix
            final_matrix = np.eye(4)
            
            # Compose all transforms in order
            for i, transform in enumerate(transforms):
                # Handle missing name field gracefully
                transform_type = transform.get('type', 'unknown')
                transform_name = transform.get('name', f'Transform_{i+1}')
                
                print(f"  📐 Transform {i+1}: type='{transform_type}', name='{transform_name}'")
                
                # Get the matrix data - could be stored as 'matrix' or 'affine'
                matrix_data = transform.get('matrix') or transform.get('affine')
                if matrix_data is None:
                    print(f"  ⚠️ No matrix data found in transform {i+1}, skipping")
                    continue
                    
                print(f"  🔍 Raw affine text: '{matrix_data}'")
                
                # Parse the affine transform
                if isinstance(matrix_data, str):
                    matrix = self._parse_affine_matrix(matrix_data)
                else:
                    # Assume it's already a matrix
                    matrix = np.array(matrix_data)
                    if matrix.shape == (3, 4):
                        # Convert 3x4 to 4x4
                        matrix = np.vstack([matrix, [0, 0, 0, 1]])
                        
                print(f"  ✅ Successfully parsed to 4x4 matrix:")
                for row_idx, row in enumerate(matrix):
                    print(f"    Row {row_idx}: [{row[0]:f}, {row[1]:f}, {row[2]:f}, {row[3]:f}]")
                
                # Compose with previous transforms (matrix multiplication)
                final_matrix = final_matrix @ matrix
                print(f"  🔗 Composed with previous transforms")
            
            print(f"🎯 Final composed transformation matrix:")
            for row_idx, row in enumerate(final_matrix):
                print(f"  Row {row_idx}: [{row[0]:f}, {row[1]:f}, {row[2]:f}, {row[3]:f}]")
            
            return final_matrix
        except Exception as e:
            print(f"❌ Error in get_transformation_matrix for view {view_id}: {e}")
            print(f"Error type: {type(e).__name__}")
            raise
        
    def transform_interest_points(self, points, transformation_matrix):
        """Transform interest points using the given transformation matrix"""
        if points is None or len(points) == 0:
            print("⚠️ No points to transform")
            return np.array([])
        
        # Print transformation matrix in the corrected format
        matrix = transformation_matrix
        print(f"   Matrix: ({matrix[0,0]:g}, {matrix[0,1]:g}, {matrix[0,2]:g}, {matrix[0,3]:g}), "
              f"({matrix[1,0]:g}, {matrix[1,1]:g}, {matrix[1,2]:g}, {matrix[1,3]:g}), "
              f"({matrix[2,0]:g}, {matrix[2,1]:g}, {matrix[2,2]:g}, {matrix[2,3]:g})")
        
        print(f"⚙️ Starting matrix multiplication for {len(points)} interest points")
        
        # Convert points to homogeneous coordinates (add 1 as 4th coordinate)
        homogeneous_points = np.column_stack([points, np.ones(len(points))])
        
        # Apply transformation: result = matrix @ points.T, then transpose back
        transformed_homogeneous = (transformation_matrix @ homogeneous_points.T).T
        
        # Convert back to 3D coordinates (remove homogeneous coordinate)
        transformed_points = transformed_homogeneous[:, :3]
        
        # Limit the number of point transformations printed based on logging configuration
        max_points_to_show = self.logging.get('interest_point_transformation', len(points))
        if max_points_to_show > 0:
            print(f"📍 Sample transformations (showing up to {max_points_to_show} points):")
            for i in range(min(max_points_to_show, len(points))):
                before = points[i]
                after = transformed_points[i]
                print(f"🔢 Point {i}: {before} → {after}")
            
            if max_points_to_show < len(points):
                print(f"  ... ({len(points) - max_points_to_show} more points transformed)")
        
        print(f"✅ Successfully transformed all {len(transformed_points)} points")

        return transformed_points.astype(np.float64)


    def get_transformed_interest_points(self, view_id, view_data=None):
        """Get and transform interest points for a specific view"""
        print(f"Loading interest points for view {view_id}")
        
        try:
            # Construct N5 path
            dataset_path = self.get_interest_points_path(view_id, view_data)
            
            # Load raw interest points
            raw_points = self.load_interest_points_from_path(dataset_path)
            
            # Get transformation matrix
            transform_matrix = self.get_transformation_matrix(view_id, view_data)
            
            # Apply transformation
            transformed_points = self.transform_interest_points(raw_points, transform_matrix)
            
            return transformed_points
            
        except Exception as e:
            print(f"ERROR: Failed to load interest points for view {view_id}")
            print(f"Error details: {e}")
            print(f"Error type: {type(e).__name__}")
            raise e

    def build_label_map(self, view_ids, sequence_description=None):
        """Build label map for specified view IDs with optional sequence description"""
        # TODO: Implement label map building
        label_map = {}
        for view_id in view_ids:
            # Each view gets default weight of 1.0 for 'beads' label
            label_map[view_id] = {'beads': 1.0}
        return label_map

    def clear_correspondences(self, view_id):
        """Clear existing correspondences for a view"""
        # TODO: Implement correspondence clearing
        pass

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
            
            # print(f"  ✅ Parsed affine matrix (3x4 → 4x4):")
            # for i, row in enumerate(matrix_4x4):
            #     print(f"    [{row[0]:f}, {row[1]:f}, {row[2]:f}, {row[3]:f}]")
            
            return matrix_4x4
            
        except Exception as e:
            print(f"❌ Error parsing affine matrix from '{affine_text}': {e}")
            # Return identity matrix as fallback
            return np.eye(4)
