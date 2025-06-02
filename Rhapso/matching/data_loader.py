import zarr
import s3fs
import numpy as np
import os

class DataLoader:
    def __init__(self, base_path):
        """Initialize data loader with base path"""
        self.base_path = base_path
        self.store = self._initialize_store()

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

    def transform_interest_points(self, interest_points, transform_matrix):
        """Apply transformation matrix to interest points"""
        if len(interest_points) == 0:
            return interest_points
            
        print(f"⚙️ Starting matrix multiplication for {len(interest_points)} interest points")
        
        # Convert to homogeneous coordinates [x, y, z, 1]
        homogeneous_points = np.column_stack([interest_points, np.ones(len(interest_points))])
        
        # Apply transformation: transformed = matrix @ points.T
        transformed = (transform_matrix @ homogeneous_points.T).T
        
        # Return only x, y, z coordinates (drop homogeneous coordinate)
        transformed_points = transformed[:, :3]
        
        print(f"✅ Successfully transformed all {len(transformed_points)} points")
        
        # Show sample transformations with proper formatting
        if len(transformed_points) > 0:
            print(f"📍 Sample transformations:")
            num_samples = min(3, len(transformed_points))
            for i in range(num_samples):
                # Format numbers to avoid scientific notation
                before = [f"{x:.2f}" for x in interest_points[i]]
                after = [f"{x:.2f}" for x in transformed_points[i]]
                print(f"   Point {i}: [{', '.join(before)}] → [{', '.join(after)}]")
            
            # Show total count if there are more points than samples shown
            if len(transformed_points) > num_samples:
                remaining = len(transformed_points) - num_samples
                print(f"   ... and {remaining} more")
            print()  # Add blank line separator
        
        return transformed_points.astype(np.float64)

    def get_transformation_matrix(self, view_id, view_data, view_registrations=None):
        """Get transformation matrix for a specific view from parsed registrations"""
        if view_registrations is None:
            return np.eye(4)
        
        # Look up transformation for this view
        if view_id in view_registrations:
            registration = view_registrations[view_id]
            transforms = registration.get('transforms', [])
            
            if transforms:
                transform = transforms[0]
                matrix = transform['matrix']
                print(f"📐 Retrieved transformation matrix for view {view_id}: {transform['type']}")
                print(f"Matrix: {matrix}")
                print()  # Add blank line after matrix
                return matrix
        
        # Default to identity matrix
        return np.eye(4)

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
