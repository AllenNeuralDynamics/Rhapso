import zarr
import numpy as np

class ResultSaver:
    def __init__(self, base_path):
        """Initialize result saver with base path"""
        self.base_path = base_path
        self.store = self._initialize_store()

    def _initialize_store(self):
        """Initialize appropriate storage backend (S3 or local)"""
        if self.base_path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            return s3fs.S3Map(root=self.base_path, s3=s3, check=False)
        else:
            return zarr.N5Store(self.base_path)

    def save_matches(self, matches, view_paths, clear_correspondences=False):
        """Save matches to storage in N5 format"""
        root = zarr.group(store=self.store, overwrite=False)
        for match in matches:
            viewA, viewB, idxA, idxB = match
            timepoint, _ = viewA
            group_path = f"tpId_{timepoint}_viewSetupId_{viewA[1]}/beads"
            if group_path not in root:
                group = root.create_group(group_path)
            else:
                group = root[group_path]
            if 'correspondences' in group and clear_correspondences:
                del group['correspondences']
            correspondences = group.require_group('correspondences')
            matches_array = np.array([[idxA, idxB]], dtype=np.uint64)
            correspondences.create_dataset("data", data=matches_array, dtype='uint64', chunks=(1000,))

    def save_correspondences_for_view(self, view_id, matches, data_global):
        """Save correspondences for a specific view"""
        timepoint, setup_id = view_id
        group_path = f"tpId_{timepoint}_viewSetupId_{setup_id}/beads"
        
        # Create ID map and process matches
        id_map = self._create_id_map(view_id, matches, data_global)
        matches_array = self._prepare_matches_array(view_id, matches, id_map)
        
        if len(matches_array) == 0:
            print(f"No matches to save for view {view_id}")
            return
            
        # Save to store
        root = zarr.group(store=self.store, overwrite=False)
        group = root.require_group(group_path)
        correspondences = group.require_group('correspondences')
        
        # Delete existing dataset if it exists
        if 'data' in correspondences:
            del correspondences['data']
        
        # Create dataset with explicit shape and dtype
        correspondences.create_dataset(
            "data", 
            data=matches_array,
            shape=matches_array.shape,
            dtype=np.uint64,
            chunks=(min(1000, matches_array.shape[0]), 3)
        )
        correspondences.attrs['idMap'] = id_map
        print(f"Saved {len(matches_array)} correspondences for view {view_id}")

    def _create_id_map(self, view_id, matches, data_global):
        """Create ID map for correspondence storage"""
        timepoint, setup_id = view_id
        id_map = {}
        
        # Create mapping for other views this view has matches with
        for match in matches:
            if match[0] == view_id:
                other_view = match[1]
            else:
                other_view = match[0]
                
            # Create key in format "timepoint,setup,beads"
            key = f"{other_view[0]},{other_view[1]},beads"
            if key not in id_map:
                id_map[key] = len(id_map)  # Assign incrementing IDs
                
        return id_map

    def _prepare_matches_array(self, view_id, matches, id_map):
        """Prepare matches array for storage"""
        matches_list = []
        for match in matches:
            try:
                # Extract the match components safely
                if match[0] == view_id:  
                    idx_a = match[2][0] if isinstance(match[2], list) else match[2]
                    idx_b = match[3][0] if isinstance(match[3], list) else match[3]
                    other_view = match[1]
                else:  
                    idx_a = match[3][0] if isinstance(match[3], list) else match[3]
                    idx_b = match[2][0] if isinstance(match[2], list) else match[2]
                    other_view = match[0]
                
                # Get ID from map for the other view
                other_view_key = f"{other_view[0]},{other_view[1]},beads"
                other_view_id = id_map[other_view_key]
                
                # Convert all values to integers
                matches_list.append([
                    int(float(idx_a)),
                    int(float(idx_b)),
                    int(other_view_id)
                ])
            except Exception as e:
                # Only print warning if debug logging is enabled
                if hasattr(self, 'debug') and self.debug:
                    print(f"Warning: Skipping match due to error: {e}")
                continue
                
        if not matches_list:
            return np.array([], dtype=np.uint64)
            
        return np.array(matches_list, dtype=np.uint64)
