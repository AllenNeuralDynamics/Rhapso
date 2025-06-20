import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import z5py

class CorrespondenceN5Saver:
    """
    Class for saving correspondences as N5 data.
    
    Based on the InterestPointsN5.java implementation, this saves correspondences
    in a compatible format with the Multiview Reconstruction framework.
    """
    
    def __init__(self, n5_output_path: str):
        """
        Initialize the N5 saver with the output path.
        
        Args:
            n5_output_path: Path to the output N5 file/directory
        """
        self.n5_output_path = n5_output_path
        
    def create_id_map(
        self, 
        matched_views: List[Tuple[int, int, str]]
    ) -> Dict[str, int]:
        """
        Create an ID map that assigns a unique ID to each view setup.
        
        Args:
            matched_views: List of tuples containing (timepoint, viewSetup, label)
        
        Returns:
            A dictionary mapping "timepoint,viewSetup,label" to unique IDs
        """
        id_map = {}
        for i, (tp, vs, label) in enumerate(matched_views):
            key = f"{tp},{vs},{label}"
            id_map[key] = i
            
        return id_map
    
    def save_correspondences(
        self,
        reference_tp: int,
        reference_vs: int,
        ref_label: str,
        correspondences: Dict[int, List[Tuple[int, int]]],
        matched_views: List[Tuple[int, int, str]]
    ) -> None:
        """
        Save correspondences as N5 data.
        
        Args:
            reference_tp: Reference timepoint
            reference_vs: Reference view setup ID
            ref_label: Reference label (e.g., "beads")
            correspondences: Dictionary mapping target view ID to list of point correspondences 
                             where each correspondence is (ref_point_idx, target_point_idx)
            matched_views: List of tuples containing (timepoint, viewSetup, label)
        """
        # Create the base path for correspondences
        base_path = os.path.join(
            self.n5_output_path, 
            f"tpId_{reference_tp}_viewSetupId_{reference_vs}",
            ref_label,
            "correspondences"
        )
        
        # Create the ID map
        id_map = self.create_id_map(matched_views)
        
        # Create attributes.json
        attributes = {
            "correspondences": "1.0.0",  # Version
            "idMap": id_map
        }
        
        # Create the N5 file if it doesn't exist
        f = z5py.File(self.n5_output_path, use_zarr_format=False, mode='a')
        
        # Ensure all parent groups exist
        parts = base_path.replace(self.n5_output_path, '').strip('/').split('/')
        current = f
        for part in parts:
            if part not in current:
                current.create_group(part)
            current = current[part]
            
        # Write attributes.json
        for key, value in attributes.items():
            current.attrs[key] = value
        
        # Prepare the data arrays for correspondences
        ref_indices = []
        target_indices = []
        correspondence_ids = []
        
        # Process each target view and add correspondences
        for target_vs_id, corrs in correspondences.items():
            if not corrs:
                continue
                
            # Find the target view in matched_views to get its ID
            for tp, vs, label in matched_views:
                if vs == target_vs_id and label == ref_label:
                    target_id = id_map.get(f"{tp},{vs},{label}")
                    if target_id is not None:
                        # Add each correspondence
                        for ref_idx, target_idx in corrs:
                            ref_indices.append(ref_idx)
                            target_indices.append(target_idx)
                            correspondence_ids.append(target_id)
                    break
        
        # Convert to numpy arrays
        ref_indices = np.array(ref_indices, dtype=np.uint64)
        target_indices = np.array(target_indices, dtype=np.uint64)
        correspondence_ids = np.array(correspondence_ids, dtype=np.uint64)
        
        # Combine into a single dataset with shape [3, num_correspondences]
        data = np.stack([ref_indices, target_indices, correspondence_ids])
        
        # Create and write the dataset
        data_path = os.path.join(base_path, "data")
        parts = data_path.replace(self.n5_output_path, '').strip('/').split('/')
        
        # Navigate to parent of the dataset
        current = f
        for part in parts[:-1]:
            if part not in current:
                current.create_group(part)
            current = current[part]
        
        # Create the dataset
        dataset_name = parts[-1]
        chunk_size = min(300000, data.shape[1]) if data.shape[1] > 0 else 1
        if dataset_name not in current:
            dataset = current.create_dataset(
                dataset_name, 
                data=data, 
                chunks=(1, chunk_size), 
                compression='gzip'
            )
        else:
            # If it exists, overwrite it
            dataset = current[dataset_name]
            dataset[:] = data
        
        # Set dataset attributes
        dataset.attrs['dataType'] = 'uint64'
        dataset.attrs['compression'] = {'type': 'gzip', 'level': -1}
        dataset.attrs['blockSize'] = [1, chunk_size]
        dataset.attrs['dimensions'] = list(data.shape)
        
        print(f"Successfully saved {len(ref_indices)} correspondences to {data_path}")

    def save_interest_points(
        self,
        timepoint: int,
        view_setup: int,
        label: str,
        point_ids: np.ndarray,
        locations: np.ndarray
    ) -> None:
        """
        Save interest points to the N5 file.
        
        Args:
            timepoint: Timepoint ID
            view_setup: View setup ID
            label: Label (e.g., "beads")
            point_ids: Array of point IDs
            locations: Array of point locations with shape (3, num_points) for x,y,z
        """
        # Create the base path for interest points
        base_path = os.path.join(
            self.n5_output_path, 
            f"tpId_{timepoint}_viewSetupId_{view_setup}",
            label,
            "interestpoints"
        )
        
        # Create the N5 file if it doesn't exist
        f = z5py.File(self.n5_output_path, use_zarr_format=False, mode='a')
        
        # Ensure all parent groups exist
        parts = base_path.replace(self.n5_output_path, '').strip('/').split('/')
        current = f
        for part in parts:
            if part not in current:
                current.create_group(part)
            current = current[part]
            
        # Write attributes
        current.attrs['pointcloud'] = "1.0.0"
        current.attrs['type'] = "list"
        current.attrs['list version'] = "1.0.0"
        
        # Save the ID dataset
        id_path = os.path.join(base_path, "id")
        parts = id_path.replace(self.n5_output_path, '').strip('/').split('/')
        
        # Navigate to parent of the dataset
        current = f
        for part in parts[:-1]:
            if part not in current:
                current.create_group(part)
            current = current[part]
        
        # Create the ID dataset - reshape to (1, num_points)
        if len(point_ids.shape) == 1:
            point_ids = point_ids.reshape(1, -1)
            
        dataset_name = parts[-1]
        chunk_size = min(300000, point_ids.shape[1]) if point_ids.shape[1] > 0 else 1
        
        if dataset_name not in current:
            id_dataset = current.create_dataset(
                dataset_name, 
                data=point_ids, 
                chunks=(1, chunk_size), 
                compression='gzip'
            )
        else:
            id_dataset = current[dataset_name]
            id_dataset[:] = point_ids
        
        # Set ID dataset attributes
        id_dataset.attrs['dataType'] = 'uint64'
        id_dataset.attrs['compression'] = {'type': 'gzip', 'level': -1}
        id_dataset.attrs['blockSize'] = [1, chunk_size]
        id_dataset.attrs['dimensions'] = list(point_ids.shape)
        
        # Save the locations dataset
        loc_path = os.path.join(base_path, "loc")
        parts = loc_path.replace(self.n5_output_path, '').strip('/').split('/')
        
        # Navigate to parent of the dataset
        current = f
        for part in parts[:-1]:
            if part not in current:
                current.create_group(part)
            current = current[part]
        
        # Create the locations dataset - should have shape (3, num_points)
        dataset_name = parts[-1]
        chunk_size = min(300000, locations.shape[1]) if locations.shape[1] > 0 else 1
        
        if dataset_name not in current:
            loc_dataset = current.create_dataset(
                dataset_name, 
                data=locations, 
                chunks=(locations.shape[0], chunk_size), 
                compression='gzip'
            )
        else:
            loc_dataset = current[dataset_name]
            loc_dataset[:] = locations
        
        # Set locations dataset attributes
        loc_dataset.attrs['dataType'] = 'float64'
        loc_dataset.attrs['compression'] = {'type': 'gzip', 'level': -1}
        loc_dataset.attrs['blockSize'] = [locations.shape[0], chunk_size]
        loc_dataset.attrs['dimensions'] = list(locations.shape)
        
        print(f"Successfully saved {locations.shape[1]} interest points to {base_path}")


def save_correspondences(n5_output_path, reference_tp, reference_vs, ref_label, correspondences, matched_views):
    """
    Convenience function to save correspondences.
    
    Args:
        n5_output_path: Path to the output N5 file/directory
        reference_tp: Reference timepoint
        reference_vs: Reference view setup ID
        ref_label: Reference label (e.g., "beads")
        correspondences: Dictionary mapping target view ID to list of point correspondences
        matched_views: List of tuples containing (timepoint, viewSetup, label)
    """
    saver = CorrespondenceN5Saver(n5_output_path)
    saver.save_correspondences(reference_tp, reference_vs, ref_label, correspondences, matched_views)


def save_interest_points(n5_output_path, timepoint, view_setup, label, point_ids, locations):
    """
    Convenience function to save interest points.
    
    Args:
        n5_output_path: Path to the output N5 file/directory
        timepoint: Timepoint ID
        view_setup: View setup ID
        label: Label (e.g., "beads")
        point_ids: Array of point IDs
        locations: Array of point locations with shape (3, num_points) for x,y,z
    """
    saver = CorrespondenceN5Saver(n5_output_path)
    saver.save_interest_points(timepoint, view_setup, label, point_ids, locations)
