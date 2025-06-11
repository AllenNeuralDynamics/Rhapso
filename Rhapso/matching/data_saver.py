"""
Functions for saving data to N5 format, specifically designed for interest points and correspondences.
Uses zarr instead of z5py for better compatibility.
"""
import os
import numpy as np
import logging
import zarr
import json
import shutil

def save_interest_points(n5_output_path, timepoint_id, view_setup_id, label, points):
    """
    Save interest points to an N5 file format using zarr.
    
    Args:
        n5_output_path: Path to the output N5 file/directory
        timepoint_id: Timepoint ID
        view_setup_id: View setup ID
        label: Label (e.g., "beads")
        points: List of interest points, each with [x, y, z] coordinates
    """
    try:
        # Create the N5 file if it doesn't exist
        os.makedirs(n5_output_path, exist_ok=True)
        
        # Create the interest points dataset path
        ip_path = f'tpId_{timepoint_id}_viewSetupId_{view_setup_id}/{label}/interestpoints'
        full_path = os.path.join(n5_output_path, ip_path)
        os.makedirs(full_path, exist_ok=True)
        
        # Convert points to numpy array and ensure proper shape
        points_array = np.array(points)
        
        # Create ID dataset (just a sequence from 0 to n-1)
        ids = np.arange(len(points_array)).reshape(1, -1).astype(np.uint64)
        
        # Create location dataset
        # Ensure proper shape with dim 0 = 3 (x, y, z)
        locations = points_array.T.astype(np.float64)  # Transpose to get [3, n_points]
        
        # Save ID dataset
        id_store = zarr.open(os.path.join(full_path, 'id'), mode='w')
        id_chunk_size = (1, min(300000, len(ids[0])))
        id_array = id_store.create_dataset('data', data=ids, chunks=id_chunk_size, 
                                           compressor=zarr.GZip(level=1))
        
        # Save location dataset
        loc_store = zarr.open(os.path.join(full_path, 'loc'), mode='w')
        loc_chunk_size = (3, min(300000, locations.shape[1]))
        loc_array = loc_store.create_dataset('data', data=locations, chunks=loc_chunk_size, 
                                             compressor=zarr.GZip(level=1))
        
        # Add attributes
        attrs = {
            'pointcloud': '1.0.0',
            'type': 'list',
            'list version': '1.0.0'
        }
        
        # Save attributes as JSON
        with open(os.path.join(full_path, 'attributes.json'), 'w') as f:
            json.dump(attrs, f)
        
        print(f"Successfully saved {len(points)} interest points to {full_path}")
        return True
    
    except Exception as e:
        logging.error(f"Error saving interest points: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def save_correspondences(n5_output_path, reference_tp, reference_vs, ref_label, correspondences, matched_views):
    """
    Save the matching results to an N5 file format using zarr.
    
    Args:
        n5_output_path: Path to the output N5 file/directory
        reference_tp: Reference timepoint
        reference_vs: Reference view setup ID
        ref_label: Label (e.g., "beads")
        correspondences: List of correspondences, each containing (viewIdA, viewIdB, idA, idB)
        matched_views: List of tuples containing (timepoint, viewSetup, label)
    """
    try:
        # Create the N5 file if it doesn't exist
        n5_base_path = os.path.join(n5_output_path, "interestpoints.n5")
        os.makedirs(n5_base_path, exist_ok=True)
        
        # Group correspondences by target view
        view_correspondences = {}
        for viewA, viewB, idA, idB in correspondences:
            # Extract just the view setup ID from viewA and viewB (which might be tuples)
            if isinstance(viewA, tuple):
                viewA_setup = viewA[1]
                viewA_tp = viewA[0]
            else:
                viewA_setup = viewA
                viewA_tp = reference_tp
                
            if isinstance(viewB, tuple):
                viewB_setup = viewB[1]
                viewB_tp = viewB[0]
            else:
                viewB_setup = viewB
                viewB_tp = reference_tp
            
            # Create a key for the target view
            view_key = (viewB_tp, viewB_setup)
            
            # Initialize list for this view if needed
            if view_key not in view_correspondences:
                view_correspondences[view_key] = []
                
            # Store the correspondence
            view_correspondences[view_key].append((viewA, viewB, idA, idB))
        
        # Save correspondences for each view
        saved_filepaths = []
        for (view_tp, view_setup), view_corrs in view_correspondences.items():
            # Create the correspondences dataset path for this view
            corr_dir = f"tpId_{view_tp}_viewSetupId_{view_setup}/{ref_label}/correspondences"
            full_path = os.path.join(n5_base_path, corr_dir)
            os.makedirs(full_path, exist_ok=True)

            print(f"Saving {len(view_corrs)} corresponding interest points")
            
            # Create ID map for all views
            viewid_to_labels = {}
            for (tp, vs, label) in matched_views:
                if (tp, vs) not in viewid_to_labels:
                    viewid_to_labels[(tp, vs)] = set()
                viewid_to_labels[(tp, vs)].add(label)
            
            print(f"Found {len(viewid_to_labels)} unique ViewIds with correspondences")
            
            # Create the idMap
            id_map = {}
            quick_lookup = {}
            id_counter = 0
            
            print("Creating idMap for mapping (ViewId,label) combinations to unique IDs")
            for (tp, vs), labels in viewid_to_labels.items():
                if (tp, vs) not in quick_lookup:
                    quick_lookup[(tp, vs)] = {}
                
                print(f"  ViewId [TP={tp}, Setup={vs}] has {len(labels)} labels")
                
                for label in labels:
                    key = f"{tp},{vs},{label}"
                    id_map[key] = id_counter
                    quick_lookup[(tp, vs)][label] = id_counter
                    print(f"    Label '{label}' assigned ID {id_counter} (key = '{key}')")
                    id_counter += 1
            
            print(f"Total unique ViewId+label combinations: {len(id_map)}")
            print("Storing idMap as N5 attribute for later lookup during loading")
            
            # Store the ID map as an attribute
            attrs = {
                'correspondences': '1.0.0',
                'idMap': id_map
            }
            
            with open(os.path.join(full_path, 'attributes.json'), 'w') as f:
                json.dump(attrs, f)
            
            # Skip if no correspondences for this view
            if not view_corrs:
                print(f"No correspondences to save for {corr_dir}")
                continue
            
            print(f"Creating correspondence dataset with dimensions [3 × {len(view_corrs)}]")
            print("  Each entry contains [detectionId, correspondingDetectionId, viewId+label ID]")
            
            # Prepare data arrays for this view's correspondences
            corr_data = np.zeros((3, len(view_corrs)), dtype=np.uint64)
            
            for i, (viewA, viewB, idA, idB) in enumerate(view_corrs):
                # Find the index in matched_views for viewB
                if isinstance(viewB, tuple):
                    viewB_tp = viewB[0]
                    viewB_setup = viewB[1]
                else:
                    viewB_tp = reference_tp
                    viewB_setup = viewB
                    
                view_key = f"{viewB_tp},{viewB_setup},{ref_label}"
                view_idx = id_map.get(view_key, 0)
                
                # Store the correspondence
                corr_data[0, i] = idA  # Reference interest point ID
                corr_data[1, i] = idB  # Corresponding interest point ID
                corr_data[2, i] = view_idx  # View index
            
            # Save data to zarr
            data_path = os.path.join(full_path, 'data')
            if os.path.exists(data_path):
                shutil.rmtree(data_path)
                
            corr_store = zarr.open(data_path, mode='w')
            corr_chunk_size = (3, min(300000, corr_data.shape[1]))
            print(f"Saving correspondence data with block size [1 × 300000]")
            corr_array = corr_store.create_dataset('data', data=corr_data, chunks=corr_chunk_size,
                                                 compressor=zarr.GZip(level=1))
            
            print(f"Saved: {os.path.abspath(full_path)}")
            saved_filepaths.append(os.path.abspath(full_path))
        
        # Also save to the reference view directory for backward compatibility
        ref_corr_dir = f"tpId_{reference_tp}_viewSetupId_{reference_vs}/{ref_label}/correspondences"
        ref_full_path = os.path.join(n5_output_path, ref_corr_dir)
        os.makedirs(ref_full_path, exist_ok=True)
        
        # Use the same approach for the reference view
        if correspondences:
            print(f"Saving {len(correspondences)} corresponding interest points")
            
            # Save attributes for reference view
            with open(os.path.join(ref_full_path, 'attributes.json'), 'w') as f:
                json.dump(attrs, f)
            
            print(f"Creating correspondence dataset with dimensions [3 × {len(correspondences)}]")
            print("  Each entry contains [detectionId, correspondingDetectionId, viewId+label ID]")
            
            # Prepare data arrays for all correspondences
            all_corr_data = np.zeros((3, len(correspondences)), dtype=np.uint64)
            
            for i, (viewA, viewB, idA, idB) in enumerate(correspondences):
                # Extract view setup ID
                if isinstance(viewB, tuple):
                    viewB_setup = viewB[1]
                    viewB_tp = viewB[0]
                else:
                    viewB_setup = viewB
                    viewB_tp = reference_tp
                
                # Find the index in matched_views for viewB
                view_key = f"{viewB_tp},{viewB_setup},{ref_label}"
                view_idx = id_map.get(view_key, 0)
                
                # Store the correspondence
                all_corr_data[0, i] = idA
                all_corr_data[1, i] = idB
                all_corr_data[2, i] = view_idx
            
            # Save data to zarr for reference view
            print(f"Saving correspondence data with block size [1 × 300000]")
            all_data_path = os.path.join(ref_full_path, 'data')
            all_corr_store = zarr.open(all_data_path, mode='w')
            all_corr_chunk_size = (3, min(300000, all_corr_data.shape[1]))
            all_corr_array = all_corr_store.create_dataset('data', data=all_corr_data, 
                                                        chunks=all_corr_chunk_size,
                                                        compressor=zarr.GZip(level=1))
            
            print(f"Saved: {os.path.abspath(ref_full_path)}")
            saved_filepaths.append(os.path.abspath(ref_full_path))
        
        # Add summary of saved data with emojis
        n5_base_path = os.path.join(n5_output_path, "interestpoints.n5")
        print("\n📊 Save Summary:")
        print("---------------------------")
        print(f"🔢 Total correspondences saved: {len(correspondences)}")
        print(f"📂 Saved to {len(view_correspondences)} view-specific directories in: {os.path.abspath(n5_base_path)}")
        for (view_tp, view_setup), view_corrs in view_correspondences.items():
            print(f"  📁 tpId_{view_tp}_viewSetupId_{view_setup}/{ref_label}/correspondences: {len(view_corrs)} correspondences")
        print(f"📁 Reference view: {os.path.basename(ref_full_path)}: {len(correspondences)} correspondences")
        print("---------------------------")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Error saving correspondences: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
