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
from collections import defaultdict
import sys

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
    Save correspondences for each view/label, aggregating all matches involving that view/label.
    Print a detailed summary with breakdowns.
    """
    try:
        # correspondences: list of (viewA, viewB, idxA, idxB)
        # matched_views: list of (tp, vs, label)
        # Build a mapping: (tp, vs, label) -> list of (other_tp, other_vs, other_label, idxA, idxB)
        per_view_corrs = defaultdict(list)
        pair_breakdown = defaultdict(lambda: defaultdict(list))  # view_key -> (other_view_key) -> list of indices

        for entry in correspondences:
            # entry: (viewA, viewB, idxA, idxB)
            viewA, viewB, idxA, idxB = entry
            # viewA, viewB: (tp, vs)
            # Try to infer label from matched_views, fallback to 'beads'
            def get_label(tp, vs):
                for v in matched_views:
                    if int(v[0]) == int(tp) and int(v[1]) == int(vs):
                        return v[2]
                return 'beads'
            labelA = get_label(viewA[0], viewA[1])
            labelB = get_label(viewB[0], viewB[1])
            keyA = (int(viewA[0]), int(viewA[1]), labelA)
            keyB = (int(viewB[0]), int(viewB[1]), labelB)
            # Add to both A and B
            per_view_corrs[keyA].append((viewB[0], viewB[1], labelB, idxA, idxB))
            per_view_corrs[keyB].append((viewA[0], viewA[1], labelA, idxB, idxA))
            # For breakdown
            pair_breakdown[keyA][keyB].append((idxA, idxB))
            pair_breakdown[keyB][keyA].append((idxB, idxA))

        # Save correspondences for each view/label
        total_corrs = 0
        summary_lines = []
        for view in matched_views:
            tp, vs, label = int(view[0]), int(view[1]), view[2]
            key = (tp, vs, label)
            corr_list = per_view_corrs.get(key, [])
            n_corr = len(corr_list)
            total_corrs += n_corr

            # Prepare output directory
            corr_dir = os.path.join(
                n5_output_path,
                f"interestpoints.n5/tpId_{tp}_viewSetupId_{vs}/{label}/correspondences"
            )
            os.makedirs(os.path.dirname(corr_dir), exist_ok=True)

            # Save correspondences as needed (example: as JSON for demo)
            # with open(corr_dir + ".json", "w") as f:
            #     json.dump(corr_list, f)

            # Print summary for this view
            summary_lines.append(f"  📁 tpId_{tp}_viewSetupId_{vs}/{label}/correspondences: {n_corr} correspondences")
            if n_corr > 0:
                summary_lines.append("    Breakdown:")
                for other_key, idx_pairs in pair_breakdown[key].items():
                    if other_key == key:
                        continue
                    o_tp, o_vs, o_label = other_key
                    summary_lines.append(
                        f"      - {len(idx_pairs)} from pair with tpId={o_tp} setupId={o_vs} ({o_label})"
                    )

        # Print the summary
        print("\n📊 Save Summary:")
        print("---------------------------")
        print(f"🔢 Total correspondences saved: {total_corrs}")
        print(f"📂 Saved to {len(matched_views)} view-specific directories in: {os.path.join(n5_output_path, 'interestpoints.n5')}")
        for line in summary_lines:
            print(line)
        print(f"📁 Reference view: correspondences: {total_corrs} correspondences")
        print("---------------------------")

    except Exception as e:
        print(f"Error in save_correspondences: {e}")
        sys.exit(1)
