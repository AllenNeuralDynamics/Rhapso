# -----------------------------------------------------------------------------
# n5_reader.py – N5 data verification script
#
# Setup & run:
#   1. python -m venv n5Venv             # create a fresh Python virtual environment
#   2. source n5Venv/bin/activate        # activate the virtual environment
#   3. pip install .[n5_reader]          # install n5_reader dependencies from setup.py
#   4. python Rhapso/data_prep/n5_reader.py
#      # run the N5 reader for inspecting datasets
# -----------------------------------------------------------------------------

import zarr
import s3fs
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from zarr.storage import FSStore, DirectoryStore

def list_files_under_prefix(node, path):
    try:
        for item in node[path]:
            new_path = f"{path}/{item}"
            if isinstance(node[new_path], zarr.hierarchy.Group):
                print(f"Group: {new_path}")
                list_files_under_prefix(node, new_path)
            else:
                print(f"Dataset: {new_path} - {node[new_path].shape}")
    except KeyError:
        print(f"No items found under the path {path}")

# Amount of interest points in view 18,0 is 1061
# Max value for view 18,0 in corr ip index is 1017

def read_n5_data(n5_path):
    import zarr, s3fs, os

    # guard missing local path
    if not n5_path.startswith("s3://") and not os.path.isdir(n5_path):
        print(f"❌ Local N5 path not found: {n5_path}")
        return

    # open the store (S3 or local N5)
    if n5_path.startswith("s3://"):
        s3 = s3fs.S3FileSystem(anon=False)
        store = s3fs.S3Map(root=n5_path, s3=s3)
    else:
        store = zarr.N5Store(n5_path)

    print(f"\n🔍 Reading N5 data at: {n5_path}")
    root = zarr.open(store, mode='r')

    def visit_fn(path, node):
        if isinstance(node, zarr.Array):
            print(f"\n📂 Dataset: {path}")
            print(f"  🔢 dtype: {node.dtype}")
            shape = node.shape
            print(f"  📏 shape: {shape}")
            if len(shape) > 1:
                print(f"  📊 count: {shape[0]} arrays of shape {shape[1:]}")
            else:
                print(f"  📊 count: {shape[0]} elements")
            print(f"  🗂 chunks: {node.chunks}")
            print(f"  🛠 compressor: {node.compressor}")

            print("  🔎 first 5 entries:")
            sample = node[:5]
            for i, entry in enumerate(sample, start=1):
                # ensure nested array is printed clearly
                val = entry.tolist() if hasattr(entry, "tolist") else entry
                print(f"    {i}. {val}")

    root.visititems(visit_fn)

# read_n5_data("/home/martin/Documents/Allen/Data/exaSPIM_686951_2025-02-25_09-45-02_alignment_2025-06-12_19-58-52/ip_affine_alignment/interestpoints.n5")

def read_correspondences(dataset_path):
    store = zarr.N5Store(dataset_path)
    root = zarr.open(store, mode="r")
    group = root["data"]
    data = group[:]
    count = len(data)
    print(count)
    print("")
    return count

# Big Stitcher Output
# base_path = "/Users/seanfite/Desktop/IP_TIFF_XML/interestpoints.n5"
# for tp_id in [18, 30]:
#     for setup_id in range(5):  
#         path = f"{base_path}/tpId_{tp_id}_viewSetupId_{setup_id}/beads/correspondences"
#         print(f"Reading: {path}")
#         readr_correspondences(path)

def read_interest_points(full_path):
    if full_path.startswith("s3://"):
        # s3 = s3fs.S3FileSystem(anon=False)
        # store = s3fs.S3Map(root=full_path, s3=s3)
        # zarray = zarr.open_array(store, mode='r')
        # data = zarray[:]

        path = full_path.replace("s3://", "", 1)
        bucket = path.split("/")[0]
        prefix = "/".join(path.split("/")[1:])
        
        s3 = s3fs.S3FileSystem()
        store = FSStore(f"{bucket}/{prefix}", fs=s3, mode='r')
        root = zarr.open(store, mode="r")

        group = root["data"]
        data = group[:]
        count = len(data)
        print(count)
        print("")
        
    
    else:
        full_path = full_path.rstrip("/")  # remove trailing slash if any
        components = full_path.split("/")

        # Find index of the N5 root (assumes .n5 marks the root)
        try:
            n5_index = next(i for i, c in enumerate(components) if c.endswith(".n5"))
        except StopIteration:
            raise ValueError("No .n5 directory found in path")

        dataset_path = "/".join(components[:n5_index + 1])            # the store root
        dataset_rel_path = "/".join(components[n5_index + 1:])        # relative dataset path

        # Open N5 store and dataset
        store = zarr.N5Store(dataset_path)
        root = zarr.open(store, mode='r')

        if dataset_rel_path not in root:
            print(f"Skipping: {dataset_rel_path} (not found)")
            return

        zarray = root[dataset_rel_path]
        data = zarray[:]

    print("\n--- Detection Stats (Raw Rhapso Output) ---")
    print(f"Total Points: {len(data)}")

    for dim, name in zip(range(3), ['X', 'Y', 'Z']):
        values = data[:, dim]
        print(f"{name} Range: {values.min():.2f} – {values.max():.2f} | Spread (std): {values.std():.2f}")

    volume = np.ptp(data[:, 0]) * np.ptp(data[:, 1]) * np.ptp(data[:, 2])
    density = len(data) / (volume / 1e9) if volume > 0 else 0
    print(f"Estimated Density: {density:.2f} points per 1000³ volume")
    print("-----------------------")

    # --- 3D Plot ---
    max_points = 1000000000000
    sample = data if len(data) <= max_points else data[np.random.choice(len(data), max_points, replace=False)]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c='blue', alpha=0.5, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Interest Points in 3D (showing {len(sample)} points)")
    plt.tight_layout()
    plt.show()



def explain_n5_folder(base_path):
    import glob

    print(f"Input N5 path: {base_path}")
    
    # Detect if this is N5 or Zarr format
    def detect_store_type(path):
        """Detect if the path contains N5 or Zarr format files"""
        if os.path.exists(os.path.join(path, ".zgroup")):
            return "zarr"
        elif os.path.exists(os.path.join(path, "attributes.json")):
            return "n5"
        else:
            # Check subdirectories for format indicators
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file == ".zgroup":
                        return "zarr"
                    elif file == "attributes.json":
                        return "n5"
                if dirs:  # Only check first level to avoid deep traversal
                    break
            return "unknown"
    
    store_type = detect_store_type(base_path)
    print(f"Detected store type: {store_type}")
    
    if store_type == "unknown":
        print("⚠️  Could not determine store type. Trying N5 first...")
        store_type = "n5"
    
    tp_dirs = glob.glob(f"{base_path}/tpId_*_viewSetupId_*")
    grand_total = 0
    folder_summary = {}

    def count_n5_dataset(store, rel_path):
        try:
            root = zarr.open(store, mode='r')
            if rel_path in root:
                zarray = root[rel_path]
                data = zarray[:]
                return len(data)
        except Exception:
            pass
        return 0

    def create_store(path, store_type):
        """Create the appropriate store based on detected type"""
        if store_type == "zarr":
            # For Zarr stores, use DirectoryStore explicitly
            store = DirectoryStore(path)
            return zarr.open(store, mode='r')
        else:  # n5
            return zarr.N5Store(path)

    total_folders = len(tp_dirs)
    for idx, tp_dir in enumerate(tp_dirs, 1):
        tp_name = os.path.basename(tp_dir)
        folder_summary[tp_name] = []
        subfolders = [
            name for name in os.listdir(tp_dir)
            if os.path.isdir(os.path.join(tp_dir, name))
        ]

        print(f"\n{idx}/{total_folders} Summary for folder: {tp_name}")
        print(f"  Path: {tp_dir}")
        print(f"  {len(subfolders)} Subfolders found: {', '.join(subfolders) if subfolders else '(none)'}\n")

        for sub in subfolders:
            rel_split_dir = os.path.relpath(os.path.join(tp_dir, sub), base_path)
            
            # Create store based on detected type
            if store_type == "zarr":
                # For Zarr stores, use DirectoryStore explicitly
                store = DirectoryStore(base_path)
                store = zarr.open(store, mode='r')
            else:
                store = zarr.N5Store(base_path)

            subfolder_path = os.path.join(tp_dir, sub)
            dataset_counts = {}
            skip_exts = [".json", ".zgroup", ".zattrs", ".zarray"]
            if os.path.isdir(subfolder_path):
                for group_name in os.listdir(subfolder_path):
                    group_path = os.path.join(subfolder_path, group_name)
                    if os.path.isdir(group_path):
                        for dataset_name in os.listdir(group_path):
                            if any(dataset_name.endswith(ext) for ext in skip_exts):
                                continue
                            dataset_path = os.path.join(group_path, dataset_name)
                            rel_dataset_path = os.path.relpath(dataset_path, base_path)
                            
                            # Skip if rel_dataset_path is empty or just the current directory
                            if not rel_dataset_path or rel_dataset_path == '.':
                                print(f"    SKIPPING: rel_dataset_path is empty or current directory")
                                continue
                                
                            key = f"{group_name}/{dataset_name}"
                            count = 0
                            
                            # Try to check if this is a valid dataset path
                            try:
                                # First check if it's a directory
                                if os.path.isdir(dataset_path):
                                    # For directories, try to count them as datasets
                                    try:
                                        if store_type == "zarr":
                                            # For Zarr stores, try to access the path directly
                                            try:
                                                dataset = store[rel_dataset_path]
                                                if hasattr(dataset, 'shape'):
                                                    # This is a valid dataset
                                                    count = dataset.shape[0] if len(dataset.shape) > 0 else 0
                                                else:
                                                    count = 0
                                            except KeyError:
                                                print(f"    INFO: Path {rel_dataset_path} not found in zarr store")
                                                count = 0
                                        else:  # n5
                                            count = count_n5_dataset(store, rel_dataset_path)
                                    except Exception as e:
                                        print(f"    WARNING: Could not count dataset {rel_dataset_path}: {e}")
                                        count = 0
                                else:
                                    # For files, check if they're valid dataset paths in the store
                                    try:
                                        # Try to open the store and check if path exists
                                        if store_type == "zarr":
                                            try:
                                                dataset = store[rel_dataset_path]
                                                if hasattr(dataset, 'shape'):
                                                    # This is a valid dataset
                                                    count = dataset.shape[0] if len(dataset.shape) > 0 else 0
                                                else:
                                                    count = 0
                                            except KeyError:
                                                print(f"    INFO: Path {rel_dataset_path} not found in zarr store")
                                                count = 0
                                        else:  # n5
                                            root = zarr.open(store, mode='r')
                                            if rel_dataset_path in root:
                                                count = count_n5_dataset(store, rel_dataset_path)
                                            else:
                                                print(f"    INFO: Path {rel_dataset_path} not found in n5 store")
                                                count = 0
                                    except Exception as e:
                                        print(f"    WARNING: Could not access store for {rel_dataset_path}: {e}")
                                        count = 0
                            except Exception as e:
                                print(f"    ERROR: Unexpected error processing {rel_dataset_path}: {e}")
                                count = 0
                                
                            dataset_counts[key] = count
                    else:
                        continue

            # Print summary for this subfolder in the new format
            print(f"  {sub}/")
            if dataset_counts:
                for k, v in dataset_counts.items():
                    print(f"    - total {k}: {v:,}")
            else:
                print(f"    (no datasets found)")
            print("")  # blank line between subfolders

            entry = {"subfolder": sub}
            entry.update(dataset_counts)
            folder_summary[tp_name].append(entry)

    # Print formatted summary
    print(f"\n{'='*60}")
    print("📋 DETAILED SUMMARY BY SUBFOLDER TYPE")
    print(f"{'='*60}")
    
    # --- Detailed summary by subfolder type ---
    # Collect all keys found in folder_summary
    subfolder_totals = {}
    subfolder_folders = {}
    
    for tp_name, subfolders in folder_summary.items():
        for entry in subfolders:
            sub = entry['subfolder']
            if sub not in subfolder_totals:
                subfolder_totals[sub] = {}
                subfolder_folders[sub] = []
            for k, v in entry.items():
                if k == "subfolder":
                    continue
                if k not in subfolder_totals[sub]:
                    subfolder_totals[sub][k] = 0
                subfolder_totals[sub][k] += v
            subfolder_folders[sub].append(tp_name)

    # Print summary for each subfolder type
    for sub, totals in subfolder_totals.items():
        print(f"\n📁 Summary for subfolder '{sub}':")
        print(f"  Full path: {os.path.join(base_path, sub)}")
        
        # Print counts for each dataset type with consistent alignment
        max_key_length = max(len(k.replace('_', '/')) for k in totals.keys())
        for k, v in totals.items():
            display_key = k.replace('_', '/')
            # Pad the key to align all numbers
            padded_key = display_key.ljust(max_key_length)
            print(f"  - total {padded_key}: {v:,}")
        
        print(f"\n  📂 Appears in {len(subfolder_folders[sub])} folders:")
        
        # Sort folder names numerically and print them with individual counts
        sorted_folders = sorted(subfolder_folders[sub], key=lambda x: int(x.split('_')[-1]))
        for tp_name in sorted_folders:
            print(f"\n    📁 /{tp_name}")
            # Find the specific entry for this folder to get individual counts
            for entry in folder_summary[tp_name]:
                if entry['subfolder'] == sub:
                    for k, v in entry.items():
                        if k != 'subfolder':
                            display_key = k.replace('_', '/')
                            # Use same padding for consistent alignment
                            padded_key = display_key.ljust(max_key_length)
                            print(f"      {padded_key}: {v:,}")
                    break
        
        print("\n" + "─" * 60)
    
    # Print grand totals
    if subfolder_totals:
        all_keys = set()
        for totals in subfolder_totals.values():
            all_keys.update(totals.keys())
        
        grand_sums = {k: 0 for k in all_keys}
        for totals in subfolder_totals.values():
            for k in all_keys:
                grand_sums[k] += totals.get(k, 0)
        
        print(f"\n{'='*60}")
        print("📊 GRAND TOTALS:")
        print(f"{'='*60}")
        
        # Use consistent alignment for grand totals
        max_key_length = max(len(k.replace('_', '/')) for k in all_keys)
        for k, v in grand_sums.items():
            display_key = k.replace('_', '/')
            padded_key = display_key.ljust(max_key_length)
            print(f"  {padded_key}: {v:,}")
    
    print(f"\n{'='*60}")
    print("✅ Done.")
    print(f"{'='*60}")

#working previous code 
#base_path = "/home/martin/Documents/Allen/Data/exaSPIM_686951_2025-02-25_09-45-02_alignment_2025-06-12_19-58-52/interest_point_detection/interestpoints.n5"

# Rhapso
#base_path = "/home/martin/Documents/Allen/rhapso-e2e-testing/exaSPIM_686951/Rhapso/detection/interestpoints.n5"

# BSS
base_path = "/home/martin/Documents/Allen/rhapso-e2e-testing/exaSPIM_686951/BSS/interest_point_detection/interestpoints.n5"

explain_n5_folder(base_path)