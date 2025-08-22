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
from zarr.storage import FSStore

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

# # read_n5_data('/home/martin/Documents/Allen/Data/IP_TIFF_XML_2/interestpoints.n5')

def read_correspondences(dataset_path):
    if dataset_path.startswith("s3://"):
        store = zarr.storage.FSStore(dataset_path, mode="r")
        root = zarr.open(store, mode="r")
    else:
        store = zarr.N5Store(dataset_path)
        root = zarr.open(store, mode="r")
    
    if "data" not in root:
        print("Key 'data' not found in root.")
        return
    
    group = root["data"]
    data = group[:]
    print(f"Loaded {len(data)} entries.")

    # for i, entry in enumerate(data):
    #     print(f"{i}: {entry}")
    
    # print("hi")

# Big Stitcher Output
# # base_path = "/Users/seanfite/Desktop/interest_point_detection/interestpoints.n5"
# base_path = "/Users/seanfite/Desktop/ip_rigid_alignment/interestpoints.n5"
# # base_path = "/Users/seanfite/Desktop/ip_affine_alignment/interestpoints.n5"
# # base_path = "s3://rhapso-matching-test/output/interestpoints.n5"
# for tp_id in [0]:
#     for setup_id in range(20):  
#         path = f"{base_path}/tpId_{tp_id}_viewSetupId_{setup_id}/beads/correspondences"
#         print(f"Reading: {path}")
#         read_correspondences(path)

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
        
        zarray = root[dataset_rel_path + "/loc"]
        data = zarray[:]

    print("\n--- Detection Stats (Raw Rhapso Output) ---")
    print(f"Total Points: {len(data)}")

    # for dim, name in zip(range(3), ['X', 'Y', 'Z']):
    #     values = data[:, dim]
    #     print(f"{name} Range: {values.min():.2f} – {values.max():.2f} | Spread (std): {values.std():.2f}")

    # volume = np.ptp(data[:, 0]) * np.ptp(data[:, 1]) * np.ptp(data[:, 2])
    # density = len(data) / (volume / 1e9) if volume > 0 else 0
    # print(f"Estimated Density: {density:.2f} points per 1000³ volume")
    # print("-----------------------")

    # # --- 3D Plot ---
    # max_points = 1000000000000
    # sample = data if len(data) <= max_points else data[np.random.choice(len(data), max_points, replace=False)]

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c='blue', alpha=0.5, s=1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title(f"Interest Points in 3D (showing {len(sample)} points)")
    # plt.tight_layout()
    # plt.show()

# base_path = "s3://rhapso-matching-test/output/interestpoints.n5"
# base_path = "/Users/seanfite/Desktop/IP_TIFF_XML/interestpoints.n5"
base_path = "/Users/seanfite/Desktop/interestpoints.n5"
for tp_id in [0]:
    for setup_id in range(20): 
        path = f"{base_path}/tpId_{tp_id}_viewSetupId_{setup_id}/beads/interestpoints"
        print(f"For view: {setup_id}")
        read_interest_points(path)