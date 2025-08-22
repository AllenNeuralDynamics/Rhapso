import zarr
import s3fs
import os
import numpy as np
import matplotlib.pyplot as plt
import json

def read_big_stitcher_output(dataset_path):

    if dataset_path.startswith("s3://"):
        s3 = s3fs.S3FileSystem(anon=False)
        attr_path = os.path.join(dataset_path, "attributes.json")
        with s3.open(attr_path, 'r') as f:
            json.load(f)

        store_root = os.path.dirname(dataset_path.rstrip("/"))
        dataset_name = dataset_path.rstrip("/").split("/")[-1]
        store = zarr.N5Store(s3fs.S3Map(root=store_root, s3=s3))
    
    else:
        attr_path = os.path.join(dataset_path, "attributes.json")
        with open(attr_path) as f:
            json.load(f)
        store_root = os.path.dirname(dataset_path.rstrip("/"))
        dataset_name = dataset_path.rstrip("/").split("/")[-1]
        store = zarr.N5Store(store_root)

    root = zarr.open(store, mode="r")
    group = root[dataset_name]

    intensities = root['intensities'][:]

    # It's a Zarr array with shape (N, 3)
    data = group[:]

    # Print points sorted by index n
    # sorted_data = data[data[:, 2].argsort()]
    # for i, row in enumerate(data):
    #     print(f"{i:3d}: {row}")

    # General metrics
    print("\n--- Detection Stats (Raw BigStitcher Output) ---")
    print(f"Total Points: {len(data)}")
    print(f"Intensity: min={intensities.min():.2f}, max={intensities.max():.2f}, mean={intensities.mean():.2f}, std={intensities.std():.2f}")
    for dim, name in zip(range(3), ['X', 'Y', 'Z']):
        values = data[:, dim]
        print(f"{name} Range: {values.min():.2f} – {values.max():.2f} | Spread (std): {values.std():.2f}")
    volume = np.ptp(data[:, 0]) * np.ptp(data[:, 1]) * np.ptp(data[:, 2])
    density = len(data) / (volume / 1e9) if volume > 0 else 0
    print(f"Estimated Density: {density:.2f} points per 1000³ volume")
    print("--------------------------------------------------\n")

    # --- 3D Plot ---
    max_points = 10000000
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

def read_rhapso_output(full_path):
    if full_path.startswith("s3://"):
        s3 = s3fs.S3FileSystem(anon=False)
        store = s3fs.S3Map(root=full_path, s3=s3)
        zarray = zarr.open_array(store, mode='r')
        data = zarray[:]
    
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

# base_path = "s3://rhapso-matching-test/output/interestpoints.n5"
base_path = "/Users/seanfite/Desktop/interest_point_detection/interestpoints.n5"

for tp_id in [0]:
    for setup_id in range(20):  
        path = f"{base_path}/tpId_{tp_id}_viewSetupId_{setup_id}/beads/interestpoints/loc"
        print(f"Reading: {path}")
        # read_big_stitcher_output(path)
        read_rhapso_output(path)