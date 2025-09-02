import zarr
import s3fs
import numpy as np
import matplotlib.pyplot as plt

def visualize_points(full_path):
    if full_path.startswith("s3://"):
        s3 = s3fs.S3FileSystem(anon=False)
        store = s3fs.S3Map(root=full_path, s3=s3)
        zarray = zarr.open_array(store, mode='r')
        data = zarray[:] 

        # store = s3fs.S3Map(root=path_int, s3=s3)
        # zarray = zarr.open_array(store, mode='r')
        # data_int = zarray[:] 
    
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

        # store = zarr.N5Store(path_int)
        # root = zarr.open(store, mode='r')

        # if dataset_rel_path not in root:
        #     print(f"Skipping: {dataset_rel_path} (not found)")
        #     return

        # zarray = root[dataset_rel_path]
        # data_int = zarray[:]

    # intensities = data_int

    print("\n--- Detection Stats (Raw Rhapso Output) ---")
    print(f"Total Points: {len(data)}")
    # print(f"Intensity: min={intensities.min():.2f}, max={intensities.max():.2f}, mean={intensities.mean():.2f}, std={intensities.std():.2f}")

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
    ax.set_title(f"Interest Points in 3D Rhapso (showing {len(sample)} points)")
    plt.tight_layout()
    plt.show()

base_path = "s3://rhapso-matching-test/interestpoints.n5"
# base_path = "/Users/seanfite/Desktop/Rhapso-Output/interestpoints.n5"

for tp_id in [0]:
    for setup_id in range(20):  
        path = f"{base_path}/tpId_{tp_id}_viewSetupId_{setup_id}/beads/interestpoints/loc"
        print(f"Reading: {path}")
        visualize_points(path)