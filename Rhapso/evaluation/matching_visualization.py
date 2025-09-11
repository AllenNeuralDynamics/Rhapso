import json
import zarr
import matplotlib.pyplot as plt
import numpy as np
import s3fs

def extract_id_from_path(path, key):
    part = [p for p in path.parts if key in p][0]
    if key == "tpId":
        return int(part.split("tpId_")[1].split("_")[0])
    elif key == "viewSetupId":
        return int(part.split("viewSetupId_")[1])
    else:
        raise ValueError(f"Unsupported key: {key}")

def load_id_map(corr_path, cords_prefix, prefix):
    attr_path = corr_path + "/attributes.json"
    if attr_path.startswith("s3://"):
        s3 = s3fs.S3FileSystem(anon=False)
        store = s3fs.S3Map(root=cords_prefix, s3=s3) 
        root = zarr.open(store, mode='r')  

        target_path = f"{prefix}/beads/correspondences"

        if target_path not in root:
            return None
        
        zgroup = root[target_path]
        attrs = dict(zgroup.attrs)

        if "idMap" not in attrs:
            raise KeyError("'idMap' not found in .zattrs at the target path")

        return {int(v): k for k, v in attrs["idMap"].items()}
        
    else:
        with open(attr_path) as f:
            attrs = json.load(f)
        return {int(v): k for k, v in attrs["idMap"].items()}

def read_correspondences(corr_path):
    if str(corr_path).startswith("s3://"):
        store = zarr.storage.FSStore(corr_path, mode="r")
    else:
        store = zarr.N5Store(str(corr_path))
    root = zarr.open(store, mode="r")
    return root["data"][:]
                                        
def fetch_interest_points(cords_prefix, cords_path, id_map, correspondences):
    matches = []

    for idx, corr_idx, group in correspondences:

        # get interest point
        if cords_prefix.startswith("s3://"):
            store = zarr.storage.FSStore(cords_path, mode="r")
            root = zarr.open(store, mode="r")
        else:
            store = zarr.N5Store(cords_path)
            root = zarr.open(store, mode="r")
        
        source_point = root["loc"][idx]
        
        # get corr interest point
        if group in id_map:
            group = id_map[group]
            tp, setup_id, _ = group.split(',')
            path = f"{cords_prefix}/tpId_{tp}_viewSetupId_{setup_id}/beads/interestpoints" 
            
            if path.startswith("s3://"):
                store = zarr.storage.FSStore(path, mode="r")
                root = zarr.open(store, mode="r")
            
            else:
                store = zarr.N5Store(path)
                root = zarr.open(store, mode="r")

            corr_point = root["loc"][corr_idx]

            matches.append((source_point, corr_point, group))
           
    return matches

def get_match_coordinates(cords_path, corr_path, cords_prefix, prefix):
    id_map = load_id_map(corr_path, cords_prefix, prefix)
    if id_map != None:
        correspondences = read_correspondences(corr_path)
        matches = fetch_interest_points(cords_prefix, cords_path, id_map, correspondences)
        return matches
    else:
        return None

def plot_matches(matches, tp_id, setup_id):
    points_a = np.array([a for a, _, _ in matches])

    # Assign unique color per view key
    unique_views = sorted(set(key for _, _, key in matches))
    colormap = plt.get_cmap("tab20")
    view_color_map = {key: colormap(i % 20) for i, key in enumerate(unique_views)}

    _, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(points_a[:, 0], points_a[:, 1], c='red', label=f'View A (tp={tp_id}, setup={setup_id})', s=10)

    # Group matches by view key
    for view_key in unique_views:
        color = view_color_map[view_key]
        grouped = [(a, b) for a, b, k in matches if k == view_key]
        points_b = np.array([b for _, b in grouped])
        for a, b in grouped:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=0.5)
        ax.scatter(points_b[:, 0], points_b[:, 1], c=[color], s=10, label=view_key)

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Rhapso Rigid Matches\ntp={tp_id}, setup={setup_id}\nTotal matches: {len(matches)}")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def plot_matches_3d(matches, tp_id, setup_id):
    if not matches:
        return
    
    points_a = np.array([a for a, _, _ in matches])

    print(f"Rhapso Rigid Matches\ntp={tp_id}, setup={setup_id}\nTotal matches: {len(matches)}")

    # Assign unique color per matched view
    unique_views = sorted(set(k for _, _, k in matches))
    colormap = plt.get_cmap("tab20")
    view_color_map = {key: colormap(i % 20) for i, key in enumerate(unique_views)}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_a[:, 0], points_a[:, 1], points_a[:, 2], c='red', label=f'View A (tp={tp_id}, setup={setup_id})', s=10)

    for view_key in unique_views:
        color = view_color_map[view_key]
        grouped = [(a, b) for a, b, k in matches if k == view_key]
        points_b = np.array([b for _, b in grouped])
        for a, b in grouped:
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=0.5)
        ax.scatter(points_b[:, 0], points_b[:, 1], points_b[:, 2], c=[color], s=10, label=view_key)

    ax.set_title(f"Rhapso Rigid Matches\ntp={tp_id}, setup={setup_id}\nTotal matches: {len(matches)}")
        
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

# base_path = Path("/Users/seanfite/Desktop/IP_TIFF_XML-Rhapso-Affine/interestpoints.n5")
# base_path = "s3://aind-open-data/exaSPIM_686951_2025-02-25_09-45-02_alignment_2025-06-12_19-58-52/interest_point_detection"
# cords_prefix = "/Users/seanfite/Desktop/interest_point_detection/interestpoints.n5"
corr_prefix = "s3://rhapso-matching-test/output/interestpoints.n5"
corr_prefix = "s3://rhapso-matching-test/output/interestpoints.n"
for tp_id in [0]:
    for setup_id in range(20):
        if setup_id == 0:
            continue
        
        prefix = f"tpId_{tp_id}_viewSetupId_{setup_id}"
        corr_path = f"{corr_prefix}/tpId_{tp_id}_viewSetupId_{setup_id}/beads/correspondences"
        cords_path = f"{corr_prefix}/tpId_{tp_id}_viewSetupId_{setup_id}/beads/interestpoints"
        matches = get_match_coordinates(cords_path, corr_path, corr_prefix, prefix)
        print(f"View tp={tp_id}, setup={setup_id} -> {len(matches)} matches")
        plot_matches_3d(matches, tp_id, setup_id)