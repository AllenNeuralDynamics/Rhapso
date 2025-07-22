import json
import zarr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import os

def extract_id_from_path(path: Path, key: str) -> int:
    part = [p for p in path.parts if key in p][0]
    if key == "tpId":
        return int(part.split("tpId_")[1].split("_")[0])
    elif key == "viewSetupId":
        return int(part.split("viewSetupId_")[1])
    else:
        raise ValueError(f"Unsupported key: {key}")

def load_id_map(corr_path: Path) -> Dict[int, str]:
    attr_path = corr_path.parent / "correspondences/attributes.json"
    with open(attr_path) as f:
        attrs = json.load(f)
    return {int(v): k for k, v in attrs["idMap"].items()}

def read_correspondences(corr_path: Path) -> List[Tuple[int, int, int]]:
    store = zarr.N5Store(str(corr_path))
    root = zarr.open(store, mode="r")
    return root["data"][:]
                                        
def fetch_interest_points(base_path: Path, tp: int, setup: int, label: str) -> np.ndarray:
    ip_path = str(base_path / f"interestpoints.n5/tpId_{tp}_viewSetupId_{setup}" / label / "interestpoints" / "loc")
    store_root = os.path.dirname(ip_path.rstrip("/"))
    dataset_name = ip_path.rstrip("/").split("/")[-1]
    
    store = zarr.N5Store(store_root)
    root = zarr.open(store, mode="r")
    group = root[dataset_name]
    data = group[:]
    
    return data

def get_match_coordinates(corr_path: Path) -> Tuple[List[Tuple[np.ndarray, np.ndarray, str]], int, int, str]:
    base_path = corr_path.parents[3]
    tp_id = extract_id_from_path(corr_path, "tpId")
    setup_id = extract_id_from_path(corr_path, "viewSetupId")
    label = corr_path.parts[-2]

    id_map = load_id_map(corr_path)
    correspondences = read_correspondences(corr_path)
    local_points = fetch_interest_points(base_path, tp_id, setup_id, label)

    match_coords = []
    for idx_a, idx_b, view_id in correspondences:
        key = id_map[view_id]
        tp, setup, _ = key.split(",")
        remote_points = fetch_interest_points(base_path, int(tp), int(setup), label)
        match_coords.append((local_points[idx_a], remote_points[idx_b], key))

    return match_coords, tp_id, setup_id, label

def plot_matches(matches: List[Tuple[np.ndarray, np.ndarray, str]], tp_id: int, setup_id: int, label: str, dim=2):
    points_a = np.array([a for a, _, _ in matches])

    # Assign unique color per view key
    unique_views = sorted(set(key for _, _, key in matches))
    colormap = plt.get_cmap("tab20")
    view_color_map = {key: colormap(i % 20) for i, key in enumerate(unique_views)}

    fig, ax = plt.subplots(figsize=(10, 10))
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
    ax.set_title(f"Rhapso Affine Matching\ntp={tp_id}, setup={setup_id}\nTotal matches: {len(matches)}")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def plot_matches_3d(matches: List[Tuple[np.ndarray, np.ndarray, str]], tp_id: int, setup_id: int, label: str):
    points_a = np.array([a for a, _, _ in matches])

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

    ax.set_title(f"Rhapso Affine Matches\ntp={tp_id}, setup={setup_id}\nTotal matches: {len(matches)}")
        
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

# base_path = Path("/Users/seanfite/Desktop/IP_TIFF_XML-Rhapso-Affine/interestpoints.n5")
base_path = Path("/Users/seanfite/Desktop/IP_TIFF_XML-Rhapso-Affine/interestpoints.n5")
for tp_id in [18, 30]:
    for setup_id in range(5):
        path = base_path / f"tpId_{tp_id}_viewSetupId_{setup_id}" / "beads" / "correspondences"
        if path.exists():
            matches, tp, setup, label = get_match_coordinates(path)
            print(f"View tp={tp}, setup={setup} -> {len(matches)} matches")
            plot_matches_3d(matches, tp, setup, label)