#!/usr/bin/env python3
import tensorstore as ts
import xml.etree.ElementTree as ET
import os
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors

def parse_xml(xml_file):
    print(f"\n📂 Parsing XML file: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    view_paths = {}
    for vip in root.findall(".//ViewInterestPointsFile"):
        setup_id = vip.attrib['setup']
        path = vip.text.strip()
        # Normalize the path: if it ends with "/beads", remove it so we add it only once later.
        if path.endswith("/beads"):
            print(f"🔄 Normalizing path for setup {setup_id}: Removing trailing '/beads'")
            path = path[:-len("/beads")]
        view_paths[setup_id] = path
    print("✅ Successfully parsed XML file.")
    return view_paths

def open_n5_dataset(n5_path):
    attributes_path = os.path.join(n5_path, 'attributes.json')
    print(f"\n🔍 Checking for attributes.json at: {attributes_path}")
    if os.path.exists(attributes_path):
        print("✅ attributes.json found, attempting to open dataset.")
        try:
            dataset = ts.open({
                'driver': 'n5',
                'kvstore': {'driver': 'file', 'path': n5_path}
            }).result()
            print("✅ Successfully opened N5 dataset.")
            return dataset
        except Exception as e:
            print(f"❌ Error opening N5 dataset: {e}")
            return None
    else:
        print(f"❌ No valid N5 dataset found at {n5_path} (missing attributes.json)")
        return None

def print_dataset_info(dataset, label):
    try:
        num_items = dataset.shape[0]
        shape = dataset.shape
        first_var = dataset[0]
        print(f"\n📊 Dataset Info ({label}):")
        print(f"   Number of items: {num_items}")
        print(f"   Shape: {shape}")
        print(f"   Dataset Domain: {dataset.domain}")
        print("   Dataset Properties:")
        print(f"     Data Type: {dataset.dtype}")
        print(f"     Shape: {dataset.shape}")
        data = dataset.read().result()
        print("   🟢 Raw Data (NumPy Array):\n", data)
    except Exception as e:
        print(f"❌ Error retrieving dataset info: {e}")

def parse_and_read_datasets(xml_file, n5_folder_base):
    view_paths = parse_xml(xml_file)
    print(f"\n🔍 Found {len(view_paths)} view ID interest point folders to analyze.")
    interest_point_info = {}
    for idx, view in enumerate(view_paths, start=1):
        print(f"\n🔗 Processing view {idx}/{len(view_paths)}: {view}")
        # Build the full path: join base, normalized view path, then "beads/interestpoints/loc"
        full_path = os.path.join(n5_folder_base, view_paths[view], "beads", "interestpoints", "loc")
        print(f"🛠  Loading dataset from: {full_path}")
        dataset = open_n5_dataset(full_path)
        if not dataset:
            print(f"⚠️ Skipping view {view} due to failed dataset loading.")
            continue
        relative_path = os.path.relpath(full_path, os.path.dirname(xml_file))
        print_dataset_info(dataset, relative_path)
        try:
            data = dataset.read().result()
        except Exception as e:
            print(f"❌ Error reading dataset data: {e}")
            continue
        interest_point_info[view] = {'loc': {'num_items': dataset.shape[0],
                                              'shape': dataset.shape,
                                              'data': data}}
    return interest_point_info, view_paths

def compute_matches(pointsA, pointsB, difference_threshold, ratio_of_distance):
    if pointsA is None or pointsB is None:
        print("Invalid points data provided for matching.")
        return []
    nn = NearestNeighbors(n_neighbors=2).fit(pointsB)
    distances, indices = nn.kneighbors(pointsA)
    matches = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist[0] < difference_threshold and dist[0] * ratio_of_distance <= dist[1]:
            matches.append((i, idx[0]))
    return matches

def ransac_filter_matches(pointsA, pointsB, matches, num_iterations=1000, threshold=5.0):
    """
    Performs RANSAC-based outlier removal assuming a translation model:
      pB = pA + t.
    Returns a filtered list of matches and the refined translation vector.
    """
    if len(matches) == 0:
        return matches, None

    matches_arr = np.array(matches)  # shape (M, 2)
    diff_vectors = []
    for (i, j) in matches_arr:
        diff = pointsB[j] - pointsA[i]
        diff_vectors.append(diff)
    diff_vectors = np.array(diff_vectors)  # shape (M, 3)

    best_inliers = []
    best_translation = None

    for _ in range(num_iterations):
        rand_idx = np.random.randint(0, len(matches_arr))
        t_candidate = diff_vectors[rand_idx]
        # Compute the Euclidean error for each match with respect to the candidate translation.
        errors = np.linalg.norm(diff_vectors - t_candidate, axis=1)
        inlier_indices = np.where(errors < threshold)[0]
        if len(inlier_indices) > len(best_inliers):
            best_inliers = inlier_indices
            best_translation = t_candidate
            if len(best_inliers) > 0.8 * len(matches_arr):
                break

    if best_translation is None or len(best_inliers) == 0:
        print("⚠️ RANSAC failed to find a valid model; returning original matches.")
        return matches, None

    refined_translation = np.mean(diff_vectors[best_inliers], axis=0)
    print(f"🔎 RANSAC estimated translation: {refined_translation} with {len(best_inliers)} inliers out of {len(matches_arr)} matches.")
    
    errors = np.linalg.norm(diff_vectors - refined_translation, axis=1)
    final_inlier_indices = np.where(errors < threshold)[0]
    filtered_matches = matches_arr[final_inlier_indices]
    print(f"✅ RANSAC filtering retained {len(filtered_matches)} matches after outlier removal.")
    return filtered_matches.tolist(), refined_translation

def perform_pairwise_matching(interest_point_info, view_paths):
    print("\n🔗 Starting pairwise matching across views:")
    views = list(interest_point_info.keys())
    for i in range(len(views)):
        for j in range(i+1, len(views)):
            viewA = views[i]
            viewB = views[j]
            print(f"\n💥 Matching view {viewA} with view {viewB}")
            dataA = interest_point_info[viewA]['loc']['data']
            dataB = interest_point_info[viewB]['loc']['data']
            # Transpose to get an array of shape (N,3)
            pointsA = dataA.T
            pointsB = dataB.T
            print("🔍 Computing initial matches using nearest neighbors...")
            initial_matches = compute_matches(pointsA, pointsB, float('inf'), 3.0)
            print(f"⚙️ Found {len(initial_matches)} initial matches.")
            # Now perform RANSAC to remove outliers.
            filtered_matches, t_refined = ransac_filter_matches(pointsA, pointsB, initial_matches, num_iterations=1000, threshold=5.0)
            if filtered_matches:
                for match in filtered_matches:
                    ptA = pointsA[match[0]]
                    ptB = pointsB[match[1]]
                    print(f"Matched Points (Global Coordinates): [ViewSetupId: {viewA}, TimePointId: 18, x: {ptA[0]:.2f}, y: {ptA[1]:.2f}, z: {ptA[2]:.2f}] <=> [ViewSetupId: {viewB}, TimePointId: 18, x: {ptB[0]:.2f}, y: {ptB[1]:.2f}, z: {ptB[2]:.2f}]")
            else:
                print("⚠️ No inlier matches found after RANSAC filtering.")

def main(xml_file, n5_folder_base):
    interest_point_info, view_paths = parse_and_read_datasets(xml_file, n5_folder_base)
    print("\n📦 Collected Interest Point Info:")
    for view, info in interest_point_info.items():
        print(f"View {view}:")
        for subfolder, details in info.items():
            if subfolder == 'loc':
                print(f"  {subfolder}: num_items: {details['num_items']}, shape: {details['shape']}")
            else:
                print(f"  {subfolder}: {details}")
    perform_pairwise_matching(interest_point_info, view_paths)

if __name__ == "__main__":
    xml_input_file = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/Just Two Tiff/IP_TIFF_XML (original) - Just Two Tiff Files - After Matching/dataset.xml"
    n5_base_path = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/Just Two Tiff/IP_TIFF_XML (original) - Just Two Tiff Files - After Matching/interestpoints.n5"
    main(xml_input_file, n5_base_path)
