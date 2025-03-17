import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
import json
from sklearn.neighbors import NearestNeighbors
import tensorstore as ts
import argparse

def matchViews(stuff):
    print("matching views")
    return("apple")

def buildLabelMap(xml_root):
    """
    Build a global label map for all views using the provided XML root element.

    Parameters:
    xml_root (Element): Root element of the XML tree containing view interest points.

    Returns:
    dict: A dictionary mapping ViewId objects to dictionaries of label weights.
    """
    print("building label map")
    
    label_map_global = {}
    label_weights = {}

    # Parse the XML to get view IDs and interest points
    view_setups = xml_root.findall(".//ViewSetup")
    timepoints = xml_root.findall(".//Timepoint")
    
    if not view_setups:
        print("No ViewSetup elements found in the XML.")
    if not timepoints:
        print("No Timepoint elements found in the XML.")
    
    for view_setup in view_setups:
        setup = int(view_setup.find("id").text)
        label = view_setup.find("name").text

        for timepoint in timepoints:
            timepoint_id = int(timepoint.find("id").text)
            view_id = (timepoint_id, setup)

            if label not in label_weights:
                label_weights[label] = 1.0  # Assign a default weight of 1.0 for each label

            if view_id not in label_map_global:
                label_map_global[view_id] = {}

            label_map_global[view_id][label] = label_weights[label]

    return label_map_global, label_weights


'''
Methods
'''

def parse_xml(xml_path):
    # Parse the XML file to extract view IDs, timepoints, and setups
    tree = ET.parse(xml_path)
    root = tree.getroot()
    views = []
    for view in root.findall('.//View'):
        timepoint = int(view.get('timepoint'))
        setup = int(view.get('setup'))
        views.append((timepoint, setup))
    return views

def load_interest_points(views):
    # Load interest points from the dataset based on view IDs
    interest_points = {}
    for view in views:
        timepoint, setup = view
        # Load interest points for the given timepoint and setup
        # This is a placeholder, replace with actual loading logic
        interest_points[view] = np.random.rand(100, 2)  # Example: 100 random points
    return interest_points

def filter_interest_points(interest_points):
    # Filter interest points to keep only those that overlap with other views
    filtered_points = {}
    for view, points in interest_points.items():
        # Placeholder logic for filtering
        filtered_points[view] = points[points[:, 0] > 0.5]  # Example: filter points with x > 0.5
    return filtered_points

def group_views(views):
    # Group views based on specified criteria
    grouped_views = {}
    for view in views:
        timepoint, setup = view
        if timepoint not in grouped_views:
            grouped_views[timepoint] = []
        grouped_views[timepoint].append(setup)
    return grouped_views

def setup_pairwise_matching(grouped_views):
    # Set up pairwise matching tasks
    pairs = []
    for timepoint, setups in grouped_views.items():
        for i in range(len(setups)):
            for j in range(i + 1, len(setups)):
                pairs.append(((timepoint, setups[i]), (timepoint, setups[j])))
    return pairs

def ransac_parameters():
    # Set up RANSAC parameters
    params = {
        'max_iterations': 10000,
        'max_error': 5.0,
        'min_inlier_ratio': 0.1,
        'min_inlier_factor': 3.0
    }
    return params

def create_matcher(method, params):
    # Create an instance of the matcher based on the specified parameters
    if method == 'FAST_ROTATION':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif method == 'FAST_TRANSLATION':
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        raise ValueError("Unsupported method")
    return matcher

def compute_pairwise_match(matcher, points1, points2):
    # Compute the pairwise match between two sets of interest points
    matches = matcher.match(points1, points2)
    return matches

def add_correspondences(matches, interest_points):
    # Add correspondences between interest points to the dataset
    correspondences = []
    for match in matches:
        correspondences.append((interest_points[match.queryIdx], interest_points[match.trainIdx]))
    return correspondences

def save_results(correspondences, output_path):
    # Save the resulting correspondences and matched interest points to the output file
    with open(output_path, 'w') as f:
        for corr in correspondences:
            f.write(f"{corr[0]} {corr[1]}\n")

def build_label_map(data, view_ids, label_weights):
    """
    Build a global label map for all views using the provided data and view IDs.

    Parameters:
    data (SpimData2): The SpimData2 object containing view interest points.
    view_ids (list of ViewId): List of ViewId objects representing the views.
    label_weights (dict): Dictionary mapping labels to their weights.

    Returns:
    dict: A dictionary mapping ViewId objects to dictionaries of label weights.
    """
    label_map_global = {}

    for view_id in view_ids:
        view_interest_points = data.get_view_interest_points().get_view_interest_point_lists(view_id)

        # Ensure the label exists for all views that should be processed
        for label in label_weights.keys():
            if view_interest_points.get_interest_point_list(label) is None:
                print(f"Error, label '{label}' does not exist for ViewId {view_id}")
                exit(1)

        # Needs to be ViewId, not ViewDescription, then it's serializable
        label_map_global[view_id] = label_weights

    return label_map_global

def parse_xml(xml_file):
    print(f"\n📂 Parsing XML file: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    view_paths = {}
    for vip in root.findall(".//ViewInterestPointsFile"):
        setup_id = vip.attrib['setup']
        timepoint = int(vip.attrib['timepoint'])
        path = vip.text.strip()
        if path.endswith("/beads"):
            print(f"🔄 Normalizing path for setup {setup_id}: Removing trailing '/beads'")
            path = path[:-len("/beads")]
        if setup_id in view_paths:
            if timepoint < view_paths[setup_id]['timepoint']:
                view_paths[setup_id] = {'timepoint': timepoint, 'path': path}
        else:
            view_paths[setup_id] = {'timepoint': timepoint, 'path': path}
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
        full_path = os.path.join(n5_folder_base, view_paths[view]['path'], "beads", "interestpoints", "loc")
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
    if len(matches) == 0:
        return matches, None

    matches_arr = np.array(matches)
    diff_vectors = []
    for (i, j) in matches_arr:
        diff = pointsB[j] - pointsA[i]
        diff_vectors.append(diff)
    diff_vectors = np.array(diff_vectors)

    best_inliers = []
    best_translation = None

    for _ in range(num_iterations):
        rand_idx = np.random.randint(0, len(matches_arr))
        t_candidate = diff_vectors[rand_idx]
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

def perform_pairwise_matching(interest_point_info, view_paths, all_matches, labels, method):
    print("\n🔗 Starting pairwise matching across views:")
    views = list(interest_point_info.keys())
    for i in range(len(views)):
        for j in range(i+1, len(views)):
            viewA = views[i]
            viewB = views[j]
            print(f"\n💥 Matching view {viewA} with view {viewB}")
            dataA = interest_point_info[viewA]['loc']['data']
            dataB = interest_point_info[viewB]['loc']['data']
            pointsA = dataA.T
            pointsB = dataB.T
            print("🔍 Computing initial matches using nearest neighbors...")
            initial_matches = compute_matches(pointsA, pointsB, float('inf'), 3.0)
            print(f"⚙️ Found {len(initial_matches)} initial matches.")
            filtered_matches, t_refined = ransac_filter_matches(pointsA, pointsB, initial_matches, num_iterations=1000, threshold=5.0)
            if filtered_matches:
                for match in filtered_matches:
                    ptA = pointsA[match[0]]
                    ptB = pointsB[match[1]]
                    print(f"Matched Points (Global Coordinates): [ViewSetupId: {viewA}, TimePointId: {view_paths[viewA]['timepoint']}, x: {ptA[0]:.2f}, y: {ptA[1]:.2f}, z: {ptA[2]:.2f}] <=> [ViewSetupId: {viewB}, TimePointId: {view_paths[viewB]['timepoint']}, x: {ptB[0]:.2f}, y: {ptB[1]:.2f}, z: {ptB[2]:.2f}]")
                    all_matches.append((viewA, viewB, match[0], match[1]))
            else:
                print("⚠️ No inlier matches found after RANSAC filtering.")

def save_matches_as_n5(all_matches, view_paths, n5_base_path, clear_correspondences):
    print("\n💾 Saving matches as correspondences into N5 folders...")
    timepoint = min(vp['timepoint'] for vp in view_paths.values())
    sorted_views = sorted(view_paths.keys(), key=lambda x: int(x))
    idMap = {}
    for i, view in enumerate(sorted_views):
        key = f"{timepoint},{view},beads"
        idMap[key] = i

    match_list = []
    unique_id = 0
    for (viewA, viewB, idxA, idxB) in all_matches:
        keyA = f"{timepoint},{viewA},beads"
        keyB = f"{timepoint},{viewB},beads"
        if idMap[keyA] < idMap[keyB]:
            match_list.append([idxA, idxB, unique_id])
        else:
            match_list.append([idxB, idxA, unique_id])
        unique_id += 1

    if not match_list:
        print("No matches to save.")
        return

    matches_array = np.array(match_list, dtype=np.uint64).T
    total_matches = matches_array.shape[1]

    data_attributes = {
        "dataType": "uint64",
        "compression": {
            "type": "raw"
        },
        "blockSize": [1, 300000],
        "dimensions": [3, total_matches]
    }

    for view in sorted_views:
        correspondences_folder = os.path.join(n5_base_path, view_paths[view]['path'], "beads", "correspondences")
        print(f"\n\n📁 Saving correspondences for view: {view}\n    Folder: {correspondences_folder}\n{'-'*60}")
        os.makedirs(correspondences_folder, exist_ok=True)
        
        corr_attributes = {
            "correspondences": "1.0.0",
            "idMap": idMap
        }
        attributes_path = os.path.join(correspondences_folder, "attributes.json")
        with open(attributes_path, "w") as f:
            json.dump(corr_attributes, f, indent=4)
        print(f"Saved correspondences attributes.json to {attributes_path}")

        data_folder = os.path.join(correspondences_folder, "data")
        os.makedirs(data_folder, exist_ok=True)
        for i in range(3):
            subfolder = os.path.join(data_folder, str(i))
            os.makedirs(subfolder, exist_ok=True)
        
        data_attributes_path = os.path.join(data_folder, "attributes.json")
        with open(data_attributes_path, "w") as f:
            json.dump(data_attributes, f, indent=4)
        print(f"Saved data attributes.json to {data_attributes_path}")

        for i in range(3):
            row_data = matches_array[i, :]
            byte_data = row_data.tobytes()
            output_file = os.path.join(data_folder, str(i), "0")
            with open(output_file, "wb") as f:
                f.write(byte_data)
            print(f"Saved row {i} data to {output_file} (raw binary)")

        print(f"Saved matches N5 dataset with dimensions [3, {total_matches}] in folder: {correspondences_folder}")

def main(xml_file, n5_folder_base, labels, method, clear_correspondences):
    interest_point_info, view_paths = parse_and_read_datasets(xml_file, n5_folder_base)
    print("\n📦 Collected Interest Point Info:")
    for view, info in interest_point_info.items():
        print(f"View {view}:")
        for subfolder, details in info.items():
            if subfolder == 'loc':
                print(f"  {subfolder}: num_items: {details['num_items']}, shape: {details['shape']}")
            else:
                print(f"  {subfolder}: {details}")
    all_matches = []
    perform_pairwise_matching(interest_point_info, view_paths, all_matches, labels, method)
    save_matches_as_n5(all_matches, view_paths, n5_folder_base, clear_correspondences)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise Matching Pipeline")
    parser.add_argument("xml_file", type=str, help="Path to the XML file")
    parser.add_argument("n5_folder_base", type=str, help="Base path to the N5 folder")
    parser.add_argument("-l", "--label", action="append", required=True, help="Label(s) of the interest points used for registration (e.g. -l beads -l nuclei)")
    parser.add_argument("-m", "--method", required=True, choices=["FAST_ROTATION", "FAST_TRANSLATION", "PRECISE_TRANSLATION", "ICP"], help="The matching method")
    parser.add_argument("--clearCorrespondences", action="store_true", help="Clear existing corresponding interest points for processed ViewIds and label before adding new ones (default: false)")
    args = parser.parse_args()

    main(args.xml_file, args.n5_folder_base, args.label, args.method, args.clearCorrespondences)