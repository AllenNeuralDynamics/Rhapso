import xml.etree.ElementTree as ET
import cv2
import numpy as np
import boto3
import os
import json
from sklearn.neighbors import NearestNeighbors
import tensorstore as ts
from urllib.parse import urlparse 
import sys
import zarr
import s3fs

def print_dataset_info(store_path, dataset_prefix, print_data=False, num_points=30):
    if store_path.startswith("s3://"):
        s3 = s3fs.S3FileSystem(anon=False)
        store = s3fs.S3Map(root=store_path, s3=s3)
    else:
        store = zarr.N5Store(store_path)
    
    root = zarr.open(store, mode='r')
    dataset = root[dataset_prefix]
 
    print(f"Information for dataset at {store_path} in prefix {dataset_prefix}:")
    print("Data Type:", dataset.dtype)
    print("Shape:", dataset.shape)
    print("Chunks:", dataset.chunks)
    print("Compression:", dataset.compressor)
    if dataset.attrs:
        print("Attributes:")
        for attr, value in dataset.attrs.items():
            print(f"  {attr}: {value}")
    
    # Adjust slicing logic based on num_points
    if num_points == 'all':
        data_slice = dataset[:]  # Retrieve all points
        print(f"\nDisplaying all {dataset.shape[0]} points:")
    else:
        points_to_show = min(num_points, dataset.shape[0])
        data_slice = dataset[:points_to_show]  # Retrieve up to num_points
        print(f"\nDisplaying first {points_to_show} points out of {dataset.shape[0]} total points:")

    print(data_slice)
 
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

def fetch_from_s3(s3, bucket_name, input_file):
    response = s3.get_object(Bucket=bucket_name, Key=input_file)
    return response['Body'].read().decode('utf-8')

def fetch_from_local(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def fetch_xml_file(file_location):
    if file_location.startswith("s3://"):
        s3 = boto3.client('s3')
        parsed_url = urlparse(file_location)
        bucket_name = parsed_url.netloc
        input_file = parsed_url.path.lstrip('/')
        xml_content = fetch_from_s3(s3, bucket_name, input_file)
    else:
        xml_content = fetch_from_local(file_location)
    return xml_content


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

# def load_interest_points(views, xml_path, n5_base_path):
#     """
#     Load interest points from the dataset based on view IDs.

#     Parameters:
#     views (list): List of tuples containing (timepoint, setup).
#     xml_path (str): Path to the XML file.
#     n5_base_path (str): Base path to the .n5 folder.

#     Returns:
#     dict: A dictionary mapping view IDs to their interest points.
#     """
#     interest_points = {}
#     for view in views:
#         timepoint, setup = view
#         view_folder = f"tpId_{timepoint}_viewSetupId_{setup}/beads/interestpoints/loc"
#         full_path = os.path.join(n5_base_path, view_folder)

#         try:
#             # Open the N5 dataset
#             if n5_base_path.startswith("s3://"):
#                 s3 = s3fs.S3FileSystem(anon=False)
#                 store = s3fs.S3Map(root=full_path, s3=s3)
#             else:
#                 store = zarr.N5Store(full_path)

#             root = zarr.open(store, mode='r')
#             loc_dataset = root["0/0"]  # Assuming the interest points are stored in "0/0"

#             # Read the interest points
#             points = loc_dataset[:]
#             interest_points[view] = points

#         except Exception as e:
#             print(f"⚠️ Failed to load interest points for view {view}: {e}")
#             interest_points[view] = None  # Mark as None if loading fails

#     return interest_points

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
            print(f"🔄 Normalizing path for setup {setup_id} at timepoint {timepoint}: Removing trailing '/beads'")
            path = path[:-len("/beads")]
        
        # Use tuple of (timepoint, setup_id) as key to handle multiple timepoints
        key = (timepoint, setup_id)
        view_paths[key] = {'timepoint': timepoint, 'path': path, 'setup': setup_id}
    
    print(f"✅ Successfully parsed XML file. Found {len(view_paths)} view/timepoint combinations.")
    return view_paths

def open_n5_dataset(n5_path):
    # Check if the direct path exists
    attributes_path = os.path.join(n5_path, 'attributes.json')
    print(f"\n🔍 Checking for attributes.json at: {attributes_path}")
    
    if not os.path.exists(attributes_path):
        # Try the alternate path structure by removing 'interestpoints.n5' if it exists in the path
        if 'interestpoints.n5' in n5_path:
            alt_path = n5_path.replace('interestpoints.n5/', '')
            alt_path = alt_path.replace('interestpoints.n5', '')
            attributes_path = os.path.join(alt_path, 'attributes.json')
            print(f"🔄 Path not found. Trying alternate path: {attributes_path}")
        
        # If path still doesn't exist and doesn't use the base n5_folder_base structure, try the direct tpId path
        if not os.path.exists(attributes_path) and '/tpId_' not in n5_path:
            base_dir = os.path.dirname(n5_path)
            if os.path.exists(base_dir):
                for item in os.listdir(base_dir):
                    if item.startswith('tpId_'):
                        view_id = item.split('_')[3]
                        if f'viewSetupId_{view_id}' in n5_path:
                            alt_path = os.path.join(base_dir, item, 'beads', 'interestpoints', 'loc')
                            attributes_path = os.path.join(alt_path, 'attributes.json')
                            print(f"🔄 Path not found. Trying tpId-based path: {attributes_path}")
                            if os.path.exists(attributes_path):
                                n5_path = alt_path
                                break
    
    if os.path.exists(attributes_path):
        print(f"✅ attributes.json found at: {attributes_path}")
        try:
            dataset = ts.open({
                'driver': 'n5',
                'kvstore': {
                    'driver': 'file', 
                    'path': os.path.dirname(attributes_path)
                }
            }).result()
            print("✅ Successfully opened N5 dataset.")
            return dataset
        except Exception as e:
            print(f"❌ Error opening N5 dataset: {e}")
            return None
    else:
        print(f"❌ No valid N5 dataset found at {n5_path} (missing attributes.json)")
        
        # Debug info - list nearby directories to help troubleshoot
        try:
            base_dir = os.path.dirname(n5_path)
            print(f"📁 Contents of parent directory {base_dir}:")
            if os.path.exists(base_dir):
                for item in os.listdir(base_dir):
                    print(f"   - {item}")
                    
                # If parent directory contains tpId folders, print their structure
                if any(i.startswith('tpId_') for i in os.listdir(base_dir)):
                    for item in os.listdir(base_dir):
                        if item.startswith('tpId_'):
                            full_path = os.path.join(base_dir, item)
                            print(f"   📂 {item} contents:")
                            if os.path.isdir(full_path):
                                for subitem in os.listdir(full_path):
                                    print(f"      - {subitem}")
            else:
                print(f"   Directory does not exist")
        except Exception as e:
            print(f"   Error listing directory: {e}")
            
        return None

def print_dataset_info(dataset, label):
    try:
        # Add a type check to ensure the dataset is valid
        if not hasattr(dataset, 'shape'):
            raise ValueError(f"Invalid dataset object for label '{label}'. Expected an object with a 'shape' attribute, got {type(dataset)}.")
        
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
    print(f"\n🔍 Found {len(view_paths)} view ID/timepoint interest point folders to analyze.")
    interest_point_info = {}
    for idx, view_key in enumerate(view_paths, start=1):
        timepoint, setup_id = view_key
        view_info = view_paths[view_key]
        print(f"\n🔗 Processing view {idx}/{len(view_paths)}: Setup {setup_id}, Timepoint {timepoint}")
        full_path = os.path.join(n5_folder_base, view_info['path'], "beads", "interestpoints", "loc")
        print(f"🛠  Loading dataset from: {full_path}")
        dataset = open_n5_dataset(full_path)
        if not dataset:
            print(f"⚠️ Skipping view {setup_id} at timepoint {timepoint} due to failed dataset loading.")
            continue
        relative_path = os.path.relpath(full_path, os.path.dirname(xml_file))
        print_dataset_info(dataset, relative_path)
        try:
            data = dataset.read().result()
        except Exception as e:
            print(f"❌ Error reading dataset data: {e}")
            continue
        interest_point_info[view_key] = {'loc': {'num_items': dataset.shape[0],
                                          'shape': dataset.shape,
                                          'data': data}}
    return interest_point_info, view_paths

def compute_matches(pointsA, pointsB, difference_threshold=100.0, ratio_of_distance=5.0):
    """Relaxed matching criteria with higher threshold and ratio"""
    if pointsA is None or pointsB is None:
        print("Invalid points data provided for matching.")
        return []
    nn = NearestNeighbors(n_neighbors=2).fit(pointsB)
    distances, indices = nn.kneighbors(pointsA)
    matches = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        # More lenient distance threshold and ratio check
        if dist[0] < difference_threshold and dist[0] * ratio_of_distance <= dist[1]:
            matches.append((i, idx[0]))
    return matches

def ransac_filter_matches(pointsA, pointsB, matches, num_iterations=1000, threshold=10.0):
    """Relaxed RANSAC parameters with higher threshold"""
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
        # More lenient inlier threshold
        inlier_indices = np.where(errors < threshold)[0]
        if len(inlier_indices) > len(best_inliers):
            best_inliers = inlier_indices
            best_translation = t_candidate
            # Relaxed early termination condition
            if len(best_inliers) > 0.6 * len(matches_arr):  # Reduced from 0.8
                break

    if best_translation is None or len(best_inliers) == 0:
        print("⚠️ RANSAC failed to find a valid model; returning original matches.")
        return matches, None

    refined_translation = np.mean(diff_vectors[best_inliers], axis=0)
    print(f"🔎 RANSAC estimated translation: {refined_translation} with {len(best_inliers)} inliers out of {len(matches_arr)} matches.")
    
    # More lenient final filtering
    errors = np.linalg.norm(diff_vectors - refined_translation, axis=1)
    final_inlier_indices = np.where(errors < threshold * 1.5)[0]  # Increased threshold
    filtered_matches = matches_arr[final_inlier_indices]
    print(f"✅ RANSAC filtering retained {len(filtered_matches)} matches after outlier removal.")
    return filtered_matches.tolist(), refined_translation

def perform_pairwise_matching(interest_point_info, view_paths, all_matches, labels, method):
    print("\n🔗 Starting pairwise matching across views:")
    
    # Group views by timepoint
    timepoint_groups = {}
    for view_key in interest_point_info.keys():
        timepoint, setup_id = view_key
        if timepoint not in timepoint_groups:
            timepoint_groups[timepoint] = []
        timepoint_groups[timepoint].append(view_key)
    
    # Process each timepoint separately
    for timepoint, views in timepoint_groups.items():
        print(f"\n⏱️  Processing timepoint {timepoint} with {len(views)} views")
        for i in range(len(views)):
            for j in range(i+1, len(views)):
                viewA = views[i]
                viewB = views[j]
                tp_A, setup_A = viewA
                tp_B, setup_B = viewB
                
                print(f"\n💥 Matching view {setup_A} with view {setup_B} at timepoint {timepoint}")
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
                        print(f"Matched Points (Global Coordinates): [ViewSetupId: {setup_A}, TimePointId: {tp_A}, x: {ptA[0]:.2f}, y: {ptA[1]:.2f}, z: {ptA[2]:.2f}] <=> [ViewSetupId: {setup_B}, TimePointId: {tp_B}, x: {ptB[0]:.2f}, y: {ptB[1]:.2f}, z: {ptB[2]:.2f}]")
                        all_matches.append((viewA, viewB, match[0], match[1]))
                else:
                    print("⚠️ No inlier matches found after RANSAC filtering.")


def read_interest_points(store_path, dataset_path):
    """Read interest points from N5/Zarr store"""
    try:
        if store_path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            store = s3fs.S3Map(root=store_path, s3=s3, check=False)
        else:
            store = zarr.N5Store(store_path)
            
        root = zarr.open(store, mode='r')
        
        # Check if the path exists
        if dataset_path not in root:
            print(f"⚠️ Dataset path {dataset_path} not found")
            return None
            
        # Read interest points data
        loc_path = f"{dataset_path}/interestpoints/loc"
        if loc_path in root:
            return root[loc_path][:]
        else:
            print(f"⚠️ Location data not found at {loc_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error reading interest points: {e}")
        return None

def save_matches_as_n5(all_matches, view_paths, n5_base_path, clear_correspondences):
    print("\n💾 Saving matches as correspondences into N5 folders...")
    
    # Determine if we're using S3
    using_s3 = n5_base_path.startswith("s3://")
    local_temp_dir = "/tmp/n5_temp_output"
    
    if using_s3:
        # Parse S3 path
        parsed_url = urlparse(n5_base_path)
        bucket_name = parsed_url.netloc
        s3_prefix = parsed_url.path.lstrip('/')
        
        # Create local temp directory
        os.makedirs(local_temp_dir, exist_ok=True)
        local_n5_path = os.path.join(local_temp_dir, "output.n5")
    else:
        local_n5_path = n5_base_path

    try:
        # Create local N5 store
        store = zarr.N5Store(local_n5_path)
        root = zarr.group(store=store, overwrite=False)
        root.attrs['n5'] = '4.0.0'

        # Group matches by timepoint
        timepoint_matches = {}
        for match in all_matches:
            viewA, viewB, idxA, idxB = match
            timepoint, _ = viewA
            if timepoint not in timepoint_matches:
                timepoint_matches[timepoint] = []
            timepoint_matches[timepoint].append(match)

        # Process each timepoint's matches
        for timepoint, matches in timepoint_matches.items():
            print(f"\n⏱️  Processing matches for timepoint {timepoint}")
            
            # Get all views for this timepoint
            all_views_for_timepoint = [view_key for view_key in view_paths.keys() 
                                     if view_key[0] == timepoint]
            all_setups = sorted({view_key[1] for view_key in all_views_for_timepoint}, 
                              key=lambda x: int(x))
            
            # Create ID map
            idMap = {f"{timepoint},{setup_id},beads": i 
                    for i, setup_id in enumerate(all_setups)}

            if not matches:
                print(f"No matches to save for timepoint {timepoint}.")
                continue

            # Prepare match data
            match_data = []
            for (viewA, viewB, idxA, idxB) in matches:
                _, setup_A = viewA
                _, setup_B = viewB
                keyA = f"{timepoint},{setup_A},beads"
                keyB = f"{timepoint},{setup_B},beads"
                
                # Dynamically determine matchId based on idMap
                if idMap[keyA] < idMap[keyB]:
                    matchId = idMap[keyA]
                else:
                    matchId = idMap[keyB]

                if idMap[keyA] < idMap[keyB]:
                    match_data.append([idxA, idxB, matchId])
                else:
                    match_data.append([idxB, idxA, matchId])

            # Convert to numpy array
            matches_array = np.array(match_data, dtype=np.uint64)
            
            # Save for each view
            for view_key in all_views_for_timepoint:
                _, setup_id = view_key
                group_path = f"tpId_{timepoint}_viewSetupId_{setup_id}/beads"
                print(f"\n📁 Processing view {setup_id} at path: {group_path}")
                
                try:
                    # Create groups
                    if group_path in root:
                        group = root[group_path]
                    else:
                        group = root.create_group(group_path)
                    
                    # Handle correspondences
                    if 'correspondences' in group:
                        if clear_correspondences:
                            del group['correspondences']
                            correspondences = group.create_group('correspondences')
                        else:
                            correspondences = group['correspondences']
                    else:
                        correspondences = group.create_group('correspondences')
                    
                    # Set attributes
                    correspondences.attrs.update({
                        "pointcloud": "1.0.0",
                        "type": "list",
                        "list version": "1.0.0",
                        "idMap": idMap,
                        "timepoint": timepoint
                    })
                    
                    # Save matches data
                    if 'data' in correspondences:
                        del correspondences['data']
                    correspondences.create_dataset(
                        "data",
                        data=matches_array,
                        dtype='uint64',
                        chunks=(min(300000, matches_array.shape[0]),),
                        compressor=zarr.GZip()
                    )
                    
                    print(f"✅ Saved {matches_array.shape[0]} matches for view {setup_id}")
                    
                except Exception as e:
                    print(f"❌ Error processing view {setup_id}: {str(e)}")
                    continue

        # Upload to S3 if needed
        if using_s3:
            print("\n📤 Uploading results to S3...")
            s3 = boto3.client('s3')
            
            # Walk through local directory and upload files
            for root_dir, dirs, files in os.walk(local_n5_path):
                for file in files:
                    local_path = os.path.join(root_dir, file)
                    relative_path = os.path.relpath(local_path, local_n5_path)
                    s3_key = f"{s3_prefix}/{relative_path}"
                    
                    print(f"Uploading {relative_path} to s3://{bucket_name}/{s3_key}")
                    with open(local_path, 'rb') as f:
                        s3.upload_fileobj(f, bucket_name, s3_key)
            
            # Clean up
            import shutil
            shutil.rmtree(local_temp_dir)
            print("✅ Successfully uploaded results to S3")

        return True

    except Exception as e:
        print(f"❌ Error in save_matches_as_n5: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(xml_file, n5_folder_base, labels, method, clear_correspondences):
    print("main")

def fetch_n5_folder(s3_path, local_temp_dir="/tmp/n5_temp"):
    try:
        if s3_path.startswith("s3://"):
            print(f"📥 Fetching N5 folder from S3: {s3_path}")
            s3 = boto3.client('s3')
            parsed_url = urlparse(s3_path)
            bucket_name = parsed_url.netloc
            prefix = parsed_url.path.lstrip('/')
            local_path = os.path.join(local_temp_dir, os.path.basename(prefix))
            os.makedirs(local_path, exist_ok=True)
            print(f"📂 Local directory for N5 folder: {local_path}")
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    local_file_path = os.path.join(local_temp_dir, key[len(prefix):].lstrip('/'))
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    s3.download_file(bucket_name, key, local_file_path)
                    print(f"  📄 Downloaded: {local_file_path}")
            print(f"✅ N5 folder downloaded to: {local_path}")
            return local_path
        else:
            print(f"📂 Using local N5 folder: {s3_path}")
            return s3_path
    except Exception as e:
        print(f"❌ Error fetching N5 folder: {e}")
        sys.exit(1)


def start_matching(xml_file, n5_folder_base, output_s3_path=None):
    try:
        if xml_file.startswith("s3://"):
            print(f"📥 Fetching XML file from S3: {xml_file}")
            xml_file = fetch_xml_file(xml_file)
            local_xml_path = "/tmp/xml_temp.xml"
            with open(local_xml_path, "w") as f:
                f.write(xml_file)
            print(f"📂 Local XML file saved to: {local_xml_path}")
            xml_file = local_xml_path
        else:
            print(f"📂 Using local XML file: {xml_file}")

        # Parse the XML to get all defined timepoints
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get all timepoints defined in the XML
        all_timepoints = set()
        
        # First, try to find timepoints in a "list" type format
        timepoints_list = root.find(".//Timepoints[@type='list']")
        if timepoints_list is not None:
            for timepoint in timepoints_list.findall("Timepoint"):
                all_timepoints.add(int(timepoint.find("id").text))
        
        # If no timepoints found, try "range" format
        if not all_timepoints:
            timepoint_range = root.find(".//Timepoints[@type='range']")
            if timepoint_range is not None:
                first = int(timepoint_range.find("first").text)
                last = int(timepoint_range.find("last").text)
                all_timepoints = set(range(first, last + 1))
        
        # If still no timepoints found, try to get timepoints from ViewInterestPointsFile elements
        if not all_timepoints:
            for vip in root.findall(".//ViewInterestPointsFile"):
                all_timepoints.add(int(vip.attrib['timepoint']))
        
        if all_timepoints:
            print(f"📊 XML file contains {len(all_timepoints)} timepoints: {sorted(all_timepoints)}")
        else:
            print("⚠️ Warning: Could not find any timepoints defined in the XML file")
            all_timepoints = {0}  # Default to avoid errors

        if n5_folder_base.startswith("s3://"):
            n5_folder_base = fetch_n5_folder(n5_folder_base)
        
        # If output path is not specified, use the same as input n5 folder
        if output_s3_path is None:
            output_s3_path = n5_folder_base

        labels = ["beads"]
        method = "FAST_ROTATION"
        clear_correspondences = False

        # Parse the XML and parse datasets using the updated functions that handle multiple timepoints
        interest_point_info, view_paths = parse_and_read_datasets(xml_file, n5_folder_base)
        print("\n📦 Collected Interest Point Info:")
        
        # Group by timepoint for better visualization
        loaded_timepoints = set(tp for (tp, _) in interest_point_info.keys())
        for timepoint in sorted(loaded_timepoints):
            print(f"\n⏱️  Timepoint {timepoint}:")
            for view_key, info in interest_point_info.items():
                tp, setup = view_key
                if tp == timepoint:
                    print(f"  View {setup}:")
                    for subfolder, details in info.items():
                        if subfolder == 'loc':
                            print(f"    {subfolder}: num_items: {details['num_items']}, shape: {details['shape']}")
                        else:
                            print(f"    {subfolder}: {details}")
        
        # Report on missing timepoints
        missing_timepoints = all_timepoints - loaded_timepoints
        if missing_timepoints:
            print(f"\n⚠️ Warning: Could not load interest points for {len(missing_timepoints)} timepoints: {sorted(missing_timepoints)}")
            print("   This could be because the interest point files don't exist or couldn't be accessed.")
                            
        all_matches = []
        perform_pairwise_matching(interest_point_info, view_paths, all_matches, labels, method)
        save_matches_as_n5(all_matches, view_paths, output_s3_path, clear_correspondences)
        
        # Print comprehensive summary
        print(f"\n✅ Successfully processed XML file with {len(all_timepoints)} timepoints defined in XML")
        print(f"✅ Successfully loaded interest points for {len(loaded_timepoints)} timepoints")
        print(f"✅ Generated and saved {len(all_matches)} matches across all loaded timepoints")
        print(f"✅ Output saved to: {output_s3_path}")
    except Exception as e:
        print(f"❌ Error in start_matching function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise Matching Pipeline")
    parser.add_argument("xml_file", type=str, help="Path to the XML file")
    parser.add_argument("n5_folder_base", type=str, help="Base path to the N5 folder")
    parser.add_argument("-l", "--label", action="append", required=True, help="Label(s) of the interest points used for registration (e.g. -l beads -l nuclei)")
    parser.add_argument("-m", "--method", required=True, choices=["FAST_ROTATION", "FAST_TRANSLATION", "PRECISE_TRANSLATION", "ICP"], help="The matching method")
    parser.add_argument("--clearCorrespondences", action="store_true", help="Clear existing corresponding interest points for processed ViewIds and label before adding new ones (default: false)")
    args = parser.parse_args()

    main(args.xml_file, args.n5_folder_base, args.label, args.method, args.clearCorrespondences)