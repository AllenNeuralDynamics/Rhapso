import tensorstore as ts
import xml.etree.ElementTree as ET
import os
import numpy as np
import json

def parse_and_read_datasets(xml_file, n5_folder_base):
    print(f"\n📂 Parsing XML file: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    view_paths = {}
    for vip in root.findall(".//ViewInterestPointsFile"):
        setup_id = vip.attrib['setup']
        path = vip.text.strip()
        view_paths[setup_id] = path
    print("✅ Successfully parsed XML file.")
    print(f"🔍 Found {len(view_paths)} view ID interest point folders to analyze.")

    interest_point_info = {}
    for idx, view in enumerate(view_paths, start=1):
        print(f"\n🔗 Processing view {idx}/{len(view_paths)}: {view}")
        view_id_path = os.path.join(n5_folder_base, view_paths[view])
        interest_point_info[view] = {}
        for subfolder in ['loc', 'id']:
            path = os.path.join(view_id_path, 'interestpoints', subfolder)
            dataset = open_n5_dataset(path)
            
            if not dataset:
                print(f"⚠️ Skipping {subfolder} due to failed dataset loading.")
                continue
            
            relative_path = os.path.relpath(path, os.path.dirname(xml_file))
            print_dataset_info(dataset, relative_path)
            interest_point_info[view][subfolder] = {
                'num_items': dataset.shape[0],
                'shape': dataset.shape,
                # 'first_var': dataset[0]  # Uncomment if you want to store the first variable
            }
    
    return interest_point_info

def open_n5_dataset(n5_path):
    attributes_path = os.path.join(n5_path, 'attributes.json')
    print(f"\n🔍 Checking for attributes.json at: {attributes_path}")
    if os.path.exists(attributes_path):
        print("✅ attributes.json found, attempting to open dataset.")
        try:
            dataset = ts.open({
                'driver': 'n5',
                'kvstore': {
                    'driver': 'file',
                    'path': n5_path
                }
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
        print(f"  Data Type: {dataset.dtype}")
        print(f"  Shape: {dataset.shape}")
        data = dataset.read().result()
        print("   🟢 Raw Data (NumPy Array):\n", data)
    except Exception as e:
        print(f"❌ Error retrieving dataset info: {e}")

def create_correspondences_output(interest_point_info, n5_folder_base):
    for view, info in interest_point_info.items():
        view_id_path = os.path.join(n5_folder_base, view)
        correspondences_path = os.path.join(view_id_path, 'beads', 'correspondences')
        os.makedirs(correspondences_path, exist_ok=True)
        print(f"📂 Created correspondences folder: {correspondences_path}")

        attributes_path = os.path.join(correspondences_path, 'attributes.json')
        attributes = {
            "dataType": "uint64",
            "compression": {
                "type": "gzip",
                "useZlib": False,
                "level": -1
            },
            "blockSize": [1, 300000],
            "dimensions": [3, 2096]
        }
        with open(attributes_path, 'w') as f:
            json.dump(attributes, f, indent=4)
        print(f"📄 Created attributes.json: {attributes_path}")

        data_path = os.path.join(correspondences_path, 'data')
        os.makedirs(data_path, exist_ok=True)
        print(f"📂 Created data folder: {data_path}")

        for i in range(3):
            subfolder_path = os.path.join(data_path, str(i))
            os.makedirs(subfolder_path, exist_ok=True)
            print(f"📂 Created subfolder: {subfolder_path}")
            data_file_path = os.path.join(subfolder_path, '0')
            np.save(data_file_path, np.random.randint(0, 100, size=(2096,), dtype=np.uint64))
            print(f"📝 Created data file: {data_file_path}")

def main(xml_file, n5_folder_base):
    interest_point_info = parse_and_read_datasets(xml_file, n5_folder_base)
    print("\n📦 Collected Interest Point Info:")
    for view, info in interest_point_info.items():
        print(f"View {view}:")
        for subfolder, details in info.items():
            print(f"  {subfolder}:")
            for key, value in details.items():
                print(f"    {key}: {value}")

    # matching_tasks = generate_matching_tasks(interest_point_info)
    # filtered_points = filter_interest_points(interest_points)
    # grouped_views = group_views(views)
    # pairs = setup_pairwise_matching(grouped_views)
    # params = ransac_parameters()
    # matcher = create_matcher('FAST_ROTATION', params)

    # for pair in pairs:
    #     points1 = filtered_points[pair[0]]
    #     points2 = filtered_points[pair[1]]
    #     matches = compute_pairwise_match(matcher, points1, points2)
    #     correspondences = add_correspondences(matches, interest_points)
    #     save_results(correspondences, "output.txt")


    # create_correspondences_output(interest_point_info, n5_folder_base)

if __name__ == "__main__":
    xml_input_file = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/Just Two Tiff/IP_TIFF_XML (original) - Just Two Tiff Files - After Matching/dataset.xml"
    n5_base_path = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/Just Two Tiff/IP_TIFF_XML (original) - Just Two Tiff Files - After Matching/interestpoints.n5"
    main(xml_input_file, n5_base_path)
