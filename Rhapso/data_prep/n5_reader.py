import zarr
import s3fs
import os

# correspondences are the indexes of interest points, correspondences, and group (hashmap of data)
# interest points are the actual locations of all points (data)

def print_dataset_info(store_path, dataset_prefix, print_data=False, num_points=30):
    if store_path.startswith("s3://"):
        s3 = s3fs.S3FileSystem(anon=False) 
        store = s3fs.S3Map(root=store_path, s3=s3)
    else:
        store = zarr.N5Store(store_path)
        print(store_path)
    
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

    first_column = dataset[:]  
    ip_index = first_column[0]
    print(max(ip_index))
    print("hi")

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

def compare_n5_stores():
    path1 = "/Users/seanfite/Desktop/IP_TIFF_XML/output/interestpoints.n5/"
    path2 = "/Users/seanfite/Desktop/AllenInstitute/Rhapso/Data/IP_TIFF_XML/output/interestpoints.n5/"
    path3 = "s3://interest-point-detection/output/interestpoints.n5/"
    path4 = "/Users/seanfite/Desktop/interestpoints-after-matching.n5/"
    path5 = "/Users/seanfite/Desktop/interestpoints-after-solver.n5/"
    prefix = "tpId_18_viewSetupId_0/beads/correspondences/data/"
    # prefix = "tpId_18_viewSetupId_0/beads/interestpoints/loc/"

    print("\n--- Comparing N5 Stores ---")
    print("\nN5 Store:")
    print_dataset_info(path1, prefix)

    # print("\n--- Comparing N5 Stores ---")
    # print("\nN5 Store:")
    # print_dataset_info(path5, prefix)

compare_n5_stores()

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
        
        # If path with 'tpId' doesn't use the base n5_folder_base structure, try the direct tpId path
        if not os.path.exists(attributes_path) and '/tpId_' not in n5_path:
            base_dir = os.path.dirname(n5_path)
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

open_n5_dataset('/Users/seanfite/Desktop/IP_TIFF_XML/interestpoints.n5/')  


