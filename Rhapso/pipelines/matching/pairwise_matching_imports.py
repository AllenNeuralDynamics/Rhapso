'''
Run in glue 5.0 with:
--additional-python-modules s3://PATH-TO-RHAPSO.whl,bioio==1.3.0,bioio-tifffile==1.0.0,tifffile==2025.1.10,opencv-python,scikit-learn,tensorstore==0.1.56
'''
#!/usr/bin/env python3
import sys
import zarr
import s3fs
import os
import numpy as np
from Rhapso.matching.interest_point_matching import print_dataset_info, start_matching, fetch_xml_file, parse_xml, parse_and_read_datasets, perform_pairwise_matching, save_matches_as_n5

def print_dataset_info(store_path, dataset_prefix, print_data=False, num_points=30):
    print("\n=====================================")
    print("📂Checking dataset information")
    print("Store path:", store_path)
    print("Dataset prefix:", dataset_prefix)
    try:
        if store_path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=False)
            store = s3fs.S3Map(root=store_path, s3=s3, check=False)  # Ensure check=False for S3 compatibility
        else:
            store = zarr.N5Store(store_path)

        root = zarr.open(store, mode='r')
        if dataset_prefix not in root:
            raise KeyError(f"Dataset prefix '{dataset_prefix}' not found in the store.")

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

        # Add explanations for Shape and Chunks
        print("\n🔍 Explanation:")
        print(f"   - Shape: {dataset.shape} means the dataset contains {dataset.shape[0]} rows and {dataset.shape[1]} columns.")
        print(f"     Each row represents a single data point, and each column represents a property of the data (e.g., indices, match ID).")
        print(f"   - Chunks: {dataset.chunks} means the data is stored in chunks of {dataset.chunks[0]} rows and {dataset.chunks[1]} columns.")
        print(f"     This chunking is used for efficient reading and writing of large datasets.")

        if print_data:
            # Adjust slicing logic based on num_points
            if num_points == 'all':
                np.set_printoptions(threshold=sys.maxsize)  # Disable truncation for large arrays
                data_slice = dataset[:]  # Retrieve all points
            else:
                data_slice = dataset[:min(num_points, dataset.shape[0])]  # Retrieve up to num_points
            
            print("\n🟢 Data:")
            print(data_slice)

            # Print out id map if this is a correspondence dataset
            if "correspondences" in dataset_prefix:
                try:
                    id_map = dataset.attrs.get("idMap", {})
                    print("\n🔍 ID Map:")
                    for key, value in id_map.items():
                        print(f"   {key} -> {value}")
                except Exception as e:
                    print(f"\n⚠️ Could not retrieve ID map: {e}")

            # Count occurrences of each unique value in the third column
            if num_points == 'all' and data_slice.shape[1] >= 3:
                third_col = data_slice[:, 2]
                unique, counts = np.unique(third_col, return_counts=True)
                total = len(third_col)
                print("\n🔢 Third Column Value Counts:")
                for value, count in zip(unique, counts):
                    percentage = (count / total) * 100
                    print(f"   {value}: appears {count} times out of a total of {total} ({percentage:.2f}%)")
    except Exception as e:
        print(f"❌ Error accessing dataset: {e}")
    print("=====================================\n")

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
 
if __name__ == "__main__":
    try:
        # Run output detection sanity check on BSS matching output
        prefix = "tpId_18_viewSetupId_1/beads/correspondences/data/"
        #n5_bss= "/mnt/c/Users/marti/Documents/allen/data/April Dataset for Interest Points As TIFF XML (unaligned)/BSS after detection, matching IP_TIFF_XML/interestpoints.n5"
        #print_dataset_info(n5_bss, prefix, print_data=True, num_points='all') 

        # Run Rhapso matching on local data
        xml_path = "/mnt/c/Users/marti/Documents/allen/data/April Dataset for Interest Points As TIFF XML (unaligned)/BSS detection, rhapso matching/dataset.xml"
        n5_base_path = "/mnt/c/Users/marti/Documents/allen/data/April Dataset for Interest Points As TIFF XML (unaligned)/BSS detection, rhapso matching/interestpoints.n5"
        output_path = n5_base_path 
        start_matching(xml_path, n5_base_path, output_path)
        
        # Run Rhapso matching on S3 data
        #xml_path = "s3://rhapso-fused-zarr-output-data/matching_input_sample/dataset.xml"
        #n5_base_path = "s3://rhapso-fused-zarr-output-data/matching_input_sample/interestpoints.n5"
        #output_path = n5_base_path  # Set output path to be the same as input n5 data
        #start_matching(xml_path, n5_base_path, output_path)

        # Check our output dataset
        print_dataset_info(n5_base_path, prefix, print_data=True, num_points=30) 

    except Exception as e:
        print(f"❌ Unexpected error in script execution: {e}")
        sys.exit(1)
