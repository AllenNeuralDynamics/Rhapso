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
from Rhapso.matching.xml_parser import XMLParser
from Rhapso.matching.data_loader import DataLoader
from Rhapso.matching.matcher import Matcher
from Rhapso.matching.result_saver import ResultSaver
from Rhapso.matching.ransac import RANSAC

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

def main(xml_file, n5_folder_base, output_path):
    try:
        # Parse XML file
        xml_parser = XMLParser(xml_file)
        datasets = xml_parser.parse()

        # Load data
        data_loader = DataLoader(n5_folder_base)
        data = data_loader.load(datasets)

        # Perform matching
        matcher = Matcher()
        ransac = RANSAC(iterations=1000, threshold=10.0)
        matches = matcher.match(data)
        filtered_matches, translation = ransac.filter_matches(data['pointsA'], data['pointsB'], matches)

        # Save results
        result_saver = ResultSaver(output_path)
        result_saver.save(filtered_matches)

        print("✅ Matching process completed successfully.")
    except Exception as e:
        print(f"❌ Error during matching process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    xml_file = "/path/to/dataset.xml"
    n5_folder_base = "/path/to/n5/folder"
    output_path = "/path/to/output"
    main(xml_file, n5_folder_base, output_path)
