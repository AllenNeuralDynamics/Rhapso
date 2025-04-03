'''
Run in glue 5.0 with:
--additional-python-modules s3://PATH-TO-RHAPSO.whl,bioio==1.3.0,bioio-tifffile==1.0.0,tifffile==2025.1.10,opencv-python,scikit-learn,tensorstore==0.1.56
'''
#!/usr/bin/env python3
import sys
import zarr
import s3fs
import os
from Rhapso.matching.interest_point_matching import print_dataset_info, start_matching, fetch_xml_file, parse_xml, parse_and_read_datasets, perform_pairwise_matching, save_matches_as_n5

def print_dataset_info(store_path, dataset_prefix, print_data=False, num_points=30):
    print("\n=====================================")
    print("📂Checking dataset information")
    print("Store path:", store_path)
    print("Dataset prefix:", dataset_prefix)
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
 
    data_slice = dataset[:min(30, dataset.shape[1])]
    print(data_slice)
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

        # Check working dataset:
        sanity_n5_path = "/mnt/c/Users/marti/Documents/allen/data/April Dataset for Interest Points As TIFF XML (unaligned)/BSS after detection, matching IP_TIFF_XML/interestpoints.n5"
        prefix = "tpId_18_viewSetupId_1/beads/correspondences/data/"
        print_dataset_info(sanity_n5_path, prefix)

        # Run Rhapso matching
        xml_path = "/mnt/c/Users/marti/Documents/allen/data/April Dataset for Interest Points As TIFF XML (unaligned)/BSS detection, rhapso matching/dataset.xml"
        n5_base_path = "/mnt/c/Users/marti/Documents/allen/data/April Dataset for Interest Points As TIFF XML (unaligned)/BSS detection, rhapso matching/interestpoints.n5"
        output_path = n5_base_path 
        # start_matching(xml_path, n5_base_path, output_path)
        
        # Check our output dataset
        print_dataset_info(n5_base_path, prefix)

    except Exception as e:
        print(f"❌ Unexpected error in script execution: {e}")
        sys.exit(1)
