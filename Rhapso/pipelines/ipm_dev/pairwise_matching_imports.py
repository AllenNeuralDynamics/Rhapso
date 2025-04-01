'''
Run in glue 5.0 with:
--additional-python-modules s3://rhapso-dev/rhapso-whl-package/branch=main/Rhapso-0.1.5-py3-none-any.whl,bioio==1.3.0,bioio-tifffile==1.0.0,tifffile==2025.1.10,opencv-python,scikit-learn,tensorstore==0.1.56
'''
#!/usr/bin/env python3
import sys
from Rhapso.matching.interest_point_matching import fetch_xml_file, parse_xml, parse_and_read_datasets, perform_pairwise_matching, save_matches_as_n5
from urllib.parse import urlparse
import boto3
import os

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

def main(xml_file, n5_folder_base, output_s3_path=None):
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

        if n5_folder_base.startswith("s3://"):
            n5_folder_base = fetch_n5_folder(n5_folder_base)
        
        # If output path is not specified, use the same as input n5 folder
        if output_s3_path is None:
            output_s3_path = n5_folder_base

        labels = ["beads"]
        method = "FAST_ROTATION"
        clear_correspondences = False

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
        save_matches_as_n5(all_matches, view_paths, output_s3_path, clear_correspondences)
        print(f"✅ Successfully saved matches to: {output_s3_path}")
    except Exception as e:
        print(f"❌ Error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        # # S3
        xml_s3_path = "s3://rhapso-fused-zarr-output-data/matching_input_sample/dataset.xml"
        n5_base_path = "s3://rhapso-fused-zarr-output-data/matching_input_sample/interestpoints.n5"
        output_path = n5_base_path  # Set output path to be the same as input n5 data

        #xml_s3_path = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/Just Two Tiff/IP_TIFF_XML (original) - Just Two Tiff Files - After Matching/dataset.xml"
        #n5_base_path = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/Just Two Tiff/IP_TIFF_XML (original) - Just Two Tiff Files - After Matching/interestpoints.n5"
        #output_path = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/Just Two Tiff/IP_TIFF_XML (original) - Just Two Tiff Files - After Matching/matchingOutput/interestpoints.n5"

        main(xml_s3_path, n5_base_path, output_path)
    except Exception as e:
        print(f"❌ Unexpected error in script execution: {e}")
        sys.exit(1)
