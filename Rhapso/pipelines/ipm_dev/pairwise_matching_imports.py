'''
Run in glue 5.0 with:
--additional-python-modules s3://PATH-TO-RHAPSO.whl,bioio==1.3.0,bioio-tifffile==1.0.0,tifffile==2025.1.10,opencv-python,scikit-learn,tensorstore==0.1.56
'''
#!/usr/bin/env python3
import sys
from Rhapso.matching.interest_point_matching import start_matching, fetch_xml_file, parse_xml, parse_and_read_datasets, perform_pairwise_matching, save_matches_as_n5

if __name__ == "__main__":
    try:
        # s3 examples
        xml_path = "s3://rhapso-fused-zarr-output-data/matching_input_sample/dataset.xml"
        n5_base_path = "s3://rhapso-fused-zarr-output-data/matching_input_sample/interestpoints.n5"
        output_path = n5_base_path  # Set output path to be the same as input n5 data

        # Multiple timepoints local example
        #xml_path = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/IP_TIFF_XML/2. IP_TIFF_XML (after matching)/dataset.xml"
        #n5_base_path = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/IP_TIFF_XML/2. IP_TIFF_XML (after matching)/interestpoints.n5"
        #output_path = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/IP_TIFF_XML/2. IP_TIFF_XML (after matching)/interestpoints.n5"
        
        start_matching(xml_path, n5_base_path, output_path)
    except Exception as e:
        print(f"❌ Unexpected error in script execution: {e}")
        sys.exit(1)
