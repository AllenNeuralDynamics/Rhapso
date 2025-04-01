#!/usr/bin/env python3
from Rhapso.matching.interest_point_matching import parse_xml, parse_and_read_datasets, perform_pairwise_matching, save_matches_as_n5

def main(self):
    labels = ["beads"]
    method = "FAST_ROTATION"
    clear_correspondences = False

    interest_point_info, view_paths = parse_and_read_datasets(self.xml_file, self.n5_folder_base)
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
    save_matches_as_n5(all_matches, view_paths, self.n5_folder_base, clear_correspondences)

if __name__ == "__main__":
    xml_input_file = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/Just Two Tiff/IP_TIFF_XML (original) - Just Two Tiff Files - After Matching/dataset.xml"
    n5_base_path = "/home/martin/Documents/Allen/BigStitcherSpark Example Datasets/Interest Points (unaligned)/Just Two Tiff/IP_TIFF_XML (original) - Just Two Tiff Files - After Matching/interestpoints.n5"
    main(xml_input_file, n5_base_path)
