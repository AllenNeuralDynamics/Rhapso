from Rhapso.matching.xml_parser import XMLParser
from Rhapso.matching.data_loader import DataLoader
from Rhapso.matching.matcher import Matcher
from Rhapso.matching.result_saver import ResultSaver

def main(xml_file, n5_folder_base, output_path):
    # Parse XML
    parser = XMLParser(xml_file)
    view_paths = parser.parse_view_paths()
    timepoints = parser.parse_timepoints()

    # Load data
    loader = DataLoader(n5_folder_base)
    interest_point_info = {}
    for view_key, view_info in view_paths.items():
        dataset_path = f"{view_info['path']}/beads/interestpoints/loc"
        interest_point_info[view_key] = loader.load_interest_points(dataset_path)

    # Perform matching
    matcher = Matcher()
    all_matches = []
    for viewA, pointsA in interest_point_info.items():
        for viewB, pointsB in interest_point_info.items():
            if viewA != viewB:
                matches = matcher.compute_matches(pointsA, pointsB)
                filtered_matches, _ = matcher.ransac_filter(pointsA, pointsB, matches)
                all_matches.extend([(viewA, viewB, m[0], m[1]) for m in filtered_matches])

    # Save results
    saver = ResultSaver(output_path)
    saver.save_matches(all_matches, view_paths)

if __name__ == "__main__":
    xml_file = "/path/to/dataset.xml"
    n5_folder_base = "/path/to/n5/folder"
    output_path = "/path/to/output"
    main(xml_file, n5_folder_base, output_path)
