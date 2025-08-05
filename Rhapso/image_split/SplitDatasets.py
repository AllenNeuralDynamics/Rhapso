"""
python3 -m Rhapso.image_split.SplitDatasets
"""

# main file 

import sys
import numpy as np
from xml.etree import ElementTree as ET

from Rhapso.image_split.SplitViews import collect_image_sizes, find_min_step_size, closest_larger_long_divisable_by
from Rhapso.image_split.SplittingTools import split_images

def main():
    print("beginning image splitting...")
    # Default values based on SplitDatasets.java
    xml_input = None  # required in Java
    xml_output = None  # defaults to overwrite input in Java
    target_image_size_string = None  # required in Java
    target_overlap_string = None  # required in Java
    disable_optimization = False
    fake_interest_points = False
    fip_density = 100.0
    fip_min_num_points = 20
    fip_max_num_points = 500
    fip_error = 0.5
    fip_exclusion_radius = 20.0
    assign_illuminations = False
    display_result = False

    # # Set required values (these would normally come from command line)
    # xml_input = "/path/to/input.xml"  # Set your input path here
    # xml_output = "/path/to/output.xml"  # Set your output path here
    # target_image_size_string = "512,512,256"  # Set your target size here
    # target_overlap_string = "32,32,32"  # Set your target overlap here

    # # Register a namespace to handle the default namespace in BigStitcher XML
    # try:
    #     ET.register_namespace('', 'SpimData.xsd')
    # except AttributeError:
    #     pass

    # print(f"Loading XML: {xml_input}")
    # try:
    #     tree = ET.parse(xml_input)
    #     root = tree.getroot()
    # except ET.ParseError as e:
    #     print(f"Error: Could not parse XML file. {e}", file=sys.stderr)
    #     sys.exit(1)

    # # Parse target values
    # try:
    #     target_image_size = np.array([int(s) for s in target_image_size_string.split(',')], dtype=np.int64)
    #     target_overlap = np.array([int(s) for s in target_overlap_string.split(',')], dtype=np.int64)
    # except ValueError:
    #     print("Error: targetImageSize and targetOverlap must be comma-separated integers.", file=sys.stderr)
    #     sys.exit(1)

    # # --- Data Collection and Adjustment ---
    # sizes, min_img_size = sv_utils.collect_image_sizes(root)
    # print("Current image sizes of dataset:")
    # for size, count in sizes.items():
    #     print(f"{count}x: {size}")

    # min_step_size = sv_utils.find_min_step_size(root)
    # print(f"Target image sizes and overlaps need to be divisible by {min_step_size}")

    # # Adjust size and overlap to be divisible by min_step_size
    # adjusted_size = np.array([
    #     sv_utils.closest_larger_long_divisable_by(target_image_size[d], min_step_size[d]) for d in range(3)
    # ])
    # adjusted_overlap = np.array([
    #     sv_utils.closest_larger_long_divisable_by(target_overlap[d], min_step_size[d]) for d in range(3)
    # ])

    # print(f"Adjusted target image size: {adjusted_size}")
    # print(f"Adjusted target overlap: {adjusted_overlap}")

    # if np.any(adjusted_overlap > adjusted_size):
    #     print("Error: Overlap cannot be bigger than size.", file=sys.stderr)
    #     sys.exit(1)

    # # --- Splitting ---
    # print("\nStarting image splitting process...")
    # new_root = st.split_images(
    #     root=root,
    #     overlap=adjusted_overlap,
    #     target_size=adjusted_size,
    #     min_step_size=min_step_size,
    #     assign_illuminations=assign_illuminations,
    #     optimize=(not disable_optimization),
    # )
    # print("Splitting process finished.")

    # # --- Saving Output ---
    # new_tree = ET.ElementTree(new_root)
    # if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    #     ET.indent(new_tree, space="  ", level=0)
    
    # print(f"Saving new XML to: {xml_output}")
    # new_tree.write(xml_output, encoding='utf-8', xml_declaration=True)
    # print("Done.")


if __name__ == '__main__':
    main()