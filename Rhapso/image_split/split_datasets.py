"""
python3 -m Rhapso.image_split.SplitDatasets
"""

# main file 

import sys
import numpy as np
from xml.etree import ElementTree as ET

from Rhapso.image_split.split_views import collect_image_sizes, find_min_step_size, closest_larger_long_divisable_by
from Rhapso.image_split.splitting_tools import split_images

def main():
    print("beginning image splitting...")
    
    # Set default values based on SplitDatasets.java
    xml_input = None 
    xml_output = None 
    target_image_size_string = None 
    target_overlap_string = None
    disable_optimization = False
    fake_interest_points = False
    fip_density = 100.0
    fip_min_num_points = 20
    fip_max_num_points = 500
    fip_error = 0.5
    fip_exclusion_radius = 20.0
    assign_illuminations = False
    display_result = False

    # CLI user input values - these would be replaced with proper argument parsing
    xml_input = "/mnt/c/Users/marti/Documents/allen/data/exaSPIM_686951 EXAMPLE/ip_affine_alignment/bigstitcher_affine.xml"
    xml_output = "/mnt/c/Users/marti/Documents/allen/data/exaSPIM_686951 EXAMPLE/results/bigstitcher_affine_split-RHAPSO.xml"
    target_image_size_string = "7000,7000,4000"
    target_overlap_string = "128,128,128"
    fake_interest_points = True
    fip_exclusion_radius = 200
    assign_illuminations = True
    
    # Register a namespace to handle the default namespace in BigStitcher XML
    try:
        ET.register_namespace('', 'SpimData.xsd')
    except AttributeError as e:
        print(f"Warning: Could not register namespace: {e}", file=sys.stderr)

    # Step 1: Load SpimData2 from XML input, set as dataGlobal
    print(f"Loading XML: {xml_input}")
    try:
        tree = ET.parse(xml_input)
        data_global = tree.getroot()  # This is equivalent to dataGlobal in Java
    except ET.ParseError as e:
        print(f"Error: Could not parse XML file. {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: XML file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error loading XML. {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Parse target values (targetImageSize and targetOverlap)
    try:
        target_image_size = np.array([int(s) for s in target_image_size_string.split(',')], dtype=np.int64)
        target_overlap = np.array([int(s) for s in target_overlap_string.split(',')], dtype=np.int64)
        
        if len(target_image_size) != 3 or len(target_overlap) != 3:
            raise ValueError("Target image size and overlap must contain exactly 3 values (x,y,z)")
        print(f"Target image size: {target_image_size}")
        print(f"Target overlap: {target_overlap}")
    except ValueError as e:
        print(f"Error: {e}. Target image size and overlap must be comma-separated integers.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error parsing target values. {e}", file=sys.stderr)
        sys.exit(1)

    # Step 3: Collect image sizes and print current dataset sizes
    try:
        img_sizes = collect_image_sizes(data_global)
        print("Current image sizes of dataset:")
        for size_str, count in img_sizes[0].items():
            print(f"{count}x: {size_str}")
    except KeyError as e:
        print(f"Error: XML structure missing expected elements for collecting image sizes. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to collect image sizes. {e}", file=sys.stderr)
        sys.exit(1)

    # Step 4: Find minimum step size
    try:
        min_step_size = find_min_step_size(data_global)
        print(f"Target image sizes and overlaps need to be divisible by {min_step_size}")
    except Exception as e:
        print(f"Error: Failed to find minimum step size. {e}", file=sys.stderr)
        sys.exit(1)

    # Step 5: Calculate adjusted sizes and overlaps
    try:
        sx = closest_larger_long_divisable_by(target_image_size[0], min_step_size[0])
        sy = closest_larger_long_divisable_by(target_image_size[1], min_step_size[1])
        sz = closest_larger_long_divisable_by(target_image_size[2], min_step_size[2])
        
        ox = closest_larger_long_divisable_by(target_overlap[0], min_step_size[0])
        oy = closest_larger_long_divisable_by(target_overlap[1], min_step_size[1])
        oz = closest_larger_long_divisable_by(target_overlap[2], min_step_size[2])

        print(f"Adjusted target image size: [{sx}, {sy}, {sz}]")
        print(f"Adjusted target overlap: [{ox}, {oy}, {oz}]")
    except Exception as e:
        print(f"Error: Failed to calculate adjusted sizes and overlaps. {e}", file=sys.stderr)
        sys.exit(1)

        print(f"Adjusted target image size: [{sx}, {sy}, {sz}]")
        print(f"Adjusted target overlap: [{ox}, {oy}, {oz}]")

    # Step 6: Check if overlap is bigger than size and exit if so
    if ox > sx or oy > sy or oz > sz:
        print("Error: Overlap cannot be bigger than size.", file=sys.stderr)
        sys.exit(1)

    # Step 7: Call splitImages to create new data
    try:
        print("\nStarting image splitting process...")
        new_data = split_images(
            data_global,
            [ox, oy, oz],
            [sx, sy, sz],
            min_step_size,
            assign_illuminations,
            not disable_optimization,
            fake_interest_points,
            fip_density,
            fip_min_num_points,
            fip_max_num_points,
            fip_error,
            fip_exclusion_radius
        )
        print("Splitting process finished.")
        if new_data is None:
            print("Error: split_images returned None. Aborting.", file=sys.stderr)
            sys.exit(1)
    except KeyError as e:
        print(f"Error: XML structure missing expected elements for splitting. {e}", file=sys.stderr)
        sys.exit(1)
    except MemoryError as e:
        print(f"Error: Insufficient memory for splitting operation. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed during image splitting process. {e}", file=sys.stderr)
        sys.exit(1)

    # Step 8: Save the new data to XML output file
    try:
        new_tree = ET.ElementTree(new_data)
        
        # Improved XML formatting
        def indent(elem, level=0):
            i = "\n" + level*"  "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "  "
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for elem in elem:
                    indent(elem, level+1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i
        
        # Use built-in indent for Python 3.9+, otherwise use our custom function
        if sys.version_info.major == 3 and sys.version_info.minor >= 9:
            ET.indent(new_tree, space="  ", level=0)
        else:
            indent(new_data)
        
        print(f"Saving new XML to: {xml_output}")
        # Use encoding and method parameters to ensure consistent formatting
        new_tree.write(
            xml_output, 
            encoding='utf-8', 
            xml_declaration=True, 
            method="xml", 
            short_empty_elements=True
        )
        print("Done.")
    except PermissionError as e:
        print(f"Error: Permission denied when saving output file. {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error: IO error when saving output file. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to save output XML. {e}", file=sys.stderr)
        sys.exit(1)

    print("Split-Images run finished")

if __name__ == '__main__':
    main()